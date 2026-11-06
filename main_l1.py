import argparse
import math
import os
import logging
import numpy as np
import torch
import time
from model import Encoder, model_dict
from dataset import *
from utils import *

print = logging.info


class RMSELoss(torch.nn.Module):
    """Per-sample multivariate RMSE, matching main_linear.py."""

    def forward(self, prediction, target):
        return torch.linalg.vector_norm(prediction - target) / math.sqrt(target.shape[0])


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--save_curr_freq', type=int, default=1, help='save curr last frequency')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=0,
        help='batch size; 0 (default) uses the entire split as one batch',
    )
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=400, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')
    parser.add_argument('--seed', type=int, default=0, help='random split and training seed')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='fraction used for training')

    parser.add_argument('--data_folder', type=str, default='./data', help='path to custom dataset')
    parser.add_argument('--dataset', type=str, default='AgeDB', choices=get_dataset_names(), help='dataset')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'mlp'])
    parser.add_argument('--resume', type=str, default='', help='resume ckpt path')
    parser.add_argument('--aug', type=str, default='crop,flip,color,grayscale', help='augmentations')

    opt = parser.parse_args()
    if opt.batch_size < 0:
        parser.error('--batch_size must be 0 (full batch) or a positive integer.')
    if is_sdf_dataset(opt.dataset) and opt.model in ['resnet18', 'resnet50']:
        opt.model = 'mlp'
    if is_sdf_dataset(opt.dataset):
        opt.aug = 'none'

    opt.model_path = './save/{}_models'.format(opt.dataset)
    batch_size_tag = 'full' if opt.batch_size == 0 else opt.batch_size
    opt.model_name = 'RMSE_{}_{}_ep_{}_lr_{}_d_{}_wd_{}_mmt_{}_bsz_{}_aug_{}_split_{}_seed_{}_trial_{}'. \
        format(opt.dataset, opt.model, opt.epochs, opt.learning_rate, opt.lr_decay_rate, opt.weight_decay, opt.momentum,
               batch_size_tag, opt.aug, opt.train_ratio, opt.seed, opt.trial)
    if len(opt.resume):
        opt.model_name = opt.resume.split('/')[-2]

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(opt.save_folder, 'training.log')),
            logging.StreamHandler()
        ])

    return opt


def set_loader(opt):
    train_transform = IdentityTransform() if is_sdf_dataset(opt.dataset) else get_transforms(split='train', aug=opt.aug)
    eval_transform = IdentityTransform() if is_sdf_dataset(opt.dataset) else get_transforms(split='val', aug=opt.aug)

    train_kwargs = dict(
        data_folder=opt.data_folder,
        transform=train_transform,
        split='train',
    )
    test_kwargs = dict(
        data_folder=opt.data_folder,
        transform=eval_transform,
        split='test',
    )
    if is_sdf_dataset(opt.dataset):
        sdf_kwargs = dict(
            save_dir=opt.save_folder,
            seed=opt.seed,
            train_ratio=opt.train_ratio,
        )
        train_kwargs.update(sdf_kwargs)
        test_kwargs.update(sdf_kwargs)

    train_dataset = globals()[opt.dataset](**train_kwargs)
    test_dataset = globals()[opt.dataset](**test_kwargs)
    val_dataset = None if is_sdf_dataset(opt.dataset) else globals()[opt.dataset](
        data_folder=opt.data_folder,
        transform=eval_transform,
        split='val',
    )

    if val_dataset is None:
        print(f'Train set size: {train_dataset.__len__()}\tTest set size: {test_dataset.__len__()}')
    else:
        print(f'Train set size: {train_dataset.__len__()}\t'
              f'Val set size: {val_dataset.__len__()}\t'
              f'Test set size: {test_dataset.__len__()}')

    train_batch_size = len(train_dataset) if opt.batch_size == 0 else opt.batch_size
    val_batch_size = (
        None if val_dataset is None
        else len(val_dataset) if opt.batch_size == 0 else opt.batch_size
    )
    test_batch_size = len(test_dataset) if opt.batch_size == 0 else opt.batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True
    )
    val_loader = None if val_dataset is None else torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def set_model(opt):
    device = get_device()
    criterion = RMSELoss()
    if is_sdf_dataset(opt.dataset):
        model = None
        dim_in = get_input_dim(opt.dataset)
    else:
        model = Encoder(name=opt.model)
        dim_in = model_dict[opt.model][1]

    dim_out = get_label_dim(opt.dataset)
    regressor = torch.nn.Linear(dim_in, dim_out)

    if device.type == 'cuda':
        if model is not None and torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
    if model is not None:
        model = model.to(device)
    regressor = regressor.to(device)
    criterion = criterion.to(device)

    return model, regressor, criterion


def train(train_loader, model, regressor, criterion, optimizer, epoch, opt):
    if model is not None:
        model.train()
    regressor.train()
    device = next(regressor.parameters()).device

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        bsz = labels.shape[0]

        images = images.to(device)
        labels = labels.to(device)

        features = images if model is None else model(images)
        output = regressor(features)

        loss = criterion(output, labels)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            to_print = 'Train: [{0}][{1}/{2}]\t'\
                       'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                       'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                       'rmse {loss.val:.5f} ({loss.avg:.5f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses
            )
            print(to_print)


def evaluate_rmse(data_loader, model, regressor):
    if model is not None:
        model.eval()
    regressor.eval()
    device = next(regressor.parameters()).device
    predictions = []
    targets = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            features = images if model is None else model(images)
            output = regressor(features)
            predictions.append(output.detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    label_scaler = getattr(data_loader.dataset, 'label_scaler', None)
    if label_scaler is not None:
        prediction = label_scaler.inverse_transform(prediction)
        target = label_scaler.inverse_transform(target)

    residual = prediction.astype(np.float64) - target.astype(np.float64)
    return float(np.linalg.norm(residual, ord='fro') / math.sqrt(len(target)))


def save_rmse_model(model, regressor, optimizer, opt, epoch, save_file):
    state = {
        'opt': opt,
        'model': None if model is None else model.state_dict(),
        'regressor': regressor.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def main():
    opt = parse_option()
    set_seed(opt.seed)

    # build data loader
    train_loader, val_loader, test_loader = set_loader(opt)

    # build model and criterion
    model, regressor, criterion = set_model(opt)

    # build optimizer
    predictor = regressor if model is None else torch.nn.ModuleList([model, regressor])
    optimizer = set_optimizer(opt, predictor)

    start_epoch = 1
    if len(opt.resume):
        ckpt_state = torch.load(opt.resume)
        if model is not None:
            model.load_state_dict(ckpt_state['model'])
        regressor.load_state_dict(ckpt_state['regressor'])
        optimizer.load_state_dict(ckpt_state['optimizer'])
        start_epoch = ckpt_state['epoch'] + 1
        print(f"<=== Epoch [{ckpt_state['epoch']}] Resumed from {opt.resume}!")

    best_error = 1e5
    save_file_best = os.path.join(opt.save_folder, 'best.pth')

    # training routine
    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        train(train_loader, model, regressor, criterion, optimizer, epoch, opt)

        if val_loader is not None:
            valid_error = evaluate_rmse(val_loader, model, regressor)
            is_best = valid_error < best_error
            best_error = min(valid_error, best_error)
            print(f"Val RMSE: {valid_error:.3f}, best: {best_error:.3f}")
        else:
            is_best = epoch == opt.epochs

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_rmse_model(model, regressor, optimizer, opt, epoch, save_file)

        if epoch % opt.save_curr_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'curr_last.pth'.format(epoch=epoch))
            save_rmse_model(model, regressor, optimizer, opt, epoch, save_file)

        if is_best:
            state = {
                'epoch': epoch,
                'model': None if model is None else model.state_dict(),
                'regressor': regressor.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            if val_loader is not None:
                state['best_error'] = best_error
            torch.save(state, save_file_best)

    checkpoint = torch.load(save_file_best)
    if model is not None:
        model.load_state_dict(checkpoint['model'])
    regressor.load_state_dict(checkpoint['regressor'])
    test_loss = evaluate_rmse(test_loader, model, regressor)
    print('Test RMSE: {:.6g}'.format(test_loss))


if __name__ == '__main__':
    main()
