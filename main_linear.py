import argparse
import json
import math
import os
import logging
import numpy as np
import torch
import time
from model import Encoder
from dataset import *
from utils import *

print = logging.info


class RMSELoss(torch.nn.Module):
    def forward(self, prediction, target):
        return torch.linalg.vector_norm(prediction - target) / math.sqrt(target.shape[0])


def load_checkpoint(path, map_location='cpu'):
    """Load this project's trusted checkpoints across PyTorch versions."""
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        # Older PyTorch releases do not support the weights_only argument.
        return torch.load(path, map_location=map_location)


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=0,
        help='training batch size; 0 (default) uses the entire training set',
    )
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument(
        '--optimizer', choices=['adamw', 'sgd'], default='adamw',
        help='optimizer for frozen linear regression',
    )
    parser.add_argument(
        '--predictor-loss', '--prediction-loss',
        dest='predictor_loss',
        choices=['rmse', 'l1', 'mse', 'huber'],
        default='rmse',
        help=(
            'training loss for the frozen linear prediction head; '
            'model selection and reported metrics remain RMSE'
        ),
    )
    parser.add_argument(
        '--huber-delta', type=float, default=1.0,
        help='transition point used only when --predictor-loss huber',
    )
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')
    parser.add_argument('--seed', type=int, default=42, help='random split and training seed')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='fraction used for training')
    parser.add_argument(
        '--val_ratio', type=float, default=0.0,
        help='fraction of the outer training partition reserved for validation',
    )
    parser.add_argument(
        '--evaluation_split',
        choices=['val', 'outer_val', 'test'], default='test',
        help='split written to result_file after linear training',
    )
    parser.add_argument(
        '--split_file', type=str, default='',
        help='JSON file containing fixed train/val/test indices; overrides ratio splitting',
    )
    parser.add_argument(
        '--fixed_epoch_evaluation', action='store_true',
        help='evaluate the predictor at the final epoch instead of selecting an epoch on validation data',
    )

    parser.add_argument('--data_folder', type=str, default='./data', help='path to custom dataset')
    parser.add_argument('--dataset', type=str, default='Atp1d', choices=get_dataset_names(), help='dataset')
    parser.add_argument('--model', type=str, default='mlp', choices=['resnet18', 'resnet50', 'mlp'])
    parser.add_argument(
        '--hidden_dims', '--hidden-dims', dest='hidden_dims', nargs='+', type=int,
        default=None,
        help='MLP hidden widths used by the encoder checkpoint',
    )
    parser.add_argument(
        '--feat_dim', '--feat-dim', dest='feat_dim', type=int, default=None,
        help='representation dimension used by the encoder checkpoint',
    )
    parser.add_argument('--resume', type=str, default='', help='resume ckpt path')
    parser.add_argument('--aug', type=str, default='crop,flip,color,grayscale', help='augmentations')

    parser.add_argument('--ckpt', type=str, default=r'C:\Users\Duardian\Documents\MO-RNC\save\Atp1d_mahalanobis_only\ckpt_epoch_400.pth', help='path to the trained encoder')
    parser.add_argument(
        '--result_file',
        type=str,
        default='',
        help='optional JSON path for test metrics',
    )
    parser.add_argument(
        '--output_dir', type=str, default='',
        help='optional directory for predictor logs, scalers, and checkpoints',
    )

    opt = parser.parse_args()
    if opt.batch_size < 0:
        parser.error('--batch_size must be 0 (full training set) or a positive integer.')
    if opt.hidden_dims is not None and any(width < 1 for width in opt.hidden_dims):
        parser.error('--hidden-dims values must all be positive.')
    if opt.feat_dim is not None and opt.feat_dim < 1:
        parser.error('--feat-dim must be positive.')
    if opt.huber_delta <= 0:
        parser.error('--huber-delta must be positive.')
    if not 0 <= opt.val_ratio < 1:
        parser.error('--val_ratio must be in [0, 1).')
    if opt.evaluation_split == 'val' and opt.val_ratio == 0 and not opt.split_file:
        parser.error('--evaluation_split val requires --val_ratio > 0.')
    opt.ckpt = os.path.abspath(opt.ckpt)
    if is_sdf_dataset(opt.dataset) and opt.model in ['resnet18', 'resnet50']:
        opt.model = 'mlp'
    if is_sdf_dataset(opt.dataset):
        opt.aug = 'none'

    opt.model_name = 'Reg_seed_{}_{}'.format(opt.seed, opt.trial)
    if len(opt.resume):
        opt.model_name = opt.resume.split('/')[-1][:-len('_last.pth')]
    opt.save_folder = opt.output_dir or os.path.dirname(opt.ckpt)
    os.makedirs(opt.save_folder, exist_ok=True)

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(opt.save_folder, f'{opt.model_name}.log')),
            logging.StreamHandler()
        ]
    )

    return opt


def set_loader(opt):
    train_transform = IdentityTransform() if is_sdf_dataset(opt.dataset) else get_transforms(split='train', aug=opt.aug)
    eval_transform = IdentityTransform() if is_sdf_dataset(opt.dataset) else get_transforms(split='val', aug=opt.aug)

    dataset_kwargs = {}
    if is_sdf_dataset(opt.dataset):
        dataset_kwargs = {
            'seed': opt.seed,
            'train_ratio': opt.train_ratio,
            'val_ratio': opt.val_ratio,
            'save_dir': opt.save_folder,
            'split_file': opt.split_file or None,
        }
    train_dataset = globals()[opt.dataset](
        data_folder=opt.data_folder,
        transform=train_transform,
        split='train',
        **dataset_kwargs,
    )
    test_dataset = globals()[opt.dataset](
        data_folder=opt.data_folder,
        transform=eval_transform,
        split='test',
        **dataset_kwargs,
    )
    val_dataset = None
    fixed_val_size = 0
    if opt.split_file:
        with open(opt.split_file, 'r', encoding='utf-8') as handle:
            fixed_val_size = len(json.load(handle).get('val', []))
    if not is_sdf_dataset(opt.dataset) or opt.val_ratio > 0 or fixed_val_size > 0:
        val_dataset = globals()[opt.dataset](
            data_folder=opt.data_folder,
            transform=eval_transform,
            split='val',
            **dataset_kwargs,
        )

    if val_dataset is None:
        print(f'Train set size: {train_dataset.__len__()}\tTest set size: {test_dataset.__len__()}')
    else:
        print(f'Train set size: {train_dataset.__len__()}\t'
              f'Val set size: {val_dataset.__len__()}\t'
              f'Test set size: {test_dataset.__len__()}')

    batch_size = len(train_dataset) if opt.batch_size == 0 else opt.batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True
    )
    val_loader = None if val_dataset is None else torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def set_model(opt):
    device = get_device()
    input_dim = get_input_dim(opt.dataset) if is_sdf_dataset(opt.dataset) else None
    model = Encoder(
        name=opt.model,
        input_dim=input_dim,
        hidden_dims=opt.hidden_dims,
        feat_dim=opt.feat_dim,
    )
    criterion = {
        'rmse': RMSELoss,
        'l1': torch.nn.L1Loss,
        'mse': torch.nn.MSELoss,
        'huber': lambda: torch.nn.HuberLoss(delta=opt.huber_delta),
    }[opt.predictor_loss]()

    dim_in = model.output_dim
    dim_out = get_label_dim(opt.dataset)
    regressor = torch.nn.Linear(dim_in, dim_out)
    ckpt = load_checkpoint(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        model.encoder = torch.nn.DataParallel(model.encoder)
    elif torch.cuda.device_count() <= 1:
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    model = model.to(device)
    regressor = regressor.to(device)
    criterion = criterion.to(device)

    return model, regressor, criterion


def train(train_loader, model, regressor, criterion, optimizer, epoch, opt):
    model.eval()
    regressor.train()
    device = next(regressor.parameters()).device

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.to(device)
        labels = labels.to(device)
        bsz = labels.shape[0]

        with torch.no_grad():
            features = model(images)

        output = regressor(features.detach())
        loss = criterion(output, labels)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  '{loss_name} {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,
                loss_name=opt.predictor_loss))


def evaluate_rmse(test_loader, model, regressor):
    model.eval()
    regressor.eval()
    device = next(regressor.parameters()).device
    predictions = []
    targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            output = regressor(model(images))
            predictions.append(output.detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    label_scaler = getattr(test_loader.dataset, 'label_scaler', None)
    if label_scaler is not None:
        prediction = label_scaler.inverse_transform(prediction)
        target = label_scaler.inverse_transform(target)

    residual = prediction.astype(np.float64) - target.astype(np.float64)
    return float(np.linalg.norm(residual, ord='fro') / math.sqrt(len(target)))


def main():
    opt = parse_option()
    set_seed(opt.seed)

    train_loader, val_loader, test_loader = set_loader(opt)
    model, regressor, criterion = set_model(opt)

    if opt.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            regressor.parameters(), lr=opt.learning_rate,
            weight_decay=opt.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.epochs,
        )
    else:
        optimizer = torch.optim.SGD(
            regressor.parameters(), lr=opt.learning_rate,
            momentum=opt.momentum, weight_decay=opt.weight_decay,
        )
        scheduler = None

    save_file_best = os.path.join(opt.save_folder, f"{opt.model_name}_best.pth")
    save_file_last = os.path.join(opt.save_folder, f"{opt.model_name}_last.pth")
    best_error = 1e5

    start_epoch = 1
    if len(opt.resume):
        ckpt_state = load_checkpoint(opt.resume, map_location='cpu')
        regressor.load_state_dict(ckpt_state['state_dict'])
        start_epoch = ckpt_state['epoch'] + 1
        best_error = ckpt_state.get('best_error', best_error)
        print(f"<=== Epoch [{ckpt_state['epoch']}] Resumed from {opt.resume}!")


    # training routine
    for epoch in range(start_epoch, opt.epochs + 1):
        if scheduler is None:
            adjust_learning_rate(opt, optimizer, epoch)
        # train for one epoch
        train(train_loader, model, regressor, criterion, optimizer, epoch, opt)
        if scheduler is not None:
            scheduler.step()

        if val_loader is not None:
            valid_error = evaluate_rmse(val_loader, model, regressor)
            if opt.fixed_epoch_evaluation:
                is_best = epoch == opt.epochs
            else:
                is_best = valid_error < best_error
                best_error = min(valid_error, best_error)
            print(f"Val RMSE: {valid_error:.3f}, best: {best_error:.3f}")
        else:
            valid_error = None
            is_best = epoch == opt.epochs

        if is_best:
            state = {
                'epoch': epoch,
                'state_dict': regressor.state_dict(),
            }
            if val_loader is not None:
                state['best_error'] = best_error
            torch.save(state, save_file_best)

        torch.save({
            'epoch': epoch,
            'state_dict': regressor.state_dict(),
            'last_error': valid_error
        }, save_file_last)

    checkpoint = load_checkpoint(save_file_best, map_location='cpu')
    regressor.load_state_dict(checkpoint['state_dict'])
    evaluation_loader = (
        val_loader if opt.evaluation_split == 'val' else test_loader
    )
    evaluation_rmse = evaluate_rmse(
        evaluation_loader, model, regressor
    )
    print('{} RMSE: {:.6g}'.format(
        opt.evaluation_split.capitalize(), evaluation_rmse
    ))

    result = {
        'seed': opt.seed,
        'train_size': len(train_loader.dataset),
        'val_size': 0 if val_loader is None else len(val_loader.dataset),
        'test_size': len(test_loader.dataset),
        'metric_split': opt.evaluation_split,
        'rmse': evaluation_rmse,
        'predictor_loss': opt.predictor_loss,
        'huber_delta': (
            opt.huber_delta if opt.predictor_loss == 'huber' else None
        ),
    }
    # Keep the default basename short: experiment folders already encode the
    # full configuration and can approach Windows' traditional MAX_PATH limit.
    result_file = opt.result_file or os.path.join(
        opt.save_folder, f'test_seed{opt.seed}_{opt.trial}.json'
    )
    os.makedirs(os.path.dirname(os.path.abspath(result_file)), exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(f'Test metrics saved to {os.path.abspath(result_file)}')
    return result


if __name__ == '__main__':
    main()
