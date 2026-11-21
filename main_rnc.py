import argparse
import os
import logging
import numpy as np
import torch
import time
from sklearn.covariance import LedoitWolf
from dataset import *
from utils import *
from model import Encoder
from loss import MoRnCLoss, RnCLoss

print = logging.info

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--save_curr_freq', type=int, default=30, help='save curr last frequency')
    parser.add_argument(
        '--save_initial', action='store_true',
        help='save the randomly initialized encoder as ckpt_epoch_0.pth',
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=0,
        help='training batch size; 0 (default) uses the entire training set',
    )
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=400, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument(
        '--optimizer', choices=['adamw', 'sgd'], default='adamw',
        help='optimizer for encoder pretraining',
    )
    parser.add_argument('--trial', type=str, default='', help='id for recording multiple runs')
    parser.add_argument('--seed', type=int, default=42, help='random split and training seed')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='fraction used for training')
    parser.add_argument(
        '--val_ratio', type=float, default=0.0,
        help='fraction of the outer training partition reserved for validation',
    )
    parser.add_argument(
        '--split_file', type=str, default='',
        help='JSON file containing fixed train/val/test indices; overrides ratio splitting',
    )

    parser.add_argument('--data_folder', type=str, default='./data', help='path to custom dataset')
    parser.add_argument('--dataset', type=str, default='Atp7d', choices=get_dataset_names(), help='dataset')
    parser.add_argument('--model', type=str, default='mlp', choices=['resnet18', 'resnet50', 'mlp'])
    parser.add_argument(
        '--hidden_dims', '--hidden-dims', dest='hidden_dims', nargs='+', type=int,
        default=None,
        help='MLP hidden widths, for example: --hidden-dims 256 128',
    )
    parser.add_argument(
        '--feat_dim', '--feat-dim', dest='feat_dim', type=int, default=None,
        help='output representation dimension; default keeps the historical input dimension',
    )
    parser.add_argument('--resume', type=str, default='', help='resume ckpt path')
    parser.add_argument(
        '--aug',
        type=str,
        default='none',
        help=(
            'comma-separated image augmentations; the default uses only '
            'deterministic preprocessing, and tabular datasets always use none'
        ),
    )
    view_group = parser.add_mutually_exclusive_group()
    view_group.add_argument(
        '--two_view',
        dest='two_view',
        action='store_true',
        help='use the original RnC two-view training layout',
    )
    view_group.add_argument(
        '--single_view',
        dest='two_view',
        action='store_false',
        help='use one view per training sample',
    )
    parser.set_defaults(two_view=False)

    parser.add_argument(
        '--label_diff',
        type=str,
        default='l2',
        choices=['l1', 'l2', 'l2_mixed', 'lmmo_knn', 'yqy', 'mo_rnc'],
        help='pairwise ordering score used by RnC',
    )
    augmentation_group = parser.add_mutually_exclusive_group()
    augmentation_group.add_argument(
        '--mixup',
        action='store_true',
        help='use clean and Mixup samples as two tabular views',
    )
    augmentation_group.add_argument(
        '--gaussian_noise', '--gaussian',
        dest='gaussian_noise',
        action='store_true',
        help='use clean and Gaussian-noise samples as two tabular views',
    )
    parser.add_argument('--mixup_alpha', type=float, default=1.0,
                        help='Beta distribution shape for Mixup weights')
    parser.add_argument('--mixup_min_lambda', type=float, default=0.8,
                        help='minimum clean-sample weight in the Mixup view')
    parser.add_argument('--mixup_k', type=int, default=10,
                        help='number of label-L2 neighbors available for Mixup pairing')
    parser.add_argument('--noise_sigma', type=float, default=0.1,
                        help='Gaussian noise standard deviation after standardization')
    parser.add_argument('--local_weight', type=float, default=0.5,
                        help='lambda multiplying the dimension-wise RnC loss')
    parser.add_argument('--local_dims', type=int, default=2,
                        help='number of randomly sampled output dimensions per batch')
    parser.add_argument('--temp_alpha', type=float, default=1.0,
                        help='alpha in adaptive temperature tau_i=alpha*mean(d_i)+beta')
    parser.add_argument('--temp_beta', type=float, default=0.1,
                        help='beta in adaptive temperature tau_i=alpha*mean(d_i)+beta')
    parser.add_argument('--mahalanobis_ridge', '--mahalanobis-ridge',
                        dest='mahalanobis_ridge', type=float, default=0.1,
                        help='diagonal ridge used when fitting the fixed Mahalanobis matrix')
    parser.add_argument(
        '--mahalanobis_mode', '--mahalanobis-mode',
        dest='mahalanobis_mode', choices=['diagonal', 'full'],
        default='diagonal',
        help='use a diagonal or full fixed Mahalanobis precision matrix',
    )
    parser.add_argument(
        '--mahalanobis_shrinkage', '--mahalanobis-shrinkage',
        dest='mahalanobis_shrinkage', type=float, default=0.0,
        help='rho in (1-rho)Sigma + rho*mu*I, in [0, 1]',
    )
    parser.add_argument(
        '--mahalanobis_covariance', '--mahalanobis-covariance',
        dest='mahalanobis_covariance',
        choices=['empirical', 'ledoit_wolf'], default='empirical',
        help='empirical covariance or automatic Ledoit-Wolf shrinkage',
    )
    parser.add_argument(
        '--use_input_knn',
        action='store_true',
        help='restrict RnC positives and denominator candidates to input-x kNN',
    )

    parser.add_argument(
        '--knn_k',
        type=int,
        default=10,
        help='number of original-input x neighbors used by the RnC loss',
    )
    parser.add_argument(
        '--lmmo_k',
        dest='lmmo_k',
        type=int,
        default=10,
        help='number of neighbors used by LMMO metric pretraining',
    )
    parser.add_argument(
        '--lmmo_C',
        type=float,
        default=10.0,
        help='paper Eq. (8) squared-slack trade-off coefficient',
    )
    parser.add_argument(
        '--lmmo_lambda_X',
        type=float,
        default=0.1,
        help='official-implementation scaled-ridge coefficient for the LMMO input projection M',
    )
    parser.add_argument('--lmmo_max_iter', type=int, default=100,
                        help='maximum number of APG iterations')
    parser.add_argument('--lmmo_tol', type=float, default=1e-3,
                        help='LMMO relative objective-decrease tolerance (default: 1e-3)')
    parser.add_argument('--lmmo_backtracking', type=float, default=0.4,
                        help='APG backtracking factor in (0, 1)')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='',
        help='optional exact output directory',
    )

    opt = parser.parse_args()
    if is_sdf_dataset(opt.dataset) and opt.model in ['resnet18', 'resnet50']:
        opt.model = 'mlp'
    if is_sdf_dataset(opt.dataset):
        opt.aug = 'none'

    if opt.knn_k < 1:
        parser.error('--knn_k must be at least 1.')
    if opt.batch_size < 0:
        parser.error('--batch_size must be 0 (full training set) or a positive integer.')
    if opt.hidden_dims is not None and any(width < 1 for width in opt.hidden_dims):
        parser.error('--hidden-dims values must all be positive.')
    if opt.feat_dim is not None and opt.feat_dim < 1:
        parser.error('--feat-dim must be positive.')
    if opt.local_dims < 1:
        parser.error('--local_dims must be at least 1.')
    if opt.temp_alpha < 0 or opt.temp_beta <= 0:
        parser.error('--temp_alpha must be non-negative and --temp_beta positive.')
    if opt.mahalanobis_ridge < 0:
        parser.error('--mahalanobis_ridge must be non-negative.')
    if not 0 <= opt.mahalanobis_shrinkage <= 1:
        parser.error('--mahalanobis-shrinkage must be in [0, 1].')
    if opt.mixup_alpha <= 0:
        parser.error('--mixup_alpha must be positive.')
    if opt.mixup_k < 1:
        parser.error('--mixup_k must be at least 1.')
    if not 0.5 <= opt.mixup_min_lambda < 1:
        parser.error('--mixup_min_lambda must be in [0.5, 1).')
    if opt.noise_sigma < 0:
        parser.error('--noise_sigma must be non-negative.')
    if not 0 <= opt.val_ratio < 1:
        parser.error('--val_ratio must be in [0, 1).')
    if opt.mixup or opt.gaussian_noise:
        if not is_sdf_dataset(opt.dataset):
            parser.error('Tabular augmentation requires an SDF tabular dataset.')
        if not opt.two_view:
            parser.error('Tabular augmentation requires --two_view.')

    if opt.label_diff in ('lmmo_knn', 'yqy'):
        if not is_sdf_dataset(opt.dataset):
            parser.error('lmmo_knn and yqy require a vector-valued SDF tabular dataset.')

    opt.model_path = './save/{}_models'.format(opt.dataset)
    input_knn_tag = '_xknn{}'.format(opt.knn_k) \
        if opt.use_input_knn else '_original_rnc'
    loss_tag = '{}{}'.format(opt.label_diff, input_knn_tag)
    if opt.label_diff == 'mo_rnc':
        loss_tag += '_local{}_lambda{}'.format(
            opt.local_dims, opt.local_weight
        )
    loss_tag += '_at{}_{}'.format(opt.temp_alpha, opt.temp_beta)
    if opt.mixup:
        view_tag = 'two_view_label_l2_knn_mixup_k{}_a{}_min{}'.format(
            opt.mixup_k, opt.mixup_alpha, opt.mixup_min_lambda
        )
    elif opt.gaussian_noise:
        view_tag = 'two_view_gaussian_sigma{}'.format(opt.noise_sigma)
    else:
        view_tag = 'two_view' if opt.two_view else 'single_view'
    batch_size_tag = 'full' if opt.batch_size == 0 else opt.batch_size
    opt.model_name = 'RnC_{}_{}_ep_{}_lr_{}_d_{}_wd_{}_mmt_{}_bsz_{}_aug_{}_{}_label_{}_split_{}_seed_{}_trial_{}'. \
        format(opt.dataset, opt.model, opt.epochs, opt.learning_rate, opt.lr_decay_rate, opt.weight_decay, opt.momentum,
               batch_size_tag, opt.aug, view_tag, loss_tag,
               opt.train_ratio, opt.seed, opt.trial)
    if len(opt.resume):
        opt.model_name = opt.resume.split('/')[-2]

    opt.save_folder = opt.output_dir or os.path.join(opt.model_path, opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(opt.save_folder, 'training.log')),
            logging.StreamHandler()
        ])

    print(f"Run: {opt.model_name}")

    return opt

def set_loader(opt):
    train_transform = IdentityTransform() if is_sdf_dataset(opt.dataset) else get_transforms(split='train', aug=opt.aug)
    if opt.mixup:
        dataset_transform = FixedMixupTwoView(
            train_transform,
            k=opt.mixup_k,
            alpha=opt.mixup_alpha,
            min_lambda=opt.mixup_min_lambda,
            seed=opt.seed,
        )
    elif opt.gaussian_noise:
        dataset_transform = FixedGaussianNoiseTwoView(
            train_transform,
            sigma=opt.noise_sigma,
            seed=opt.seed,
        )
    else:
        dataset_transform = (
            TwoCropTransform(train_transform)
            if opt.two_view
            else train_transform
        )

    dataset_kwargs = dict(
        data_folder=opt.data_folder,
        transform=dataset_transform,
        split='train',
    )
    if is_sdf_dataset(opt.dataset):
        dataset_kwargs['save_dir'] = opt.save_folder
        dataset_kwargs['seed'] = opt.seed
        dataset_kwargs['train_ratio'] = opt.train_ratio
        dataset_kwargs['val_ratio'] = opt.val_ratio
        dataset_kwargs['split_file'] = opt.split_file or None
    train_dataset = globals()[opt.dataset](**dataset_kwargs)
    batch_size = len(train_dataset) if opt.batch_size == 0 else opt.batch_size
    print(f'Train samples: {len(train_dataset)}, batch size: {batch_size}')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True,
        drop_last=not is_sdf_dataset(opt.dataset))

    return train_loader

def prepare_lmmo_metric(opt, train_loader):
    if opt.label_diff not in ('lmmo_knn', 'yqy'):
        return None

    features = np.asarray(train_loader.dataset.features)
    labels = np.asarray(train_loader.dataset.labels)

    from metric_learning.lmmo import LMMO

    start = time.time()
    learner = LMMO(
        k=opt.lmmo_k,
        C=opt.lmmo_C,
        lambda_X=opt.lmmo_lambda_X,
        max_iter=opt.lmmo_max_iter,
        tol=opt.lmmo_tol,
        backtracking=opt.lmmo_backtracking,
    ).fit(features, labels)
    elapsed = time.time() - start

    metric = {
        'M': torch.from_numpy(np.asarray(learner.M_, dtype=np.float32)),
        'V': torch.from_numpy(np.asarray(learner.V_, dtype=np.float32)),
    }
    print(
        f'LMMO: {elapsed:.2f}s, {learner.n_iter_} iterations, '
        f'objective={learner.objective_history_[-1]:.6g}'
    )
    return metric

def prepare_fixed_mahalanobis(opt, train_loader):
    if opt.label_diff != 'mo_rnc':
        return None

    labels = np.asarray(train_loader.dataset.labels, dtype=np.float64)
    if opt.mahalanobis_covariance == 'ledoit_wolf':
        estimator = LedoitWolf(assume_centered=False).fit(labels)
        covariance = np.atleast_2d(estimator.covariance_)
        effective_shrinkage = float(estimator.shrinkage_)
    else:
        covariance = np.atleast_2d(np.cov(labels, rowvar=False))

        mean_variance = float(np.trace(covariance) / covariance.shape[0])
        spherical_covariance = mean_variance * np.eye(covariance.shape[0])
        covariance = (
            (1.0 - opt.mahalanobis_shrinkage) * covariance
            + opt.mahalanobis_shrinkage * spherical_covariance
        )
        effective_shrinkage = opt.mahalanobis_shrinkage
    covariance += opt.mahalanobis_ridge * np.eye(covariance.shape[0])
    matrix = np.linalg.pinv(covariance, hermitian=True)

    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    matrix = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T

    if opt.mahalanobis_mode == 'diagonal':
        matrix = np.diag(np.maximum(np.diag(matrix), 0.0))
    print(
        f'Fixed {opt.mahalanobis_mode} Mahalanobis metric: '
        f'dim={matrix.shape[0]}, covariance={opt.mahalanobis_covariance}, '
        f'ridge={opt.mahalanobis_ridge:g}, '
        f'shrinkage={effective_shrinkage:g}, '
        f'covariance_condition={np.linalg.cond(covariance):.6g}'
    )
    print(f'Final Mahalanobis matrix M:\n{matrix}')
    return torch.from_numpy(matrix.astype(np.float32))

def set_model(opt, lmmo_metric=None, mahalanobis_matrix=None):
    device = get_device()
    input_dim = get_input_dim(opt.dataset) if is_sdf_dataset(opt.dataset) else None
    model = Encoder(
        name=opt.model,
        input_dim=input_dim,
        hidden_dims=opt.hidden_dims,
        feat_dim=opt.feat_dim,
    )
    metric_kwargs = {}
    if lmmo_metric is not None:
        metric_kwargs = {
            'lmmo_M': lmmo_metric['M'],
            'lmmo_V': lmmo_metric['V'],
        }
    if opt.label_diff in ('mo_rnc', 'l2_mixed'):
        criterion = MoRnCLoss(
            mahalanobis_matrix=mahalanobis_matrix,
            global_distance='l2' if opt.label_diff == 'l2_mixed' else 'mahalanobis',
            local_weight=opt.local_weight,
            local_dims=opt.local_dims,
            temperature_alpha=opt.temp_alpha,
            temperature_beta=opt.temp_beta,
            input_knn_k=opt.knn_k if opt.use_input_knn else None,
        )
    else:
        criterion = RnCLoss(
            temperature_alpha=opt.temp_alpha,
            temperature_beta=opt.temp_beta,
            label_diff=opt.label_diff,
            input_knn_k=opt.knn_k if opt.use_input_knn else None,
            **metric_kwargs,
        )

    if device.type == 'cuda':
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
    model = model.to(device)
    criterion = criterion.to(device)

    return model, criterion

def train(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()
    device = next(model.parameters()).device

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    global_losses = AverageMeter()
    local_losses = AverageMeter()

    end = time.time()
    for idx, data_tuple in enumerate(train_loader):
        images, labels = data_tuple
        data_time.update(time.time() - end)
        bsz = labels.shape[0]
        if opt.two_view:
            images = torch.cat([images[0], images[1]], dim=0)

        images = images.to(device)
        labels = labels.to(device)

        features = model(images)
        if opt.two_view:
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = criterion(
            features,
            labels,
            inputs=images,
        )
        losses.update(loss.item(), bsz)
        if isinstance(criterion, MoRnCLoss):
            global_losses.update(criterion.last_global_loss.item(), bsz)
            local_losses.update(criterion.last_local_loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            to_print = 'Train: [{0}][{1}/{2}]\t' \
                       'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                       'DT {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                       'loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses
            )
            print(to_print)
            if isinstance(criterion, MoRnCLoss):
                print(
                    'Mo-RNC: global {:.5f}, local {:.5f}, '
                    'mean tau {:.5f}, dimensions {}'.format(
                        global_losses.val,
                        local_losses.val,
                        criterion.last_temperature.item(),
                        criterion.last_local_dims.tolist(),
                    )
                )

def main():
    opt = parse_option()
    set_seed(opt.seed)

    train_loader = set_loader(opt)
    lmmo_metric = prepare_lmmo_metric(opt, train_loader)
    mahalanobis_matrix = prepare_fixed_mahalanobis(opt, train_loader)

    model, criterion = set_model(
        opt,
        lmmo_metric=lmmo_metric,
        mahalanobis_matrix=mahalanobis_matrix,
    )
    if opt.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=opt.learning_rate,
            weight_decay=opt.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.epochs,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=opt.learning_rate,
            momentum=opt.momentum, weight_decay=opt.weight_decay,
        )
        scheduler = None

    start_epoch = 1
    if len(opt.resume):
        ckpt_state = torch.load(opt.resume)
        model.load_state_dict(ckpt_state['model'])
        optimizer.load_state_dict(ckpt_state['optimizer'])
        start_epoch = ckpt_state['epoch'] + 1
        print(f"<=== Epoch [{ckpt_state['epoch']}] Resumed from {opt.resume}!")

    if opt.save_initial and start_epoch == 1:
        save_model(
            model, optimizer, opt, 0,
            os.path.join(opt.save_folder, 'ckpt_epoch_0.pth'),
        )

    for epoch in range(start_epoch, opt.epochs + 1):
        if scheduler is None:
            adjust_learning_rate(opt, optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch, opt)
        if scheduler is not None:
            scheduler.step()

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

        if epoch % opt.save_curr_freq == 0:
            save_file = os.path.join(opt.save_folder, 'curr_last.pth')
            save_model(model, optimizer, opt, epoch, save_file)

    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

if __name__ == '__main__':
    main()
