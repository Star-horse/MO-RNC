import math
import random

import numpy as np
import torch
from dataset import SDF_DATASETS

try:
    from torchvision import transforms
except ModuleNotFoundError:
    transforms = None


def get_device():
    """Select CUDA, Apple Metal (MPS), or CPU in that order."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class FixedMixupTwoView:
    """Mix each sample with one of its label-space L2 nearest neighbors."""

    requires_dataset_binding = True
    returns_labels = True

    def __init__(self, transform, k=10, alpha=1.0, min_lambda=0.8, seed=0):
        self.transform = transform
        self.k = k
        self.alpha = alpha
        self.min_lambda = min_lambda
        self.seed = seed

    def bind(self, features, labels):
        if len(features) < 2:
            raise ValueError('Mixup requires at least two training samples.')

        rng = np.random.RandomState(self.seed)
        norms = (labels ** 2).sum(1)
        distances = (
            norms[:, None] + norms[None, :] - 2 * labels @ labels.T
        )
        np.fill_diagonal(distances, np.inf)
        k = min(self.k, len(features) - 1)
        neighbors = np.argpartition(distances, k - 1, axis=1)[:, :k]
        choices = rng.randint(k, size=len(features))
        partners = neighbors[np.arange(len(features)), choices]
        weights = self.min_lambda + (1.0 - self.min_lambda) * rng.beta(
            self.alpha, self.alpha, size=len(features)
        )

        self.features = features
        self.labels = labels
        self.partner_indices = partners
        self.mixed_features = (
            weights[:, None] * features
            + (1.0 - weights[:, None]) * features[partners]
        ).astype(np.float32)
        self.mixed_labels = (
            weights[:, None] * labels
            + (1.0 - weights[:, None]) * labels[partners]
        ).astype(np.float32)

    def __call__(self, index):
        views = [
            self.transform(self.features[index]),
            self.transform(self.mixed_features[index]),
        ]
        labels = np.stack([self.labels[index], self.mixed_labels[index]])
        return views, labels


class FixedGaussianNoiseTwoView:
    """Return one clean view and one reproducibly perturbed view."""

    requires_dataset_binding = True

    def __init__(self, transform, sigma=0.1, seed=0):
        self.transform = transform
        self.sigma = sigma
        self.seed = seed

    def bind(self, features, labels=None):
        rng = np.random.RandomState(self.seed)
        noise = rng.normal(0.0, self.sigma, size=features.shape)
        self.features = features
        self.noisy_features = (features + noise).astype(np.float32)

    def __call__(self, index):
        return [
            self.transform(self.features[index]),
            self.transform(self.noisy_features[index]),
        ]


class IdentityTransform:
    def __call__(self, x):
        return x


def is_sdf_dataset(dataset):
    return dataset in SDF_DATASETS


def get_dataset_names():
    return ['AgeDB'] + list(SDF_DATASETS.keys())


def get_transforms(split, aug):
    if transforms is None:
        raise ModuleNotFoundError("torchvision is required for image datasets/transforms, but SDF tabular datasets do not need it.")
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    if split == 'train':
        aug_list = aug.split(',')
        transforms_list = []

        if 'crop' in aug_list:
            transforms_list.append(transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)))
        else:
            transforms_list.append(transforms.Resize(256))
            transforms_list.append(transforms.CenterCrop(224))

        if 'flip' in aug_list:
            transforms_list.append(transforms.RandomHorizontalFlip())

        if 'color' in aug_list:
            transforms_list.append(transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8))

        if 'grayscale' in aug_list:
            transforms_list.append(transforms.RandomGrayscale(p=0.2))

        transforms_list.append(transforms.ToTensor())
        transforms_list.append(normalize)
        transform = transforms.Compose(transforms_list)
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    return transform


def get_label_dim(dataset):
    if dataset in ['AgeDB']:
        label_dim = 1
    elif dataset in SDF_DATASETS:
        cfg = SDF_DATASETS[dataset]
        label_dim = len(cfg['labels']) if 'labels' in cfg else cfg['label_dim']
    else:
        raise ValueError(dataset)
    return label_dim


def get_input_dim(dataset):
    if dataset not in SDF_DATASETS:
        raise ValueError(dataset)
    return SDF_DATASETS[dataset]['attr_dim'] - get_label_dim(dataset)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    eta_min = lr * (args.lr_decay_rate ** 3)
    lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_model(model, optimizer, opt, epoch, save_file):
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def set_optimizer(opt, model):
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate,
                                momentum=opt.momentum, weight_decay=opt.weight_decay)

    return optimizer
