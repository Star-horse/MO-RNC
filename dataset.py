import json
import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import arff
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils import data

SDF_DATASETS = {
    'Atp1d': {'file': 'atp1d.arff', 'label_dim': 6, 'attr_dim': 417},
    'Atp7d': {'file': 'atp7d.arff', 'label_dim': 6, 'attr_dim': 417},
    'Enb': {'file': 'enb.arff', 'labels': ['y1', 'y2'], 'attr_dim': 10},
    'Jura': {'file': 'jura.arff', 'labels': ['Cd', 'Co', 'Cu'], 'attr_dim': 18},
    'Oes10': {'file': 'oes10.arff', 'label_dim': 16, 'attr_dim': 314},
    'Oes97': {'file': 'oes97.arff', 'label_dim': 16, 'attr_dim': 279},
    'Osales': {'file': 'osales.arff', 'label_dim': 12, 'attr_dim': 413},
    'Sarcos': {'file': 'Sarcos.arff', 'label_dim': 7, 'attr_dim': 28},
    'Scpf': {'file': 'scpf.arff', 'labels': ['num_views', 'num_votes', 'num_comments'], 'attr_dim': 26},
    'slump': {'file': 'slump.arff', 'labels': ['SLUMP_cm', 'FLOW_cm', 'Compressive_Strength_Mpa'], 'attr_dim': 10},
    'rf1': {'file': 'rf1.arff', 'label_dim': 8, 'attr_dim': 72},
    'rf2': {'file': 'rf2.arff', 'label_dim': 8, 'attr_dim': 584},
    'scm1d': {'file': 'scm1d.arff', 'label_dim': 16, 'attr_dim': 296},
    'scm20d': {'file': 'scm20d.arff', 'label_dim': 16, 'attr_dim': 77},
    'sf1': {'file': 'sf1.arff', 'label_dim': 3, 'attr_dim': 13},
    'sf2': {'file': 'sf2.arff', 'label_dim': 3, 'attr_dim': 13},
}

def fixed_train_val_test_split(
    n, seed=0, train_ratio=0.8, val_ratio=0.0
):

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_train_val = int(train_ratio * n)
    n_val = int(val_ratio * n_train_val)
    n_train = n_train_val - n_val
    return (
        indices[:n_train],
        indices[n_train:n_train_val],
        indices[n_train_val:],
    )

def replace_nonfinite_with_train_median(x_train, *other_arrays):
    x_train = x_train.astype(np.float32, copy=True)
    finite_train = np.where(np.isfinite(x_train), x_train, np.nan)
    has_finite = np.isfinite(finite_train).any(axis=0)
    col_medians = np.zeros(finite_train.shape[1], dtype=np.float32)
    if has_finite.any():
        col_medians[has_finite] = np.nanmedian(finite_train[:, has_finite], axis=0).astype(np.float32)

    arrays = [x_train]
    arrays.extend(None if arr is None else arr.astype(np.float32, copy=True) for arr in other_arrays)
    cleaned = []
    for arr in arrays:
        if arr is None:
            cleaned.append(None)
            continue
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = np.take(col_medians, np.where(bad)[1])
        cleaned.append(arr)
    return tuple(cleaned)

class AgeDB(data.Dataset):
    def __init__(self, data_folder, transform=None, split='train'):
        df = pd.read_csv(f'./data/agedb.csv')
        self.df = df[df['split'] == split]
        self.split = split
        self.data_folder = data_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = np.asarray([row['age']]).astype(np.float32)
        img = Image.open(os.path.join(self.data_folder, row['path'])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, label

class SDFTabular(data.Dataset):
    def __init__(
        self,
        data_folder,
        name,
        transform=None,
        split='train',
        seed=0,
        train_ratio=0.8,
        val_ratio=0.0,
        save_dir=None,
        split_file=None,
    ):
        self.name = name
        self.split = split
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.transform = transform
        self.save_dir = os.path.abspath(save_dir) if save_dir else os.path.join(
            os.path.dirname(os.path.abspath(data_folder)), 'sdf_model', name
        )

        arff_path = os.path.join(data_folder, SDF_DATASETS[name]['file'])
        if not os.path.exists(arff_path):
            arff_path = os.path.join('SDF', 'Data', SDF_DATASETS[name]['file'])

        df = self._load_arff(arff_path)
        label_cols = self._resolve_label_columns(df, SDF_DATASETS[name])
        feature_cols = [col for col in df.columns if col not in label_cols]

        features = df[feature_cols].to_numpy(dtype=np.float32)
        labels = df[label_cols].to_numpy(dtype=np.float32)
        finite_labels = np.isfinite(labels).all(axis=1)
        if not finite_labels.all():
            features = features[finite_labels]
            labels = labels[finite_labels]

        if split_file:
            train_idx, val_idx, test_idx = self._load_split_file(
                split_file, len(labels)
            )
        else:

            train_idx, val_idx, test_idx = fixed_train_val_test_split(
                len(labels), seed=seed, train_ratio=train_ratio,
                val_ratio=val_ratio,
            )
        split_idx = self._select_split_indices(
            split, train_idx, val_idx, test_idx
        )
        self.split_indices = split_idx

        features_train, features_all = replace_nonfinite_with_train_median(features[train_idx], features)
        self.feature_scaler_path = os.path.join(self.save_dir, 'features_scaler.pkl')
        self.label_scaler_path = os.path.join(self.save_dir, 'labels_scaler.pkl')

        if split == 'train':
            self._fit_and_save_scalers(features_train, labels[train_idx])
            self.features = self.feature_scaler.transform(features_all)[split_idx].astype(np.float32)
            self.labels = self.label_scaler.transform(labels)[split_idx].astype(np.float32)
        else:
            self._load_scalers()
            self.features = self.feature_scaler.transform(features_all)[split_idx].astype(np.float32)
            self.labels = self.label_scaler.transform(labels)[split_idx].astype(np.float32)

        if getattr(self.transform, 'requires_dataset_binding', False):
            self.transform.bind(self.features, self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        y = self.labels[index]
        if getattr(self.transform, 'requires_dataset_binding', False):
            transformed = self.transform(index)
            if getattr(self.transform, 'returns_labels', False):
                return transformed
            x = transformed
        else:
            x = self.features[index]
            if self.transform is not None:
                x = self.transform(x)
        return x, y

    def _fit_and_save_scalers(self, features_train, labels_train):
        os.makedirs(self.save_dir, exist_ok=True)
        self.feature_scaler = StandardScaler().fit(features_train)
        self.label_scaler = StandardScaler().fit(labels_train)
        joblib.dump(self.feature_scaler, self.feature_scaler_path)
        joblib.dump(self.label_scaler, self.label_scaler_path)

    def _load_scalers(self):
        if not os.path.exists(self.feature_scaler_path):
            raise FileNotFoundError(
                f"Missing feature scaler for {self.name}: {self.feature_scaler_path}. "
                "Run training first so the scaler is saved."
            )
        if not os.path.exists(self.label_scaler_path):
            raise FileNotFoundError(
                f"Missing label scaler for {self.name}: {self.label_scaler_path}. "
                "Run training first so the scaler is saved."
            )
        self.feature_scaler = joblib.load(self.feature_scaler_path)
        self.label_scaler = joblib.load(self.label_scaler_path)

    @staticmethod
    def _load_arff(arff_path):
        raw, _ = arff.loadarff(arff_path)
        df = pd.DataFrame(raw)
        for col in df.columns:

            if df[col].dtype == object:
                df[col] = df[col].apply(lambda v: v.decode('utf-8') if isinstance(v, bytes) else v)
            df[col] = pd.to_numeric(df[col], errors='raise')
        return df

    @staticmethod
    def _resolve_label_columns(df, cfg):
        if 'labels' in cfg:
            lower_to_col = {col.lower(): col for col in df.columns}
            return [lower_to_col[col.lower()] for col in cfg['labels']]

        return list(df.columns[-cfg['label_dim']:])

    @staticmethod
    def _select_split_indices(split, train_idx, val_idx, test_idx):
        parts = {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx,
        }
        if split not in parts:
            raise ValueError(f'Unknown split: {split}.')
        if split == 'val' and len(val_idx) == 0:
            raise ValueError('Validation split requires val_ratio > 0.')
        return parts[split]

    @staticmethod
    def _load_split_file(split_file, sample_count):

        with open(split_file, 'r', encoding='utf-8') as handle:
            split_data = json.load(handle)

        missing = {'train', 'val', 'test'} - set(split_data)
        if missing:
            raise ValueError(
                f'Split file {split_file} is missing keys: {sorted(missing)}.'
            )

        parts = {
            name: np.asarray(split_data[name], dtype=np.int64)
            for name in ('train', 'val', 'test')
        }
        if len(parts['train']) == 0:
            raise ValueError('A fixed split must contain at least one training sample.')
        for name, indices in parts.items():
            if len(indices) != len(np.unique(indices)):
                raise ValueError(f'Duplicate indices found in split {name}.')
            if len(indices) and (indices.min() < 0 or indices.max() >= sample_count):
                raise ValueError(
                    f'Indices in split {name} are outside [0, {sample_count}).'
                )
        names = ('train', 'val', 'test')
        for left_pos, left in enumerate(names):
            for right in names[left_pos + 1:]:
                if np.intersect1d(parts[left], parts[right]).size:
                    raise ValueError(f'Splits {left} and {right} overlap.')
        return parts['train'], parts['val'], parts['test']

class Enb(SDFTabular):
    def __init__(self, data_folder='./SDF/Data', transform=None, split='train', seed=0, train_ratio=0.8, val_ratio=0.0, save_dir=None, split_file=None):
        super().__init__(data_folder, 'Enb', transform, split, seed, train_ratio, val_ratio, save_dir, split_file)

class Atp1d(SDFTabular):
    def __init__(self, data_folder='./SDF/Data', transform=None, split='train', seed=0, train_ratio=0.8, val_ratio=0.0, save_dir=None, split_file=None):
        super().__init__(data_folder, 'Atp1d', transform, split, seed, train_ratio, val_ratio, save_dir, split_file)

class Atp7d(SDFTabular):
    def __init__(self, data_folder='./SDF/Data', transform=None, split='train', seed=0, train_ratio=0.8, val_ratio=0.0, save_dir=None, split_file=None):
        super().__init__(data_folder, 'Atp7d', transform, split, seed, train_ratio, val_ratio, save_dir, split_file)

class Jura(SDFTabular):
    def __init__(self, data_folder='./SDF/Data', transform=None, split='train', seed=0, train_ratio=0.8, val_ratio=0.0, save_dir=None, split_file=None):
        super().__init__(data_folder, 'Jura', transform, split, seed, train_ratio, val_ratio, save_dir, split_file)

class Oes10(SDFTabular):
    def __init__(self, data_folder='./SDF/Data', transform=None, split='train', seed=0, train_ratio=0.8, val_ratio=0.0, save_dir=None, split_file=None):
        super().__init__(data_folder, 'Oes10', transform, split, seed, train_ratio, val_ratio, save_dir, split_file)

class Oes97(SDFTabular):
    def __init__(self, data_folder='./SDF/Data', transform=None, split='train', seed=0, train_ratio=0.8, val_ratio=0.0, save_dir=None, split_file=None):
        super().__init__(data_folder, 'Oes97', transform, split, seed, train_ratio, val_ratio, save_dir, split_file)

class Osales(SDFTabular):
    def __init__(self, data_folder='./SDF/Data', transform=None, split='train', seed=0, train_ratio=0.8, val_ratio=0.0, save_dir=None, split_file=None):
        super().__init__(data_folder, 'Osales', transform, split, seed, train_ratio, val_ratio, save_dir, split_file)

class Sarcos(SDFTabular):
    def __init__(self, data_folder='./SDF/Data', transform=None, split='train', seed=0, train_ratio=0.8, val_ratio=0.0, save_dir=None, split_file=None):
        super().__init__(data_folder, 'Sarcos', transform, split, seed, train_ratio, val_ratio, save_dir, split_file)

class Scpf(SDFTabular):
    def __init__(self, data_folder='./SDF/Data', transform=None, split='train', seed=0, train_ratio=0.8, val_ratio=0.0, save_dir=None, split_file=None):
        super().__init__(data_folder, 'Scpf', transform, split, seed, train_ratio, val_ratio, save_dir, split_file)

class slump(SDFTabular):
    def __init__(self, data_folder='./SDF/Data', transform=None, split='train', seed=0, train_ratio=0.8, val_ratio=0.0, save_dir=None, split_file=None):
        super().__init__(data_folder, 'slump', transform, split, seed, train_ratio, val_ratio, save_dir, split_file)

class rf1(SDFTabular):
    def __init__(self, data_folder='./SDF/Data', transform=None, split='train', seed=0, train_ratio=0.8, val_ratio=0.0, save_dir=None, split_file=None):
        super().__init__(data_folder, 'rf1', transform, split, seed, train_ratio, val_ratio, save_dir, split_file)

class rf2(SDFTabular):
    def __init__(self, data_folder='./SDF/Data', transform=None, split='train', seed=0, train_ratio=0.8, val_ratio=0.0, save_dir=None, split_file=None):
        super().__init__(data_folder, 'rf2', transform, split, seed, train_ratio, val_ratio, save_dir, split_file)

class scm1d(SDFTabular):
    def __init__(self, data_folder='./SDF/Data', transform=None, split='train', seed=0, train_ratio=0.8, val_ratio=0.0, save_dir=None, split_file=None):
        super().__init__(data_folder, 'scm1d', transform, split, seed, train_ratio, val_ratio, save_dir, split_file)

class scm20d(SDFTabular):
    def __init__(self, data_folder='./SDF/Data', transform=None, split='train', seed=0, train_ratio=0.8, val_ratio=0.0, save_dir=None, split_file=None):
        super().__init__(data_folder, 'scm20d', transform, split, seed, train_ratio, val_ratio, save_dir, split_file)

class sf1(SDFTabular):
    def __init__(self, data_folder='./SDF/Data', transform=None, split='train', seed=0, train_ratio=0.8, val_ratio=0.0, save_dir=None, split_file=None):
        super().__init__(data_folder, 'sf1', transform, split, seed, train_ratio, val_ratio, save_dir, split_file)

class sf2(SDFTabular):
    def __init__(self, data_folder='./SDF/Data', transform=None, split='train', seed=0, train_ratio=0.8, val_ratio=0.0, save_dir=None, split_file=None):
        super().__init__(data_folder, 'sf2', transform, split, seed, train_ratio, val_ratio, save_dir, split_file)
