MO-RNC: Learning Continuous Representations for Multi-Output Regression

This repository contains the implementation of MO-RNC, which combines correlation-aware global ranking with output-wise local ranking for multi-output regression.

## Loss Function

The loss function `MoRnCLoss` in `loss.py` takes `features` and `labels` as input and returns the loss value.

```python
from loss import MoRnCLoss

# precision_matrix: inverse regularized covariance of standardized training targets
criterion = MoRnCLoss(
    mahalanobis_matrix=precision_matrix,
    local_weight=0.5,
    local_dims=6,
    temperature_alpha=0,
    temperature_beta=0.5,
)

features = ...  # [bs, 2, feat_dim]
labels = ...    # [bs, 6], standardized targets

loss = criterion(features, labels)
```

## Running

Use a Python environment with PyTorch, NumPy, pandas, SciPy, scikit-learn, Pillow, and joblib installed.

Extract `MO-RNC_datasets.zip` into the repository root so that the ARFF files are located in `./data`.

To train MO-RNC on ATP1d, first run:

```bash
python main_rnc.py --dataset Atp1d --label_diff mo_rnc --local_dims 6 \
  --temp_alpha 0 --temp_beta 0.5 --output_dir ./save/Atp1d
```

The encoder checkpoint will be saved to `./save/Atp1d/last.pth`. Then run:

```bash
python main_linear.py --dataset Atp1d --predictor-loss l1 \
  --ckpt ./save/Atp1d/last.pth
```

This freezes the encoder and trains a linear multi-output regressor.
