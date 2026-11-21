MO-RNC: Learning Continuous Representations for Multi-Output Regression

This repository contains the implementation of MO-RNC, which combines correlation-aware global ranking with output-wise local ranking for multi-output regression.

## Loss Function

The `MoRnCLoss` class in `loss.py` takes features and standardized targets as input and returns the global–local rank-contrastive loss. For ATP1d:

```python
from loss import MoRnCLoss

# precision_matrix is estimated from standardized training targets.
criterion = MoRnCLoss(
    mahalanobis_matrix=precision_matrix,
    local_weight=0.01511934,
    local_dims=6,
    temperature_alpha=0,
    temperature_beta=0.5,
)

features = ...  # [bs, 2, feat_dim]
labels = ...    # [bs, 6]
loss = criterion(features, labels)
```

## Running

Install the required packages:

```bash
pip install torch numpy pandas scipy scikit-learn pillow joblib
```

Extract the dataset archive at the repository root so that the ARFF files are directly inside `./data`. If the archive is stored in `data`:

```bash
python -m zipfile -e ./data/MO-RNC_datasets.zip .
```

Run the main comparison:

```bash
python run_main_experiments.py
```

This runs RNC-L2, RNC-Mahalanobis, and MO-RNC on all seven datasets with seeds 0–9. The script uses the manuscript's selected architectures and hyperparameters, 80/20 splits, SGD, 400 encoder epochs, 100 linear-head epochs, an L1 prediction loss, and paired clean/Gaussian views with noise standard deviation 0.01. Both Mahalanobis variants use the full precision matrix; RNC-Mahalanobis disables the local loss.

To run MO-RNC on ATP1d with seed 0 only:

```bash
python run_main_experiments.py --datasets Atp1d --methods mo_rnc --seeds 0
```

Checkpoints and per-run test metrics are saved under `./results/main_experiments`. `summary.csv` reports mean RMSE and sample standard deviation across seeds; `commands.json` records the commands. Use a new `--output-dir` when repeating an experiment.
