import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

CONFIGS = {
    'Atp1d': ([256, 256], 256, 6, 0, 1.0, 0.5, 0.5, 0.01511934),
    'Atp7d': ([256, 256, 128], 64, 6, 0, 0.5, 0.5, 0.7, 0.00063437),
    'Oes10': ([256], 256, 16, 0, 1.0, 0.5, 0.6, 0.01118548),
    'Oes97': ([256, 256], 256, 16, 0, 0.5, 0.5, 0.4, 0.07029119),
    'scm1d': ([256, 256, 128], 128, 16, 256, 1.0, 0.5, 0.9, 0.00495016),
    'scm20d': ([256, 256, 128], 64, 16, 256, 1.0, 0.5, 0.8, 0.00100000),
    'Sarcos': ([256, 256], 256, 7, 256, 0.5, 0.5, 0.4, 0.00055917),
}
FILES = {
    'Atp1d': 'atp1d.arff', 'Atp7d': 'atp7d.arff',
    'Oes10': 'oes10.arff', 'Oes97': 'oes97.arff',
    'scm1d': 'scm1d.arff', 'scm20d': 'scm20d.arff', 'Sarcos': 'Sarcos.arff',
}
METHODS = ('rnc_l2', 'rnc_mahalanobis', 'mo_rnc')
ROOT = Path(__file__).resolve().parent

def parse_args():
    parser = argparse.ArgumentParser(description="Run the fixed configurations in the manuscript's main RNC comparison.")
    parser.add_argument('--datasets', nargs='+', choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(10)))
    parser.add_argument('--data-folder', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/main_experiments')
    parser.add_argument('--encoder-epochs', type=int, default=400)
    parser.add_argument('--linear-epochs', type=int, default=100)
    parser.add_argument('--dry-run', action='store_true', help='print the command plan without training')
    args = parser.parse_args()
    if min(args.encoder_epochs, args.linear_epochs) < 1:
        parser.error('epoch counts must be positive')
    for name in ('datasets', 'methods', 'seeds'):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            parser.error(f'--{name} must not contain duplicates')
    args.data_folder = args.data_folder.resolve()
    args.output_dir = args.output_dir.resolve()
    return args

def build_plan(args):
    plan = []
    for dataset in args.datasets:
        hidden, feat, outputs, batch, tau_l2, tau_mo, rho, weight = CONFIGS[dataset]
        for method in args.methods:
            for seed in args.seeds:
                folder = args.output_dir / dataset / method / f'seed_{seed}'
                common = [
                    '--dataset', dataset, '--data_folder', str(args.data_folder),
                    '--model', 'mlp', '--seed', str(seed), '--train_ratio', '0.8',
                    '--val_ratio', '0', '--optimizer', 'sgd',
                    '--learning_rate', '0.05', '--momentum', '0.9',
                    '--weight_decay', '1e-4', '--batch_size', str(batch),
                    '--hidden-dims', *map(str, hidden), '--feat-dim', str(feat),
                    '--output_dir', str(folder),
                ]
                encoder = [sys.executable, str(ROOT / 'main_rnc.py'), *common,
                    '--epochs', str(args.encoder_epochs), '--lr_decay_rate', '0.1',
                    '--label_diff', 'l2' if method == 'rnc_l2' else 'mo_rnc',
                    '--local_dims', str(outputs),
                    '--local_weight', str(weight if method == 'mo_rnc' else 0),
                    '--temp_alpha', '0', '--temp_beta', str(tau_l2 if method == 'rnc_l2' else tau_mo),
                    '--mahalanobis-mode', 'full', '--mahalanobis-covariance', 'empirical',
                    '--mahalanobis-shrinkage', str(rho), '--mahalanobis-ridge', '1e-6',
                    '--two_view', '--gaussian_noise', '--noise_sigma', '0.01',
                ]
                linear = [sys.executable, str(ROOT / 'main_linear.py'), *common,
                    '--epochs', str(args.linear_epochs), '--lr_decay_rate', '0.2',
                    '--predictor-loss', 'l1', '--fixed_epoch_evaluation',
                    '--evaluation_split', 'test', '--ckpt', str(folder / 'last.pth'),
                    '--result_file', str(folder / 'test_metrics.json'),
                ]
                plan.append(dict(dataset=dataset, method=method, seed=seed,
                                 output_dir=str(folder), encoder=encoder, linear=linear))
    return plan

def summarize(records, output_dir):
    with (output_dir / 'runs.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['dataset', 'method', 'seed', 'rmse'])
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['dataset', 'method', 'runs', 'mean_rmse', 'std_rmse'])
        writer.writeheader()
        for dataset in CONFIGS:
            for method in METHODS:
                values = [r['rmse'] for r in records if r['dataset'] == dataset and r['method'] == method]
                if values:
                    writer.writerow(dict(dataset=dataset, method=method, runs=len(values),
                                         mean_rmse=statistics.mean(values),
                                         std_rmse=statistics.stdev(values) if len(values) > 1 else ''))

def main():
    args = parse_args()
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    for dataset in args.datasets:
        path = args.data_folder / FILES[dataset]
        if not path.is_file():
            raise FileNotFoundError(f'Extract the dataset archive first: missing {path}')

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'Use a new empty --output-dir: {args.output_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'commands.json').write_text(json.dumps(plan, indent=2))
    records = []
    for index, run in enumerate(plan, 1):
        folder = Path(run['output_dir'])
        folder.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(plan)}] {run['dataset']} {run['method']} seed={run['seed']}", flush=True)
        for stage in ('encoder', 'linear'):
            with (folder / f'{stage}_console.log').open('w') as log:
                try:
                    subprocess.run(run[stage], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                except subprocess.CalledProcessError:
                    print(f"Failed: see {folder / (stage + '_console.log')}", file=sys.stderr)
                    raise
        metrics = json.loads((folder / 'test_metrics.json').read_text())
        rmse = float(metrics['rmse'])
        if not math.isfinite(rmse):
            raise ValueError(f'Non-finite RMSE in {folder}')
        records.append(dict(dataset=run['dataset'], method=run['method'], seed=run['seed'], rmse=rmse))
        summarize(records, args.output_dir)
    print(f"Completed {len(records)} runs. Summary: {args.output_dir / 'summary.csv'}", flush=True)

if __name__ == '__main__':
    main()
