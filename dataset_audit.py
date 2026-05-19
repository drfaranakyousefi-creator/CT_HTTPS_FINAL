import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats

# import your existing extraction pipeline
from new_dataset import extract_data


def detect_outliers_iqr(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    mask = (x < low) | (x > high)
    return mask, low, high


def detect_outliers_zscore(x, threshold=3):
    z = np.abs(stats.zscore(x, nan_policy='omit'))
    return z > threshold


def label_report(y):
    y = np.asarray(y)
    iqr_mask, low, high = detect_outliers_iqr(y)
    z_mask = detect_outliers_zscore(y)

    return {
        'count': len(y),
        'mean': float(np.mean(y)),
        'std': float(np.std(y)),
        'min': float(np.min(y)),
        'max': float(np.max(y)),
        'iqr_outliers_%': float(iqr_mask.mean()*100),
        'zscore_outliers_%': float(z_mask.mean()*100),
        'iqr_bounds': (float(low), float(high))
    }


def feature_report(x):
    # x shape: (samples, W, N)
    flat = x.reshape(-1, x.shape[-1])
    report = []
    for i in range(flat.shape[1]):
        col = flat[:, i]
        report.append({
            'feature': i,
            'mean': float(np.mean(col)),
            'std': float(np.std(col)),
            'nan_%': float(np.isnan(col).mean()*100),
            'zero_%': float((col == 0).mean()*100),
            'variance': float(np.var(col))
        })
    return report


def mask_report(mask):
    valid_ratio = mask.mean()
    return {
        'valid_timestep_%': float(valid_ratio*100),
        'padding_%': float((1-valid_ratio)*100)
    }


def save_plots(y, outdir):
    outdir.mkdir(exist_ok=True, parents=True)

    plt.figure(figsize=(8,4))
    plt.hist(y, bins=50)
    plt.title('Label Distribution')
    plt.savefig(outdir / 'label_hist.png')
    plt.close()

    plt.figure(figsize=(6,6))
    stats.probplot(y, dist='norm', plot=plt)
    plt.title('QQ Plot')
    plt.savefig(outdir / 'qq_plot.png')
    plt.close()


def write_report(path, label_stats, feat_stats, mask_stats):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('DATASET AUDIT REPORT\n')
        f.write('='*50 + '\n\n')

        f.write('LABEL REPORT\n')
        for k,v in label_stats.items():
            f.write(f'{k}: {v}\n')

        f.write('\nMASK REPORT\n')
        for k,v in mask_stats.items():
            f.write(f'{k}: {v}\n')

        f.write('\nFEATURE REPORT\n')
        for row in feat_stats:
            f.write(str(row) + '\n')

        f.write('\nRECOMMENDATION\n')
        if label_stats['iqr_outliers_%'] > 5:
            f.write('Consider robust loss (SmoothL1Loss) before deleting outliers.\n')
        else:
            f.write('No urgent outlier removal needed.\n')

        if mask_stats['padding_%'] > 70:
            f.write('WARNING: too much padding; check window size W.\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--dataset', choices=['metavision','carevue'], required=True)
    parser.add_argument('--target', choices=['spO2','BP','RR'], required=True)
    parser.add_argument('--window', type=int, default=40)
    parser.add_argument('--outdir', default='audit_output')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    x, y, mask = extract_data(
        dataset_name=args.dataset,
        df_chartevents=df,
        w=args.window,
        target=args.target,
        normalize=True
    )

    x = x.numpy()
    y = y.numpy()
    mask = mask.numpy()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    label_stats = label_report(y)
    feat_stats = feature_report(x)
    mask_stats = mask_report(mask)

    save_plots(y, outdir)
    write_report(outdir/'report.txt', label_stats, feat_stats, mask_stats)

    print(f'Report saved in: {outdir}')


if __name__ == '__main__':
    main()
