# modules/eda_signal.py

"""
Signal EDA functions for PTB-XL ECG dataset.

Functions:
    load_sample_per_class: Load one representative ECG per superclass.
    load_quality_samples: Load one ECG per quality flag category.
    load_all_signals: Load all ECG records across the full cohort.
    load_split_signals: Load all ECG records for a given split.
    sample_signals: Load a random sample of ECG records.
    compute_lead_stats: Compute per-lead amplitude statistics from array.
    compute_lead_stats: Compute per-lead statistics without loading all signals.
    compute_mean_psd: Compute mean PSD per lead without loading all signals.
    check_estimate_stability: Check whether amplitude estimates stabilize across sample sizes.
    plot_ecg: Plot a standard 12-lead ECG.
    plot_quality_comparison: Plot Lead II clean vs quality-flagged records.
    plot_lead_stats: Visualize per-lead amplitude statistics across splits as heatmaps.
    plot_psd: Plot mean PSD per lead in a 3x4 grid.
    plot_preprocessing_comparison: Plot Lead II at raw, filtered, and normalized stages.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wfdb
from pathlib import Path
from scipy.signal import welch
from tqdm import tqdm

# Module-level constants
CLASS_ORDER  = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
COLORS       = ['#378ADD', '#1D9E75', '#EF9F27', '#D4537E', '#7F77DD']
LEADS        = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
QUALITY_COLS = ['static_noise', 'burst_noise', 'baseline_drift',
                'electrodes_problems', 'extra_beats', 'pacemaker']
FLAG_COLORS  = {
    'clean':               '#1D9E75',
    'static_noise':        '#D4537E',
    'burst_noise':         '#EF9F27',
    'baseline_drift':      '#378ADD',
    'electrodes_problems': '#7F77DD',
    'extra_beats':         '#D85A30',
    'pacemaker':           '#888780',
}


# ── Loading functions ─────────────────────────────────────────────────────────

def load_sample_per_class(df: pd.DataFrame, path: Path,
                           fs: int = 100) -> dict:
    """
    Load one representative ECG record per diagnostic superclass.

    Prefers single-label records for clean class examples. Falls back to
    any record containing the class if no single-label records exist.

    Args:
        df (pd.DataFrame): Dataframe with 'superclass', 'split', 'filename_lr',
                           and 'filename_hr' columns.
        path (Path): Path to PTB-XL root directory.
        fs (int): Sampling frequency — 100 or 500 Hz. Default 100.

    Returns:
        dict: Keys are superclass names, values are (timesteps, 12) numpy arrays.

    Example:
        >>> samples = load_sample_per_class(df, PTBXL_ROOT, fs=100)
        >>> samples['NORM'].shape
        (1000, 12)
    """
    # Default to 100 Hz file and if not use 500 Hz file
    key      = 'filename_lr' if fs == 100 else 'filename_hr'
    # USe the train dataframe to extract samples
    train_df = df[df['split'] == 'train'].copy()
    samples  = {}

    for cls in CLASS_ORDER:
        # Prefer single-label records for a clean example
        mask   = train_df['superclass'].apply(lambda x: cls in x and len(x) == 1)
        subset = train_df[mask]

        if len(subset) == 0:
            mask   = train_df['superclass'].apply(lambda x: cls in x)
            subset = train_df[mask]
        # Load sample throgh wfdb
        row          = subset.iloc[0]
        signal, _    = wfdb.rdsamp(str(path / row[key]))
        samples[cls] = np.array(signal)

    return samples


def load_quality_samples(df: pd.DataFrame, path: Path,
                          fs: int = 100) -> dict:
    """
    Load one representative record per quality flag category plus one clean record.

    Args:
        df (pd.DataFrame): Dataframe with quality flag and filename columns.
        path (Path): Path to PTB-XL root directory.
        fs (int): Sampling frequency. Default 100.

    Returns:
        dict: Keys are quality flag names plus 'clean', values are
              (timesteps, 12) numpy arrays.

    Example:
        >>> quality_samples = load_quality_samples(df, DATA_PATH)
        >>> quality_samples['baseline_drift'].shape
        (1000, 12)
    """
    key      = 'filename_lr' if fs == 100 else 'filename_hr'
    train_df = df[df['split'] == 'train'].copy()
    samples  = {}

    # Clean record — no flags set on any quality column
    clean_mask       = train_df[QUALITY_COLS].isna().all(axis=1)
    clean_row        = train_df[clean_mask].iloc[0]
    signal, _        = wfdb.rdsamp(str(path / clean_row[key]))
    samples['clean'] = np.array(signal)

    # One flagged record per quality column
    for col in QUALITY_COLS:
        flagged = train_df[train_df[col].notna()]
        if len(flagged) == 0:
            continue
        row           = flagged.iloc[0]
        signal, _     = wfdb.rdsamp(str(path / row[key]))
        samples[col]  = np.array(signal)

    return samples


def load_all_signals(df: pd.DataFrame, path: Path,
                     fs: int = 100) -> np.ndarray:
    """
    Load all ECG records across the full cohort.

    Warning: requires ~2.09 GB RAM at 100 Hz. Ensure sufficient memory
    before calling. For memory-efficient statistics use
    compute_lead_stats_streaming() instead.

    Args:
        df (pd.DataFrame): Full dataframe with filename columns.
        path (Path): Path to PTB-XL root directory.
        fs (int): Sampling frequency. Default 100.

    Returns:
        np.ndarray: Array of shape (N, 1000, 12).

    Example:
        >>> X_all = load_all_signals(df, DATA_PATH)
        >>> X_all.shape
        (21799, 1000, 12)
    """
    key     = 'filename_lr' if fs == 100 else 'filename_hr'
    signals = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Loading all signals'):
        signal, _ = wfdb.rdsamp(str(path / row[key]))
        signals.append(signal)

    return np.stack(signals)


def load_split_signals(df: pd.DataFrame, path: Path,
                       split: str = 'train', fs: int = 100) -> np.ndarray:
    """
    Load all ECG records for a given split.

    Args:
        df (pd.DataFrame): Dataframe with 'split' and filename columns.
        path (Path): Path to PTB-XL root directory.
        split (str): One of 'train', 'val', 'test'. Default 'train'.
        fs (int): Sampling frequency. Default 100.

    Returns:
        np.ndarray: Array of shape (N, 1000, 12).

    Example:
        >>> X_train = load_split_signals(df, DATA_PATH, split='train')
        >>> X_train.shape
        (17440, 1000, 12)
    """
    key      = 'filename_lr' if fs == 100 else 'filename_hr'
    split_df = df[df['split'] == split]
    signals  = []

    for _, row in tqdm(split_df.iterrows(), total=len(split_df),
                       desc=f'Loading {split}'):
        signal, _ = wfdb.rdsamp(str(path / row[key]))
        signals.append(signal)

    return np.stack(signals)


def sample_signals(df: pd.DataFrame, path: Path,
                   n: int = 500, fs: int = 100,
                   seed: int = 42) -> np.ndarray:
    """
    Load a random sample of ECG records from the training set.

    Args:
        df (pd.DataFrame): Dataframe with 'split' and filename columns.
        path (Path): Path to PTB-XL root directory.
        n (int): Number of records to sample. Default 500.
        fs (int): Sampling frequency. Default 100.
        seed (int): Random seed for reproducibility. Default 42.

    Returns:
        np.ndarray: Array of shape (n, 1000, 12).

    Example:
        >>> X_sample = sample_signals(df, DATA_PATH, n=5000, seed=42)
        >>> X_sample.shape
        (5000, 1000, 12)
    """
    key      = 'filename_lr' if fs == 100 else 'filename_hr'
    train_df = df[df['split'] == 'train'].sample(n=n, random_state=seed)
    signals  = []

    for _, row in tqdm(train_df.iterrows(), total=n, desc=f'Loading {n} samples'):
        signal, _ = wfdb.rdsamp(str(path / row[key]))
        signals.append(signal)

    return np.stack(signals)


# ── Statistics functions ──────────────────────────────────────────────────────

# def compute_lead_stats(X: np.ndarray, leads: list = LEADS) -> pd.DataFrame:
#     """
#     Compute amplitude statistics per lead from an in-memory signal array.

#     Args:
#         X (np.ndarray): ECG array of shape (N, timesteps, 12).
#         leads (list): Lead names. Default LEADS.

#     Returns:
#         pd.DataFrame: Per-lead statistics with columns
#                       ['mean', 'std', 'min', 'max', 'p01', 'p99'].

#     Example:
#         >>> X_sample = sample_signals(df, DATA_PATH, n=5000)
#         >>> lead_stats = compute_lead_stats(X_sample)
#         >>> display(lead_stats)
#     """
#     stats = []
#     for i, lead in enumerate(leads):
#         lead_data = X[:, :, i].flatten()
#         stats.append({
#             'lead': lead,
#             'mean': lead_data.mean(), # Mean
#             'std':  lead_data.std(), # Standard Deviation
#             'min':  lead_data.min(), # Min Val
#             'max':  lead_data.max(), # Max Val
#             'p01':  np.percentile(lead_data, 1), # 1st percentile
#             'p99':  np.percentile(lead_data, 99), # 99th percentile
#         })

#     return pd.DataFrame(stats).set_index('lead').round(4)


def compute_lead_stats(df: pd.DataFrame, path: Path,
                                  fs: int = 100) -> pd.DataFrame:
    """
    Compute per-lead amplitude statistics without loading all signals into memory.

    Uses a single-pass running accumulator for mean and std. Drops p01/p99
    as exact percentiles require storing all values. Use compute_lead_stats()
    on a sample if percentiles are needed.

    Args:
        df (pd.DataFrame): Dataframe with filename columns.
        path (Path): Path to PTB-XL root directory.
        fs (int): Sampling frequency. Default 100.

    Returns:
        pd.DataFrame: Per-lead statistics with columns ['mean', 'std', 'min', 'max'].

    Example:
        >>> lead_stats = compute_lead_stats(df[df['split'] == 'train'], DATA_PATH)
        >>> display(lead_stats)
    """
    key      = 'filename_lr' if fs == 100 else 'filename_hr'
    n_leads  = 12
    sums     = np.zeros(n_leads)
    sums_sq  = np.zeros(n_leads)
    mins     = np.full(n_leads,  np.inf)
    maxs     = np.full(n_leads, -np.inf)
    n_total  = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Computing lead stats'):
        signal, _ = wfdb.rdsamp(str(path / row[key]))
        signal    = np.array(signal)

        sums    += signal.sum(axis=0)
        sums_sq += (signal ** 2).sum(axis=0)
        mins     = np.minimum(mins, signal.min(axis=0))
        maxs     = np.maximum(maxs, signal.max(axis=0))
        n_total += signal.shape[0]

    means = sums / n_total
    stds  = np.sqrt(sums_sq / n_total - means ** 2)

    stats = []
    for i, lead in enumerate(LEADS):
        stats.append({
            'lead': lead,
            'mean': round(float(means[i]), 4),
            'std':  round(float(stds[i]),  4),
            'min':  round(float(mins[i]),  4),
            'max':  round(float(maxs[i]),  4),
        })

    return pd.DataFrame(stats).set_index('lead')


def compute_mean_psd(df: pd.DataFrame, path: Path,
                                fs: int = 100) -> tuple:
    """
    Compute mean power spectral density per lead without loading all signals.

    Args:
        df (pd.DataFrame): Dataframe with filename columns.
        path (Path): Path to PTB-XL root directory.
        fs (int): Sampling frequency in Hz. Default 100.

    Returns:
        tuple:
            freqs (np.ndarray): Frequency bins in Hz.
            mean_psd (np.ndarray): Mean PSD of shape (n_freqs, 12).

    Example:
        >>> freqs, mean_psd = compute_mean_psd(df, DATA_PATH)
        >>> freqs.shape, mean_psd.shape
        ((129,), (129, 12))
    """
    key       = 'filename_lr' if fs == 100 else 'filename_hr'
    psd_accum = None
    n_records = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Computing PSD'):
        signal, _ = wfdb.rdsamp(str(path / row[key]))
        signal    = np.array(signal)

        psds = []
        for i in range(12):
            freqs, psd = welch(signal[:, i], fs=fs, nperseg=256)
            psds.append(psd)
        psds = np.stack(psds, axis=1)  # shape: (n_freqs, 12)

        psd_accum  = psds if psd_accum is None else psd_accum + psds
        n_records += 1

    return freqs, psd_accum / n_records

# ── Plotting functions ────────────────────────────────────────────────────────

def plot_ecg(signal: np.ndarray, title: str, fs: int = 100) -> plt.Figure:
    """
    Plot a standard 12-lead ECG from a (timesteps, 12) numpy array.

    Args:
        signal (np.ndarray): ECG signal of shape (timesteps, 12).
        title (str): Plot title.
        fs (int): Sampling frequency in Hz. Default 100.

    Returns:
        plt.Figure

    Example:
        >>> signal = samples['NORM']
        >>> fig = plot_ecg(signal, title='Example ECG — NORM')
        >>> plt.show()
    """
    t   = np.arange(signal.shape[0]) / fs
    fig, axes = plt.subplots(12, 1, figsize=(14, 16), sharex=True)

    for i, (ax, lead) in enumerate(zip(axes, LEADS)):
        ax.plot(t, signal[:, i], linewidth=0.7, color='#1a1a1a')
        ax.set_ylabel(lead, fontsize=9, rotation=0, labelpad=28, va='center')
        mean = signal[:, i].mean()
        ax.set_ylim(mean - 1.5, mean + 1.5)
        sns.despine(ax=ax, bottom=(i < 11))

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.001)
    fig.tight_layout()
    return fig


def plot_quality_comparison(quality_samples: dict, fs: int = 100) -> plt.Figure:
    """
    Plot Lead II for a clean record vs each quality-flagged record.

    Args:
        quality_samples (dict): Output from load_quality_samples().
        fs (int): Sampling frequency in Hz. Default 100.

    Returns:
        plt.Figure

    Example:
        >>> quality_samples = load_quality_samples(df, PTBXL_ROOT)
        >>> fig = plot_quality_comparison(quality_samples)
        >>> plt.show()
    """
    LEAD_IDX = 1
    t        = np.arange(1000) / fs
    n_plots  = len(quality_samples)

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, n_plots * 2.5), sharex=True)

    for ax, (flag, signal) in zip(axes, quality_samples.items()):
        color = FLAG_COLORS.get(flag, '#1a1a1a')
        ax.plot(t, signal[:, LEAD_IDX], linewidth=0.8, color=color)
        label = 'Clean (reference)' if flag == 'clean' else flag.replace('_', ' ').title()
        ax.set_ylabel(label, fontsize=9, rotation=0, labelpad=120, va='center')
        mean = signal[:, LEAD_IDX].mean()
        ax.set_ylim(mean - 2.5, mean + 2.5)
        sns.despine(ax=ax)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Lead II — clean vs quality-flagged records',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_lead_stats(stats_dict: dict) -> plt.Figure:
    """
    Visualize per-lead amplitude statistics across splits as heatmaps.

    Args:
        stats_dict (dict): Keys are split names, values are DataFrames
                           from compute_lead_stats_streaming().

    Returns:
        plt.Figure

    Example:
        >>> stats_dict = {key: compute_lead_stats_streaming(df_split, PTBXL_ROOT)
        ...               for key, df_split in splits.items()}
        >>> fig = plot_lead_stats(stats_dict)
        >>> plt.show()
    """
    metrics = ['mean', 'std', 'min', 'max']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 14))

    for ax, metric in zip(axes, metrics):
        matrix = pd.DataFrame(
            {key: stats[metric] for key, stats in stats_dict.items()}
        ).T

        sns.heatmap(
            matrix,
            annot=True, fmt='.3f',
            cmap='RdBu_r' if metric == 'mean' else 'Blues',
            ax=ax,
            linewidths=0.5,
            linecolor='white',
            cbar_kws={'shrink': 0.8}
        )
        ax.set_title(f'Per-lead {metric} across splits',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Lead')
        ax.set_ylabel('Split')

    fig.suptitle('Lead amplitude statistics — cohort vs splits',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_psd(freqs: np.ndarray, mean_psd: np.ndarray) -> plt.Figure:
    """
    Plot mean power spectral density per lead in a 3x4 grid.

    Args:
        freqs (np.ndarray): Frequency bins in Hz from compute_mean_psd_streaming().
        mean_psd (np.ndarray): Mean PSD of shape (n_freqs, 12).

    Returns:
        plt.Figure

    Example:
        >>> freqs, mean_psd = compute_mean_psd_streaming(df, PTBXL_ROOT)
        >>> fig = plot_psd(freqs, mean_psd)
        >>> plt.show()
    """
    psd_colors = [plt.cm.tab20(i) for i in range(12)]
    fig, axes  = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=False)
    axes       = axes.flatten()

    for i, (ax, lead) in enumerate(zip(axes, LEADS)):
        ax.semilogy(freqs, mean_psd[:, i],
                    color=psd_colors[i], linewidth=1.2)
        ax.axvline(0.5, color='gray', linestyle='--',
                   linewidth=0.8, label='0.5 Hz (high-pass)')
        ax.axvline(40,  color='red',  linestyle='--',
                   linewidth=0.8, label='40 Hz (low-pass)')
        ax.set_title(lead, fontsize=10, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=8)
        ax.set_ylabel('PSD (mV²/Hz)',   fontsize=8)
        sns.despine(ax=ax)

    axes[0].legend(fontsize=7)
    fig.suptitle('Mean power spectral density per lead — full cohort',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_preprocessing_comparison(signal_raw: np.ndarray,
                                   signal_filtered: np.ndarray,
                                   signal_preprocessed: np.ndarray,
                                   fs: int = 100,
                                   lead_idx: int = 1) -> plt.Figure:
    """
    Plot Lead II at three preprocessing stages: raw, filtered, normalized.

    Args:
        signal_raw (np.ndarray): Raw signal of shape (timesteps, 12).
        signal_filtered (np.ndarray): After bandpass filter (timesteps, 12).
        signal_preprocessed (np.ndarray): After filter + normalization (timesteps, 12).
        fs (int): Sampling frequency. Default 100.
        lead_idx (int): Lead index to plot. Default 1 (Lead II).

    Returns:
        plt.Figure

    Example:
        >>> raw        = quality_samples['baseline_drift']
        >>> filtered   = bandpass_filter(raw)
        >>> processed  = preprocess(raw)
        >>> fig = plot_preprocessing_comparison(raw, filtered, processed)
        >>> plt.show()
    """
    t      = np.arange(signal_raw.shape[0]) / fs
    stages = {
        'Raw signal':            (signal_raw,          '#888780'),
        'After bandpass filter': (signal_filtered,     '#378ADD'),
        'After normalization':   (signal_preprocessed, '#1D9E75'),
    }

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    for ax, (label, (signal, color)) in zip(axes, stages.items()):
        ax.plot(t, signal[:, lead_idx], linewidth=0.8, color=color)
        ax.set_ylabel('Amplitude', fontsize=9)
        mean = signal[:, lead_idx].mean()
        std  = signal[:, lead_idx].std()
        ax.set_title(f'{label}   |   mean={mean:.4f}   std={std:.4f}',
                     fontsize=10, fontweight='bold')
        sns.despine(ax=ax)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('Preprocessing pipeline validation — Lead II',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig