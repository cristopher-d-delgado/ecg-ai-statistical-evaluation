# modules/__init__.py

# Preprocessing
from .preprocessing import (
    bandpass_filter,
    normalize,
    preprocess,
)

# EDA — metadata
from .eda_metadata import (
    cohort_summary_stats,
    plot_cohort_summary_stats,
    demographics_by_superclass,
    plot_demographics_by_superclass,
    plot_cooccurrence,
)

# EDA — signal
from .eda_signal import (
    load_sample_per_class,
    load_quality_samples,
    load_all_signals,
    load_split_signals,
    sample_signals,
    compute_lead_stats,
    compute_lead_stats_streaming,
    compute_mean_psd_streaming,
    check_estimate_stability,
    plot_ecg,
    plot_quality_comparison,
    plot_lead_stats,
    plot_psd,
    plot_preprocessing_comparison,
)

__all__ = [
    # Preprocessing
    'bandpass_filter',
    'normalize',
    'preprocess',
    # Metadata EDA
    'cohort_summary_stats',
    'plot_cohort_summary_stats',
    'demographics_by_superclass',
    'plot_demographics_by_superclass',
    'plot_cooccurrence',
    # Signal EDA
    'load_sample_per_class',
    'load_quality_samples',
    'load_all_signals',
    'load_split_signals',
    'sample_signals',
    'compute_lead_stats',
    'compute_lead_stats_streaming',
    'compute_mean_psd_streaming',
    'check_estimate_stability',
    'plot_ecg',
    'plot_quality_comparison',
    'plot_lead_stats',
    'plot_psd',
    'plot_preprocessing_comparison',
]