# modules/preprocessing.py

"""
Preprocessing pipeline for PTB-XL ECG signals.

Functions:
    bandpass_filter: Zero-phase Butterworth bandpass filter.
    normalize: Per-record z-score normalization.
    preprocess: Full pipeline — filter then normalize.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal: np.ndarray, lowcut: float = 0.5, highcut: float = 40.0,
                    fs: int = 100, order: int = 4) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter to an ECG signal.

    Removes baseline wander (below 0.5 Hz) and high-frequency noise (above 40 Hz)
    while preserving all clinically meaningful ECG components (P wave, QRS complex,
    T wave). Zero-phase filtering via filtfilt ensures no phase distortion.

    Args:
        signal (np.ndarray): ECG signal of shape (timesteps, 12).
        lowcut (float): High-pass cutoff frequency in Hz. Default 0.5 Hz.
        highcut (float): Low-pass cutoff frequency in Hz. Default 40.0 Hz.
        fs (int): Sampling frequency in Hz. Default 100 Hz.
        order (int): Butterworth filter order. Default 4.

    Returns:
        np.ndarray: Filtered signal of shape (timesteps, 12), same dtype as input.

    Raises:
        ValueError: If lowcut >= highcut or if cutoffs exceed Nyquist frequency.

    Example:
        >>> signal = np.random.randn(1000, 12)
        >>> filtered = bandpass_filter(signal, fs=100)
        >>> filtered.shape
        (1000, 12)
    """
    # Define an Error guard where an invalid cutoff is present
    if lowcut >= highcut:
        raise ValueError(f"lowcut ({lowcut}) must be less than highcut ({highcut})")

    nyq = 0.5 * fs # Define Nyquist

    # Define an Error guard where the Highcut exceeds the Nyquist frequency
    if highcut >= nyq:
        raise ValueError(
            f"highcut ({highcut} Hz) must be less than Nyquist frequency ({nyq} Hz)"
        )

    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band') # Perform Signal Filtering 
    return filtfilt(b, a, signal, axis=0)


def normalize(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-record z-score normalization across all leads jointly.

    Normalizes using the global mean and standard deviation across all timesteps
    and leads in a single record.

    Args:
        signal (np.ndarray): ECG signal of shape (timesteps, 12).
        eps (float): Small constant added to std to prevent division by zero.
                     Default 1e-8.

    Returns:
        np.ndarray: Normalized signal of shape (timesteps, 12), zero mean and
                    unit variance across all leads jointly.

    Example:
        >>> signal = np.random.randn(1000, 12)
        >>> normed = normalize(signal)
        >>> abs(normed.mean()) < 1e-6
        True
    """
    mean = signal.mean()
    std  = signal.std() + eps
    return (signal - mean) / std


def preprocess(signal: np.ndarray, fs: int = 100,
               lowcut: float = 0.5, highcut: float = 40.0,
               order: int = 4) -> np.ndarray:
    """
    Full ECG preprocessing pipeline: bandpass filter then z-score normalize.

    Applies a zero-phase Butterworth bandpass filter to remove baseline wander
    and high-frequency noise, followed by per-record z-score normalization.
    Output is cast to float32 for compatibility with PyTorch.

    Confirmed on PTB-XL via PSD analysis: signal power is concentrated in
    0.5–40 Hz across all 12 leads. Per-record normalization chosen over
    per-lead to preserve inter-lead amplitude relationships.

    Args:
        signal (np.ndarray): Raw ECG signal of shape (timesteps, 12).
        fs (int): Sampling frequency in Hz. Default 100 Hz.
        lowcut (float): High-pass cutoff in Hz. Default 0.5 Hz.
        highcut (float): Low-pass cutoff in Hz. Default 40.0 Hz.
        order (int): Butterworth filter order. Default 4.

    Returns:
        np.ndarray: Preprocessed ECG signal of shape (timesteps, 12), dtype float32.

    Example:
        >>> import wfdb
        >>> signal, _ = wfdb.rdsamp('path/to/record')
        >>> signal = np.array(signal)          # (1000, 12)
        >>> processed = preprocess(signal)
        >>> processed.shape, processed.dtype
        ((1000, 12), dtype('float32'))

    References:
        Strodthoff et al. (2021). Deep learning for ECG analysis: benchmarks
        and insights from PTB-XL. IEEE Journal of Biomedical and Health
        Informatics, 25(5), 1519–1528. https://doi.org/10.1109/JBHI.2020.3022989
    """
    signal = bandpass_filter(signal, lowcut=lowcut, highcut=highcut,
                             fs=fs, order=order)
    signal = normalize(signal)
    return signal.astype(np.float32)