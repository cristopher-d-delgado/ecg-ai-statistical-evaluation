# modules/modeling.py

"""
Modeling functions and classes for ECG AI Statistical Validation.

Classes:
    ResBlock1D: Single 1D residual block.
    ResNet1D: 8-block 1D ResNet for multi-label ECG classification.
    EarlyStopping: Early stopping callback.
    TemperatureScaling: Global post-hoc temperature scaling.
    PerClassTemperatureScaling: Per-class post-hoc temperature scaling.

Functions:
    count_parameters: Count trainable model parameters.
    train_epoch: Run one training epoch.
    val_epoch: Run one validation epoch.
    find_optimal_threshold: Find F1-maximising threshold on validation set.
    compute_classification_metrics: Compute per-class classification metrics.
"""
# Libraries 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve, auc
from scipy.stats import skew, kurtosis as kurt
from tqdm import tqdm
import wfdb 
from preprocessing import preprocess
import matplotlib.pyplot as plt
import seaborn as sns

# Module level const
CLASS_ORDER = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

# -- Functions for Logistic and RF Models --------------------------------------
def extract_features(signal: np.ndarray) -> np.ndarray:
    """
    Extract per-lead statistical features from a preprocessed ECG signal.

    Computes 13 statistical features per lead across all 12 leads,
    producing a 156-dimensional feature vector per record.

    Features per lead:
        mean, std, min, max, range, absolute mean, RMS,
        skewness, kurtosis, p10, p25, p75, p90

    Args:
        signal (np.ndarray): Preprocessed ECG of shape (timesteps, 12).

    Returns:
        np.ndarray: Feature vector of shape (156,) float32.

    Example:
        >>> signal = preprocess(raw_signal)   # (1000, 12)
        >>> features = extract_features(signal)
        >>> features.shape
        (156,)
    """
    features = []
    for i in range(signal.shape[1]):
        lead = signal[:, i]
        features.extend([
            lead.mean(),
            lead.std(),
            lead.min(),
            lead.max(),
            lead.max() - lead.min(),      # range
            np.abs(lead).mean(),          # absolute mean
            np.sqrt(np.mean(lead ** 2)), # RMS
            skew(lead),                   # skewness
            kurt(lead),                   # kurtosis
            np.percentile(lead, 10),
            np.percentile(lead, 25),
            np.percentile(lead, 75),
            np.percentile(lead, 90),
        ])
    return np.array(features, dtype=np.float32)

def build_feature_matrix(df: pd.DataFrame, path: Path,
                          fs: int = 100) -> np.ndarray:
    """
    Build a feature matrix by extracting statistical features from
    all records in a dataframe split.

    Loads each record from disk, applies the full preprocessing pipeline,
    then extracts statistical features. Uses the same preprocessing as the
    deep learning model for fair comparison.

    Args:
        df (pd.DataFrame): Dataframe split with filename columns.
        path (Path): Path to PTB-XL root directory.
        fs (int): Sampling frequency. Default 100.

    Returns:
        np.ndarray: Feature matrix of shape (N, 156).

    Example:
        >>> X_train = build_feature_matrix(train_df, PTBXL_ROOT, fs=100)
        >>> X_train.shape
        (17440, 156)
    """
    key      = 'filename_lr' if fs == 100 else 'filename_hr'
    features = []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc='Extracting features'):
        signal, _ = wfdb.rdsamp(str(path / row[key]))
        signal    = np.array(signal, dtype=np.float32)
        signal    = preprocess(signal, fs=fs)
        features.append(extract_features(signal))

    return np.stack(features)  # shape: (N, 156)

# -- Dataset -------------------------------------------------------------------
# Create Pytorch dataset class
class PTBXLDataset(Dataset):
    """
    PyTorch Dataset for PTB-XL 12-lead ECG classification.

    Loads raw waveforms, applies the full preprocessing
    pipeline (bandpass filter + per-record z-score normalization),
    and returns tensors.

    Args:
        df (pd.DataFrame): Dataframe subset for this split containing
                           'filename_lr' or 'filename_hr' columns.
        labels (np.ndarray): Multi-label binary matrix of shape (N, 5).
        ptbxl_root (Path): Path to PTB-XL root directory.
        fs (int): Sampling frequency — 100 or 500 Hz. Default 100.
        augment (bool): Whether to apply data augmentation. Default False.

    Returns per item:
        signal (torch.Tensor): Shape (12, 1000) float32 — (leads, timesteps).
        label (torch.Tensor): Shape (5,) float32 — multi-label binary vector.
    """

    def __init__(self, df: pd.DataFrame, labels: np.ndarray, ptbxl_root: Path, fs: int = 100, augment: bool = False):
        """Initialize arguments; stores everything the dataset needs to do its job"""
        self.df = df.reset_index() # ensures integer positional indexing works correctly when __getitem__ uses .iloc[idx]
        self.labels = labels # Labels for the Data
        self.ptbxl_root = ptbxl_root # Path() to data
        self.key = 'filename_lr' if fs == 100 else 'filename_hr' # Default to 100 Hz data if not use 500 Hz data
        self.fs = fs # Sampling frequency
        self.augment = augment # Augment the data True or False
    
    def __len__(self) -> int:
        """Tells PyTorch how many records are in the dataset. The DataLoader uses this to know when one epoch is complete"""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Core method that lets PyTorch Train.
        For each idx 5 processes occur.
            1. Finds file path - looks up `filename_lr` in the dataframe.
            2. Loads the waveform - `wfdb.rdsamp` reads the `.dat` and `.hea` files and returns a numpy array.
            3. Pre-process - runs the bandpass filter then z-score normalization.
            4. Transposes - flips from (1000, 12) to (12, 1000) because PyTorch accepts layers in (channels, timesteps).
            5. Return Tensors - signal and label as `float32` tensors.
        """
        # Index the Signal row
        row = self.df.iloc[idx]

        # Load raw signal — shape: (1000, 12) when using 100 Hz
        signal, _ = wfdb.rdsamp(str(self.ptbxl_root / row[self.key]))
        signal    = np.array(signal, dtype=np.float32)

        # Preprocess: bandpass filter + per-record z-score
        signal = preprocess(signal, fs=self.fs)

        # Transpose to (12, 1000) for Conv1d — (leads, timesteps)
        signal = signal.T

        # Optional augmentation (training only)
        if self.augment:
            signal = self._augment(signal)

        label = self.labels[idx].astype(np.float32)

        return torch.tensor(signal), torch.tensor(label)
    
    def _augment(self, signal: np.ndarray) -> np.ndarray:
        """
        Helper function for light augmentation for training purposes. 
        Applied per-record at load time.
        """
        # Apply Gausian noise 
        if np.random.rand() < 0.5: 
            signal = signal + np.random.normal(0, 0.01, signal.shape).astype(np.float32)
        
        # Random Amplitude scaling 
        if np.random.rand() < 0.5: 
            scale = np.random.uniform(0.9, 1.1)
            signal = signal * scale
        
        return signal

# ── Architecture ──────────────────────────────────────────────────────────────
# Define the a Simple ResNet Block 
class ResBlock1D(nn.Module):
    """
    Single 1D block with two convolutional layers and a skip connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolutional kernel size. Default 7.
        stride (int): Stride for first convolution. Default 1.
        dropout (float): Dropout probability. Default 0.2.
    """
    def __init__ (self, in_channels: int, out_channels: int, kernel_size: int = 7, stride: int = 1, dropout: float = 0.2):
        super().__init__() # Adopt properties of nn.Module

        padding = kernel_size // 2 # Should be 1

        # Define the blocks
        self.main = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, 
                kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=out_channels, out_channels=out_channels, 
                kernel_size=kernel_size, stride=1, padding=padding
            ),
            nn.BatchNorm1d(out_channels),
        )
        # Skip path — Sequential if projection needed, Identity otherwise
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1,
                      stride=stride, bias=False),
            nn.BatchNorm1d(out_channels)
        ) if (in_channels != out_channels or stride != 1) else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This addition is why forward() is unavoidable in ResBlock
        return self.relu(self.main(x) + self.skip(x))

# Now we will define the entire ResNet Architecture 
class ResNet1D(nn.Module):
    def __init__(self, n_leads: int = 12, n_classes: int = 5,
                 base_filters: int = 64, dropout: float = 0.2):
        super().__init__()
        f = base_filters

        # Stem — fully Sequential
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, f, kernel_size=15,
                      stride=2, padding=7, bias=False),
            nn.BatchNorm1d(f),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # 8 blocks — Sequential of ResBlock1D objects; Use 4 for now since 8 might be overkill
        self.blocks = nn.Sequential(
            ResBlock1D(f, f, dropout=dropout),  # block 1
            #ResBlock1D(f, f, dropout=dropout),  # block 2
            ResBlock1D(f, f * 2, stride=2, dropout=dropout),  # block 3
            #ResBlock1D(f * 2, f * 2, dropout=dropout),  # block 4
            ResBlock1D(f * 2, f * 4, stride=2, dropout=dropout),  # block 5
            ResBlock1D(f * 4, f * 4, dropout=dropout),  # block 6
            #ResBlock1D(f * 4, f * 8, stride=2, dropout=dropout),  # block 7
            #ResBlock1D(f * 8, f * 8, dropout=dropout),  # block 8
        )

        # Head — fully Sequential
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            #nn.Linear(f * 8, n_classes)
            nn.Linear(f * 4, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a model.

    Args:
        model (nn.Module): PyTorch model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ── Training utilities ────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Args:
        patience (int): Epochs to wait before stopping. Default 10.
        delta (float): Minimum improvement to count. Default 1e-4.
        path (Path): Where to save best model checkpoint.
    """

    def __init__(self, patience: int = 10, delta: float = 1e-4,
                 path: Path = Path('artifacts/best_model.pt')):
        self.patience   = patience
        self.delta      = delta
        self.path       = path
        self.counter    = 0
        self.best_loss  = np.inf
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.path)
            print(f"  Checkpoint saved (val_loss={val_loss:.4f})")
        else:
            self.counter += 1
            print(f"  No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True

def train_epoch(model: nn.Module, loader, criterion,
                optimizer, device: torch.device) -> float:
    """
    Run one training epoch.

    Args:
        model (nn.Module): ResNet1D model.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device (torch.device): Compute device.

    Returns:
        float: Mean training loss.
    """
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc='Train', leave=False)
    for signals, labels in pbar:
        signals = signals.to(device)
        labels  = labels.to(device)

        optimizer.zero_grad()
        logits = model(signals)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * signals.size(0)
        pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})

    return total_loss / len(loader.dataset)


def val_epoch(model: nn.Module, loader, criterion,
              device: torch.device) -> tuple:
    """
    Run one validation epoch.

    Args:
        model (nn.Module): ResNet1D model.
        loader: Validation DataLoader.
        criterion: Loss function.
        device (torch.device): Compute device.

    Returns:
        tuple: (mean_loss, macro_auc, all_probs, all_labels)
    """
    model.eval()
    total_loss = 0.0
    all_probs  = []
    all_labels = []

    pbar = tqdm(loader, desc='Val', leave=False)
    with torch.no_grad():
        for signals, labels in pbar:
            signals = signals.to(device)
            labels  = labels.to(device)

            logits = model(signals)
            loss   = criterion(logits, labels)
            probs  = torch.sigmoid(logits)

            total_loss += loss.item() * signals.size(0)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})

    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    mean_loss  = total_loss / len(loader.dataset)
    macro_auc  = roc_auc_score(all_labels, all_probs, average='macro')

    return mean_loss, macro_auc, all_probs, all_labels

# ── Calibration ───────────────────────────────────────────────────────────────
class TemperatureScaling(nn.Module):
    """
    Global post-hoc temperature scaling.

    Args:
        init_temp (float): Initial temperature. Default 1.0.

    Reference:
        Guo et al. (2017). On calibration of modern neural networks.
        ICML 2017. https://arxiv.org/abs/1706.04599
    """

    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def calibrate(self, val_loader, model, criterion, device,
                  lr: float = 0.01, max_iter: int = 50) -> float:
        self.to(device)
        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.to(device)
                logits  = model(signals)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits).to(device)
        all_labels = torch.cat(all_labels).to(device)

        optimizer = optim.LBFGS([self.temperature], lr=lr,
                                  max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self(all_logits), all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        self.temperature.data = self.temperature.data.clamp(0.1, 10.0)
        return round(float(self.temperature.item()), 4)

class PerClassTemperatureScaling(nn.Module):
    """
    Per-class temperature scaling.
    Learns one scalar T per class independently.

    Args:
        n_classes (int): Number of output classes. Default 5.
        init_temp (float): Initial temperature. Default 1.0.
    """

    def __init__(self, n_classes: int = 5, init_temp: float = 1.0):
        super().__init__()
        self.temps = nn.ParameterList([
            nn.Parameter(torch.tensor([init_temp]))
            for _ in range(n_classes)
        ])

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            logits[:, i] / self.temps[i]
            for i in range(len(self.temps))
        ], dim=1)

    def calibrate(self, val_loader, model, criterion, device,
                  lr: float = 0.01, max_iter: int = 100) -> np.ndarray:
        self.to(device)
        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.to(device)
                logits  = model(signals)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits    = torch.cat(all_logits).to(device)
        all_labels    = torch.cat(all_labels).to(device)
        optimal_temps = []

        for i in range(len(self.temps)):
            self.temps[i].data.fill_(1.0)
            optimizer = optim.LBFGS([self.temps[i]], lr=lr,
                                     max_iter=max_iter)

            def eval_loss(idx=i):
                optimizer.zero_grad()
                scaled = all_logits[:, idx] / self.temps[idx]
                loss   = nn.BCEWithLogitsLoss()(scaled, all_labels[:, idx])
                loss.backward()
                return loss

            optimizer.step(eval_loss)
            self.temps[i].data.clamp_(0.1, 10.0)
            optimal_temps.append(round(float(self.temps[i].item()), 4))

        return np.array(optimal_temps)

# ── Evaluation ────────────────────────────────────────────────────────────────
def find_optimal_threshold(y_true: np.ndarray,
                            y_prob: np.ndarray) -> float:
    """
    Find threshold maximising F1 score.

    Args:
        y_true (np.ndarray): Binary labels of shape (N,).
        y_prob (np.ndarray): Predicted probabilities of shape (N,).

    Returns:
        float: Optimal threshold.
    """
    from sklearn.metrics import roc_curve
    _, _, thresholds = roc_curve(y_true, y_prob)
    f1_scores        = []

    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)
        f1_scores.append(f1_score(y_true, preds, zero_division=0))

    return float(thresholds[np.argmax(f1_scores)])

def compute_classification_metrics(y_true: np.ndarray,
                                    y_prob: np.ndarray,
                                    thresholds: np.ndarray,
                                    class_order: list) -> pd.DataFrame:
    """
    Compute per-class classification metrics with macro average row.

    Args:
        y_true (np.ndarray): Binary label matrix (N, n_classes).
        y_prob (np.ndarray): Probability matrix (N, n_classes).
        thresholds (np.ndarray): Per-class thresholds (n_classes,).
        class_order (list): Class names.

    Returns:
        pd.DataFrame: Per-class metrics with macro row.
    """
    results = []

    for i, cls in enumerate(class_order):
        thresh  = thresholds[i]
        preds   = (y_prob[:, i] >= thresh).astype(int)
        labels  = y_true[:, i]

        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy    = (tp + tn) / (tp + tn + fp + fn)
        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1          = f1_score(labels, preds, zero_division=0)
        npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        results.append({
            'Class':       cls,
            'Threshold':   round(thresh,      4),
            'Sensitivity': round(sensitivity, 4),
            'Specificity': round(specificity, 4),
            'Accuracy':    round(accuracy,    4),
            'Precision':   round(precision,   4),
            'F1':          round(f1,          4),
            'PPV':         round(precision,   4),
            'NPV':         round(npv,         4),
            'TP': int(tp), 'TN': int(tn),
            'FP': int(fp), 'FN': int(fn),
        })

    df = pd.DataFrame(results).set_index('Class')

    macro = {
        'Threshold':   float('nan'),
        'Sensitivity': round(df['Sensitivity'].mean(), 4),
        'Specificity': round(df['Specificity'].mean(), 4),
        'Accuracy':    round(df['Accuracy'].mean(),    4),
        'Precision':   round(df['Precision'].mean(),   4),
        'F1':          round(df['F1'].mean(),          4),
        'PPV':         round(df['PPV'].mean(),         4),
        'NPV':         round(df['NPV'].mean(),         4),
        'TP': int(df['TP'].sum()),
        'TN': int(df['TN'].sum()),
        'FP': int(df['FP'].sum()),
        'FN': int(df['FN'].sum()),
    }

    return pd.concat([df, pd.DataFrame(macro, index=['Macro (mean)'])])

def bootstrap_macro_auc(y_true: np.ndarray, y_prob: np.ndarray,
                         n_boot: int = 1000, seed: int = 42) -> tuple:
    """
    Compute 95% bootstrap confidence interval for macro-AUC.

    Resamples records with replacement n_boot times and computes
    macro-AUC on each resample. Returns the 2.5th and 97.5th
    percentiles as the 95% CI.

    Args:
        y_true (np.ndarray): Binary label matrix of shape (N, n_classes).
        y_prob (np.ndarray): Probability matrix of shape (N, n_classes).
        n_boot (int): Number of bootstrap resamples. Default 1000.
        seed (int): Random seed for reproducibility. Default 42.

    Returns:
        tuple: (point_estimate, ci_lower, ci_upper)
    """
    rng          = np.random.default_rng(seed)
    n            = len(y_true)
    point_est    = roc_auc_score(y_true, y_prob, average='macro')
    boot_aucs    = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)

        # Skip if any class has no positive samples in resample
        if y_true[idx].sum(axis=0).min() == 0:
            continue

        auc = roc_auc_score(y_true[idx], y_prob[idx], average='macro')
        boot_aucs.append(auc)

    boot_aucs = np.array(boot_aucs)
    ci_lower  = np.percentile(boot_aucs, 2.5)
    ci_upper  = np.percentile(boot_aucs, 97.5)

    return point_est, ci_lower, ci_upper

def bootstrap_per_class_auc(y_true: np.ndarray, y_prob: np.ndarray,
                              class_order: list, n_boot: int = 1000,
                              seed: int = 42) -> pd.DataFrame:
    """
    Compute 95% bootstrap confidence intervals for per-class AUC.

    Args:
        y_true (np.ndarray): Binary label matrix of shape (N, n_classes).
        y_prob (np.ndarray): Probability matrix of shape (N, n_classes).
        class_order (list): Class names.
        n_boot (int): Number of bootstrap resamples. Default 1000.
        seed (int): Random seed for reproducibility. Default 42.

    Returns:
        pd.DataFrame: Per-class AUC with 95% CI columns.
    """
    rng     = np.random.default_rng(seed)
    n       = len(y_true)
    results = []

    for i, cls in enumerate(class_order):
        point_est  = roc_auc_score(y_true[:, i], y_prob[:, i])
        boot_aucs  = []

        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            if y_true[idx, i].sum() == 0:
                continue
            auc = roc_auc_score(y_true[idx, i], y_prob[idx, i])
            boot_aucs.append(auc)

        boot_aucs = np.array(boot_aucs)
        results.append({
            'Class':    cls,
            'AUC':      round(point_est, 4),
            'CI Lower': round(np.percentile(boot_aucs, 2.5),  4),
            'CI Upper': round(np.percentile(boot_aucs, 97.5), 4),
            'CI 95%':   f"{point_est:.4f} ({np.percentile(boot_aucs, 2.5):.4f}–{np.percentile(boot_aucs, 97.5):.4f})"
        })

    return pd.DataFrame(results).set_index('Class')

# -- Plotting Functions --------------------------------------------------------
# ── Plot training curves ──────────────────────────────────────────────────────
def plot_training_curves(history: dict) -> plt.Figure:
    """
    Plot training and validation loss curves plus validation AUC.

    Args:
        history (dict): Keys 'train_loss', 'val_loss', 'val_auc'.

    Returns:
        plt.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs          = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], color='#378ADD',
             linewidth=1.5, label='Train loss')
    ax1.plot(epochs, history['val_loss'],   color='#D4537E',
             linewidth=1.5, label='Val loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and validation loss',
                  fontsize=12, fontweight='bold')
    ax1.legend()
    sns.despine(ax=ax1)

    ax2.plot(epochs, history['val_auc'], color='#1D9E75',
             linewidth=1.5, label='Val macro-AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro-AUC')
    ax2.set_title('Validation macro-AUC',
                  fontsize=12, fontweight='bold')
    ax2.set_ylim(0.5, 1.0)
    ax2.legend()
    sns.despine(ax=ax2)

    fig.tight_layout()
    return fig

def plot_per_class_auc(class_auc_table: pd.DataFrame) -> plt.Figure:
    """
    Plot per-class AUC comparison across models.

    Args:
        class_auc_table (pd.DataFrame): Output from per_class_auc comparison.

    Returns:
        plt.Figure
    """
    # Drop macro row for plotting
    plot_df = class_auc_table.drop('Macro (mean)')

    classes = plot_df.index.tolist()
    x       = np.arange(len(classes))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - width, plot_df['Logistic Regression'], width,
           label='Logistic Regression', color='#888780', edgecolor='none')
    ax.bar(x,         plot_df['Random Forest'],       width,
           label='Random Forest',       color='#EF9F27', edgecolor='none')
    ax.bar(x + width, plot_df['ResNet1D'],            width,
           label='ResNet1D',            color='#378ADD', edgecolor='none')

    ax.set_xlabel('Diagnostic superclass')
    ax.set_ylabel('AUC')
    ax.set_title('Per-class AUC — model comparison',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0.7, 1.0)
    ax.axhline(0.9, color='gray', linewidth=0.8,
               linestyle='--', label='AUC = 0.90')
    ax.legend()
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig

def plot_auc_with_ci(ci_data: dict) -> plt.Figure:
    """
    Plot AUC estimates with 95% bootstrap confidence intervals.
    Forest plot style — one row per model or class.

    Args:
        ci_data (dict): Keys are labels, values are (estimate, lower, upper) tuples.

    Returns:
        plt.Figure
    """
    labels    = list(ci_data.keys())
    estimates = [v[0] for v in ci_data.values()]
    lowers    = [v[1] for v in ci_data.values()]
    uppers    = [v[2] for v in ci_data.values()]
    errors    = [[e - l for e, l in zip(estimates, lowers)],
                 [u - e for e, u in zip(estimates, uppers)]]

    y_pos  = np.arange(len(labels))
    colors = ['#888780', '#EF9F27', '#378ADD']

    fig, ax = plt.subplots(figsize=(9, 4))

    for i, (est, lo, hi, label) in enumerate(zip(estimates, lowers, uppers, labels)):
        color = colors[i % len(colors)]
        ax.errorbar(
            x          = est,
            y          = i,
            xerr       = [[est - lo], [hi - est]],
            fmt        = 'o',
            color      = color,
            ecolor     = color,
            elinewidth = 1.5,
            capsize    = 5,
            capthick   = 1.5,
            markersize = 8,
            label      = label
        )
        ax.text(hi + 0.002, i,
                f"{est:.2f} ({lo:.2f}–{hi:.2f})",
                va='center', fontsize=9,
                color=color)

    ax.axvline(0.9, color='gray', linewidth=0.8,
               linestyle='--', label='AUC = 0.90')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Macro-AUC')
    ax.set_title('Macro-AUC with 95% bootstrap CI',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0.82, 0.96)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig

def plot_roc_curves(y_true: np.ndarray,
                    model_probs: dict,
                    class_order: list) -> plt.Figure:
    """
    Plot ROC curves for all models on the same axes,
    one subplot per diagnostic superclass plus a macro subplot.

    Args:
        y_true (np.ndarray): Binary label matrix of shape (N, n_classes).
        model_probs (dict): Keys are model names, values are prob matrices.
        class_order (list): Class names.

    Returns:
        plt.Figure
    """
    MODEL_COLORS = {
        'Logistic Regression': '#888780',
        'Random Forest':       '#EF9F27',
        'ResNet1D':            '#378ADD',
    }

    n_classes = len(class_order)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes      = axes.flatten()

    # ── Per-class ROC curves ──────────────────────────────────────────────────
    for i, cls in enumerate(class_order):
        ax = axes[i]

        for model_name, probs in model_probs.items():
            fpr, tpr, _ = roc_curve(y_true[:, i], probs[:, i])
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr,
                    color     = MODEL_COLORS[model_name],
                    linewidth = 1.5,
                    label     = f"{model_name} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], color='gray', linewidth=0.8,
                linestyle='--', label='Random classifier')
        ax.set_title(cls, fontsize=11, fontweight='bold')
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc='lower right')
        sns.despine(ax=ax)

    # ── Macro ROC — average across classes ───────────────────────────────────
    ax = axes[5]

    for model_name, probs in model_probs.items():
        # Interpolate all ROC curves to common FPR grid then average
        mean_fpr = np.linspace(0, 1, 200)
        tprs     = []

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], probs[:, i])
            tprs.append(np.interp(mean_fpr, fpr, tpr))

        mean_tpr     = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        macro_auc    = auc(mean_fpr, mean_tpr)

        ax.plot(mean_fpr, mean_tpr,
                color     = MODEL_COLORS[model_name],
                linewidth = 1.5,
                label     = f"{model_name} (AUC={macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], color='gray', linewidth=0.8,
            linestyle='--', label='Random classifier')
    ax.set_title('Macro (mean)', fontsize=11, fontweight='bold')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc='lower right')
    sns.despine(ax=ax)

    fig.suptitle('ROC curves — model comparison',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig

# ── Plot ResNet1D per-class CI ─────────────────────────────────────────────────
def plot_per_class_ci(class_ci_df: pd.DataFrame) -> plt.Figure:
    """
    Plot per-class AUC with 95% CI for a single model.

    Args:
        class_ci_df (pd.DataFrame): Output from bootstrap_per_class_auc()
                                    with 'AUC', 'CI Lower', 'CI Upper' columns.

    Returns:
        plt.Figure
    """
    PALETTE = dict(zip(CLASS_ORDER, ['#378ADD','#1D9E75','#EF9F27','#D4537E','#7F77DD']))
    classes  = class_ci_df.index.tolist()
    y_pos    = np.arange(len(classes))

    fig, ax = plt.subplots(figsize=(9, 4))

    for i, cls in enumerate(classes):
        est   = class_ci_df.loc[cls, 'AUC']
        lo    = class_ci_df.loc[cls, 'CI Lower']
        hi    = class_ci_df.loc[cls, 'CI Upper']
        color = PALETTE[cls]

        ax.errorbar(
            x          = est,
            y          = i,
            xerr       = [[est - lo], [hi - est]],
            fmt        = 'o',
            color      = color,
            ecolor     = color,
            elinewidth = 1.5,
            capsize    = 5,
            capthick   = 1.5,
            markersize = 8
        )
        ax.text(hi + 0.002, i,
                f"{est:.2f} ({lo:.2f}–{hi:.2f})",
                va='center', fontsize=9,
                color=color)

    ax.axvline(0.9, color='gray', linewidth=0.8,
               linestyle='--', label='AUC = 0.90')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.set_xlabel('AUC')
    ax.set_title('ResNet1D per-class AUC with 95% bootstrap CI',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0.78, 0.98)
    ax.legend()
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig