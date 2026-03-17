# EDA Notebook Gameplan
## `eda.ipynb` — PTB-XL ECG AI Statistical Validation

---

## Notebook Structure Overview

```
eda.ipynb
├── 0. Setup & Configuration
├── 1. Data Loading & Integrity Checks
├── 2. Metadata EDA
│   ├── 2.1 Cohort Summary (Table 1)
│   ├── 2.2 Demographics by Superclass
│   └── 2.3 Label Analysis
├── 3. Signal EDA
│   ├── 3.1 Raw Waveform Visualization
│   ├── 3.2 Signal Quality
│   ├── 3.3 Amplitude & Statistical Distributions
│   ├── 3.4 Frequency Content (PSD)
│   └── 3.5 Preprocessing Validation
└── 4. EDA Summary & Modeling Decisions
```

---

## 0. Setup & Configuration

**Purpose:** Establish reproducibility, imports, and constants used throughout the notebook.

```python
# Key items to define here
SEED        = 42
PATH        = './ptbxl/'
FS_LOW      = 100     # Hz — use for EDA and baseline models
FS_HIGH     = 500     # Hz — use for final ResNet1D training
CLASS_ORDER = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
COLORS      = ['#378ADD', '#1D9E75', '#EF9F27', '#D4537E', '#7F77DD']
LEADS       = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
```

**Checklist:**
- [ ] Set random seed globally (`np.random.seed`, `random.seed`)
- [ ] Import all libraries (`pandas`, `numpy`, `wfdb`, `matplotlib`, `seaborn`, `scipy`, `sklearn`)
- [ ] Define module-level constants (`CLASS_ORDER`, `COLORS`, `LEADS`)
- [ ] Configure matplotlib style (`sns.set_style`, `plt.rcParams`)

---

## 1. Data Loading & Integrity Checks

**Purpose:** Load raw data, apply the author-recommended splits, and verify no data integrity issues exist before any analysis.

### 1.1 Load Metadata CSV

```python
df = pd.read_csv(PATH + 'ptbxl_database.csv', index_col='ecg_id')
df.scp_codes = df.scp_codes.apply(ast.literal_eval)
```

**Checks to run:**
- [ ] Shape of dataframe — expect (21799, 28)
- [ ] Column names and dtypes
- [ ] Missing value counts per column — `df.isnull().sum()`
- [ ] Duplicate `ecg_id` check — `df.index.duplicated().sum()` should be 0
- [ ] Duplicate `patient_id` across folds — each patient must appear in exactly one fold group

### 1.2 Apply Author-Recommended Splits

```python
df['split'] = 'train'
df.loc[df['strat_fold'] == 9,  'split'] = 'val'
df.loc[df['strat_fold'] == 10, 'split'] = 'test'
```

**Checks to run:**
- [ ] Patient-level leakage assert — `set(train patient_ids) ∩ set(test patient_ids) == ∅`
- [ ] Split sizes — expect ~17,440 train / ~1,100 val / ~2,200 test (approximate)
- [ ] `strat_fold` value counts — should be 10 folds roughly equal in size

### 1.3 Map SCP Codes to Diagnostic Superclasses

```python
agg = pd.read_csv(PATH + 'scp_statements.csv', index_col=0)
agg = agg[agg.diagnostic == 1]
df['superclass'] = df.scp_codes.apply(get_superclass)
```

**Checks to run:**
- [ ] Count of records with empty superclass (no diagnostic label) — these will be excluded
- [ ] Value counts of `superclass` list lengths — confirm multi-label distribution
- [ ] Confirm 5 expected superclasses present and no unexpected values

### 1.4 Age HIPAA Correction

```python
df.loc[df['age'] >= 89, 'age'] = 90
```

**Checks to run:**
- [ ] Count of records affected — `(df['age'] >= 89).sum()` before and after
- [ ] Age range after correction — should be 0–90

---

## 2. Metadata EDA

### 2.1 Cohort Summary — Table 1

**Purpose:** Produce the clinical Table 1 describing the full cohort and each split. This is a required table in any clinical AI paper.

**Functions:** `cohort_summary_stats()`, `plot_cohort_summary_stats()`

**Run on all splits:**
```python
split_df = {
    'global': df,
    'train':  df[df['split'] == 'train'],
    'val':    df[df['split'] == 'val'],
    'test':   df[df['split'] == 'test'],
}

for key, df_split in split_df.items():
    display(cohort_summary_stats(df_split))
```

**Plots produced:**
| Plot | Key | What to look for |
|---|---|---|
| Age histogram + KDE | `age_histogram` | Shape, modality, median line |
| Sex distribution bar | `sex_bar` | ~52/48 M/F globally |
| Diagnosis prevalence | `diagnosis_prevalence` | NORM dominant, HYP rarest |
| Age by sex KDE | `age_by_sex_kde` | Overlap between sexes |

**Key questions to answer:**
- Are train/val/test demographically similar? Flag any meaningful differences.
- Is the age distribution unimodal or bimodal? (NORM pulling younger)
- Are diagnosis prevalences consistent across splits?

---

### 2.2 Demographics by Superclass — Table 2

**Purpose:** Characterize each diagnostic superclass by age and sex. Motivates subgroup analysis in the modeling phase.

**Functions:** `demographics_by_superclass()`, `plot_demographics_by_superclass()`

```python
for key, df_split in split_df.items():
    stats, df_exploded = demographics_by_superclass(df_split)
    plots = plot_demographics_by_superclass(stats, df_exploded)
    display(stats)
```

**Plots produced:**
| Plot | Key | What to look for |
|---|---|---|
| Age violin + boxplot | `age_violin` | NORM visibly younger, wider spread |
| Sex stacked bar | `sex_stacked_bar` | MI male skew (~59% male) |
| Record counts per class | `class_counts` | Class imbalance magnitude |
| Age × sex grouped boxplot | `age_sex_interaction` | Whether MI male skew is age-confounded |

**Key findings to document:**
- NORM median age (~54) vs pathological classes (~67–68) — document this gap
- MI male skew — note this as a subgroup analysis motivation
- HYP rarest class — note implications for model sensitivity and pos_weight

---

### 2.3 Label Analysis

**Purpose:** Understand multi-label structure, co-occurrence patterns, and class imbalance. Feeds directly into loss function design.

#### 2.3.1 Labels Per Record Distribution

```python
df['n_labels'] = df['superclass'].apply(len)
df['n_labels'].value_counts().sort_index()
```

**What to look for:** What fraction of records are truly multi-label vs single-label? If >80% are single-label, multi-label modeling is still necessary but the co-occurrence analysis becomes even more important.

#### 2.3.2 Label Co-occurrence Matrix

```python
# Build co-occurrence matrix then plot
cooc_matrix = build_cooccurrence_matrix(df[df['split'] == 'train'])
fig = plot_cooccurrence(cooc_matrix)
```

**What to look for:**
- MI ↔ STTC: expect highest off-diagonal co-occurrence (~48–50%)
- NORM should have near-zero co-occurrence with pathological classes
- Any unexpected high co-occurrences warrant investigation

#### 2.3.3 Class Imbalance Ratio (Training Split Only)

```python
# Compute pos_weight for BCEWithLogitsLoss — save for modeling notebook
Y_train    = mlb.fit_transform(df[df['split'] == 'train']['superclass'])
pos        = Y_train.sum(axis=0)
neg        = len(Y_train) - pos
pos_weight = neg / pos
print(dict(zip(CLASS_ORDER, pos_weight.round(2))))
```

**Save this output** — you will paste it directly into your modeling notebook as the `pos_weight` tensor.

---

## 3. Signal EDA

### 3.1 Raw Waveform Visualization

**Purpose:** Visually confirm the data loads correctly, leads are ordered as expected, and signals look like diagnostic-quality ECGs.

#### 3.1.1 One Example Per Superclass

```python
def plot_ecg(signal, title, fs=100, leads=LEADS):
    """
    Plot a 12-lead ECG from a (timesteps, 12) numpy array.
    """
    fig, axes = plt.subplots(12, 1, figsize=(14, 16), sharex=True)
    t = np.arange(signal.shape[0]) / fs
    for i, (ax, lead) in enumerate(zip(axes, leads)):
        ax.plot(t, signal[:, i], linewidth=0.7, color='#1a1a1a')
        ax.set_ylabel(lead, fontsize=9, rotation=0, labelpad=24)
        ax.set_ylim(signal[:, i].mean() - 2, signal[:, i].mean() + 2)
        sns.despine(ax=ax, bottom=(i < 11))
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig
```

**Run for:** One representative record per superclass, sampled from the training set.

**What to verify:**
- [ ] Signal spans ~10 seconds at 100 Hz (1000 timesteps)
- [ ] 12 leads present and in correct order (I, II, III, aVR, aVL, aVF, V1–V6)
- [ ] Amplitude in expected range (~mV scale, roughly -2 to +2 mV)
- [ ] No all-zero leads
- [ ] QRS complexes are visible and recognizable

#### 3.1.2 Clean vs Noisy Record Comparison

Plot one record flagged with `static_noise=1` and one clean record side by side on the same lead (Lead II). This makes the quality flag concrete and justifies any quality-based filtering decisions.

---

### 3.2 Signal Quality

**Purpose:** Quantify how quality flags distribute across splits and understand the potential impact on model training and evaluation.

#### 3.2.1 Quality Flag Prevalence Per Split

```python
quality_cols = ['static_noise', 'burst_noise', 'baseline_drift',
                'electrodes_problems', 'extra_beats', 'pacemaker']

quality_summary = (
    df.groupby('split')[quality_cols]
    .apply(lambda g: (g.notna() & (g != '')).mean() * 100)
    .round(2)
)
```

**What to look for:** Any quality flag that is disproportionately concentrated in test vs train. A test set with substantially more noisy records than train would confound performance estimates.

#### 3.2.2 Validated-by-Human Rate Per Split

```python
df.groupby('split')['validated_by_human'].mean().round(3)
```

**Expected:** Test and val should have substantially higher rates than train, confirming folds 9–10 are higher quality.

#### 3.2.3 Quality vs Superclass

```python
df_exploded.groupby('superclass')[quality_cols].apply(
    lambda g: (g.notna() & (g != '')).mean() * 100
).round(2)
```

**What to look for:** Are noisy records concentrated in any particular superclass? This would be a confound in per-class performance analysis.

---

### 3.3 Amplitude & Statistical Distributions

**Purpose:** Understand the raw signal statistics to inform normalization strategy and detect outliers before training.

#### 3.3.1 Per-Lead Amplitude Statistics

```python
def compute_lead_stats(X, leads=LEADS):
    """
    Compute mean, std, min, max per lead across a sample of records.
    X shape: (N, timesteps, 12)
    """
    stats = []
    for i, lead in enumerate(leads):
        lead_data = X[:, :, i].flatten()
        stats.append({
            'lead':   lead,
            'mean':   lead_data.mean(),
            'std':    lead_data.std(),
            'min':    lead_data.min(),
            'max':    lead_data.max(),
            'p01':    np.percentile(lead_data, 1),
            'p99':    np.percentile(lead_data, 99),
        })
    return pd.DataFrame(stats).set_index('lead')
```

**Run on a random sample** (~500 records) from the training set. Loading all 17k records for EDA is unnecessary.

**What to look for:**
- Leads with substantially different amplitude ranges — may indicate electrode placement conventions
- Any lead with near-zero std (flat lead) — would be a data issue
- Extreme min/max values — outlier records to investigate

#### 3.3.2 Record-Level Outlier Detection

```python
# Flag records with extreme global amplitude
record_max = X_sample.max(axis=(1,2))
record_min = X_sample.min(axis=(1,2))

# Any record with amplitude > 5 mV is suspicious
outlier_mask = (record_max > 5) | (record_min < -5)
print(f"Potential outlier records: {outlier_mask.sum()}")
```

**Save the IDs** of outlier records for review. Do not remove them without inspection.

#### 3.3.3 Normalization Strategy Decision

Based on the amplitude stats, decide and document which normalization to use:

| Strategy | Formula | When to use |
|---|---|---|
| Per-record z-score | `(x - mean) / std` | Default — removes amplitude variation between patients |
| Per-lead z-score | Per-lead mean/std | If leads have very different amplitude distributions |
| Min-max | `(x - min) / (max - min)` | Less robust to outliers — generally avoid |
| None | — | Only if using batch norm that handles it internally |

**Recommendation for PTB-XL:** Per-record z-score across all leads jointly.

---

### 3.4 Frequency Content (Power Spectral Density)

**Purpose:** Confirm signal bandwidth and justify bandpass filter cutoffs before preprocessing.

#### 3.4.1 Mean PSD Per Lead

```python
from scipy.signal import welch

def plot_psd(X_sample, fs=100, leads=LEADS):
    """
    Plot mean power spectral density across a sample of records.
    X_sample shape: (N, timesteps, 12)
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (ax, lead) in enumerate(zip(axes, leads)):
        psds = []
        for record in X_sample:
            f, psd = welch(record[:, i], fs=fs, nperseg=256)
            psds.append(psd)
        mean_psd = np.stack(psds).mean(axis=0)
        ax.semilogy(f, mean_psd, color=COLORS[i % len(COLORS)], linewidth=1.2)
        ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, label='0.5 Hz')
        ax.axvline(40,  color='red',  linestyle='--', linewidth=0.8, label='40 Hz')
        ax.set_title(lead, fontsize=10)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (mV²/Hz)')
        sns.despine(ax=ax)

    axes[0].legend(fontsize=8)
    fig.suptitle('Mean power spectral density per lead (training sample)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig
```

**What to look for:**
- Majority of diagnostic signal power between 0.5–40 Hz — confirms standard bandpass cutoffs
- Baseline wander visible as power spike near 0 Hz — confirms need for high-pass filter
- Power line interference spike at 50 or 60 Hz — note but do not over-filter (already below Nyquist at 100 Hz)

---

### 3.5 Preprocessing Validation

**Purpose:** Visually and quantitatively confirm your preprocessing pipeline works correctly before wiring it into the Dataset class.

#### 3.5.1 Bandpass Filter Implementation

```python
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=100, order=4):
    """
    Apply zero-phase Butterworth bandpass filter.
    signal shape: (timesteps, 12)
    """
    nyq  = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal, axis=0)
```

#### 3.5.2 Per-Record Z-Score Normalization

```python
def normalize(signal):
    """
    Per-record z-score normalization across all leads jointly.
    signal shape: (timesteps, 12)
    """
    mean = signal.mean()
    std  = signal.std() + 1e-8  # epsilon prevents division by zero
    return (signal - mean) / std
```

#### 3.5.3 Before/After Visualization

For a single record (use the Lead II trace):
- [ ] Plot raw signal
- [ ] Plot after bandpass filter — baseline wander removed, high-frequency noise reduced
- [ ] Plot after normalization — amplitude centered around 0, scaled to unit variance
- [ ] Overlay all three on the same axis for direct comparison

**What to verify:**
- QRS morphology is preserved after filtering
- Normalization does not clip or distort the signal shape
- The pipeline is deterministic — same input always produces same output

#### 3.5.4 Full Pipeline Function

```python
def preprocess(signal, fs=100):
    """
    Full preprocessing pipeline: filter → normalize.
    signal shape: (timesteps, 12)
    Returns: (timesteps, 12) float32 array
    """
    signal = bandpass_filter(signal, fs=fs)
    signal = normalize(signal)
    return signal.astype(np.float32)
```

**Test on 10 records and print shape, dtype, mean, std to confirm output is as expected.**

---

## 4. EDA Summary & Modeling Decisions

**Purpose:** Consolidate all findings into a single reference cell. This becomes the bridge between your EDA and modeling notebooks.

### 4.1 Key Findings Table

Document these explicitly in a markdown cell:

| Finding | Value | Modeling implication |
|---|---|---|
| Total training records | ~17,440 | Batch size and epoch planning |
| Class imbalance (HYP) | 12.4% prevalence | `pos_weight` required |
| MI ↔ STTC co-occurrence | ~48–50% | Multi-label sigmoid output confirmed |
| NORM median age | ~54 vs ~67 pathological | Age is a confounder — include in subgroup analysis |
| MI male skew | ~59% male | Sex subgroup analysis required for MI specifically |
| Validated-by-human (test) | High rate | Test set label quality is high — results are trustworthy |
| Noisy records in test | % from quality EDA | Flag if disproportionate |
| Dominant signal bandwidth | 0.5–40 Hz | Confirmed bandpass cutoffs |
| Normalization strategy | Per-record z-score | Document in model methods section |

### 4.2 Decisions to Carry Forward

```python
# Paste these directly into your modeling notebook header

# Splits
# Folds 1-8 = train, fold 9 = val, fold 10 = test

# Class order
CLASS_ORDER = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

# pos_weight for BCEWithLogitsLoss (computed on training split only)
# pos_weight = [value from section 2.3.3]

# Sampling rate
FS = 100  # upgrade to 500 for final model run

# Preprocessing
# bandpass: 0.5–40 Hz, Butterworth order 4, zero-phase
# normalization: per-record z-score

# Input shape to model
# (N, 12, 1000)  — (batch, leads, timesteps) at 100 Hz
```

### 4.3 Records to Flag or Exclude

Document any records identified as outliers in section 3.3.2. State whether they will be excluded from training and on what grounds.

---

## Appendix: Helper Functions Index

| Function | Location | Purpose |
|---|---|---|
| `cohort_summary_stats()` | Cell X | Table 1 — global cohort stats |
| `plot_cohort_summary_stats()` | Cell X | Plots for Table 1 |
| `demographics_by_superclass()` | Cell X | Table 2 — age/sex by class |
| `plot_demographics_by_superclass()` | Cell X | Plots for Table 2 |
| `plot_cooccurrence()` | Cell X | Co-occurrence heatmap |
| `plot_ecg()` | Cell X | 12-lead ECG waveform plot |
| `compute_lead_stats()` | Cell X | Per-lead amplitude statistics |
| `plot_psd()` | Cell X | Power spectral density per lead |
| `bandpass_filter()` | Cell X | Butterworth bandpass filter |
| `normalize()` | Cell X | Per-record z-score normalization |
| `preprocess()` | Cell X | Full preprocessing pipeline |

---

*EDA Gameplan — PTB-XL ECG AI Statistical Validation*
*Update cell references in the Appendix once notebook is finalized.*