import numpy as np
from scipy import stats
import pandas as pd
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

# -- Statistics ------------------------------------------------------------------------
# Delong
def delong_roc_variance(ground_truth: np.ndarray,
                         predictions: np.ndarray) -> tuple:
    """
    Compute AUC and its variance using the DeLong method.
    
    Based on:
    Sun & Xu (2014). Fast implementation of DeLong's algorithm for 
    comparing the areas under correlated receiver operating curves.
    IEEE Signal Processing Letters, 21(11), 1389-1393.

    Args:
        ground_truth (np.ndarray): Binary labels of shape (N,).
        predictions (np.ndarray): Predicted probabilities of shape (N,).

    Returns:
        tuple: (auc, variance)
    """
    from scipy.stats import rankdata

    n = len(ground_truth)
    pos_idx = np.where(ground_truth == 1)[0]
    neg_idx = np.where(ground_truth == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    # Rank predictions
    ranks = rankdata(predictions)

    # AUC via Wilcoxon-Mann-Whitney statistic
    auc = (ranks[pos_idx].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    # Structural components for variance
    V_pos = np.zeros(n_pos)
    V_neg = np.zeros(n_neg)

    for i, pi in enumerate(pos_idx):
        V_pos[i] = np.mean(predictions[neg_idx] < predictions[pi]) + \
                   0.5 * np.mean(predictions[neg_idx] == predictions[pi])

    for j, ni in enumerate(neg_idx):
        V_neg[j] = np.mean(predictions[pos_idx] > predictions[ni]) + \
                   0.5 * np.mean(predictions[pos_idx] == predictions[ni])

    variance = (np.var(V_pos, ddof=1) / n_pos +
                np.var(V_neg, ddof=1) / n_neg)

    return auc, variance

def delong_test(y_true: np.ndarray, y_prob_a: np.ndarray,
                y_prob_b: np.ndarray) -> tuple:
    """
    DeLong test for comparing two AUCs on the same test set.

    Args:
        y_true (np.ndarray): Binary labels of shape (N,).
        y_prob_a (np.ndarray): Probabilities from model A of shape (N,).
        y_prob_b (np.ndarray): Probabilities from model B of shape (N,).

    Returns:
        tuple: (auc_a, auc_b, z_stat, p_value)
    """
    #from scipy import stats

    auc_a, var_a = delong_roc_variance(y_true, y_prob_a)
    auc_b, var_b = delong_roc_variance(y_true, y_prob_b)

    z_stat  = (auc_a - auc_b) / np.sqrt(var_a + var_b)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return auc_a, auc_b, z_stat, p_value

def run_delong_comparison(y_true: np.ndarray,
                           model_probs: dict,
                           class_order: list) -> pd.DataFrame:
    """
    Run DeLong test comparing ResNet1D against each baseline,
    both macro-level and per-class.

    Args:
        y_true (np.ndarray): Binary label matrix of shape (N, n_classes).
        model_probs (dict): Keys are model names, values are prob matrices.
        class_order (list): Class names.

    Returns:
        pd.DataFrame: DeLong results with AUCs, z-stat, p-value, significance.
    """
    results   = []
    baselines = {k: v for k, v in model_probs.items() if k != 'ResNet1D'}
    resnet    = model_probs['ResNet1D']

    for baseline_name, baseline_probs in baselines.items():

        # ── Macro level ───────────────────────────────────────────────────────
        macro_z_stats  = []
        macro_vars_res = []
        macro_vars_bas = []

        for i in range(len(class_order)):
            _, var_res = delong_roc_variance(y_true[:, i], resnet[:, i])
            _, var_bas = delong_roc_variance(y_true[:, i], baseline_probs[:, i])
            macro_vars_res.append(var_res)
            macro_vars_bas.append(var_bas)

        auc_res  = roc_auc_score(y_true, resnet,         average='macro')
        auc_bas  = roc_auc_score(y_true, baseline_probs, average='macro')
        var_res  = np.mean(macro_vars_res)
        var_bas  = np.mean(macro_vars_bas)

        from scipy import stats
        z_stat  = (auc_res - auc_bas) / np.sqrt(var_res + var_bas)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        results.append({
            'Comparison':  f'ResNet1D vs {baseline_name}',
            'Class':       'Macro',
            'AUC ResNet1D': round(auc_res, 4),
            'AUC Baseline': round(auc_bas, 4),
            'Delta AUC':   round(auc_res - auc_bas, 4),
            'Z-stat':      round(z_stat,  4),
            'P-value':     round(p_value, 6),
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })

        # ── Per class ─────────────────────────────────────────────────────────
        for i, cls in enumerate(class_order):
            auc_r, auc_b, z, p = delong_test(
                y_true[:, i], resnet[:, i], baseline_probs[:, i]
            )
            results.append({
                'Comparison':   f'ResNet1D vs {baseline_name}',
                'Class':        cls,
                'AUC ResNet1D': round(auc_r, 4),
                'AUC Baseline': round(auc_b, 4),
                'Delta AUC':    round(auc_r - auc_b, 4),
                'Z-stat':       round(z,     4),
                'P-value':      round(p,     6),
                'Significant':  'Yes' if p < 0.05 else 'No'
            })

    return pd.DataFrame(results).set_index(['Comparison', 'Class'])

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray,
                n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Divides predictions into n_bins equal-width bins and computes
    the weighted average of |mean_predicted - actual_positive_rate|.

    Args:
        y_true (np.ndarray): Binary labels of shape (N,).
        y_prob (np.ndarray): Predicted probabilities of shape (N,).
        n_bins (int): Number of bins. Default 10.

    Returns:
        float: ECE value. Lower is better. 0 = perfect calibration.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece       = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.mean() * abs(acc - conf)

    return round(float(ece), 4)

def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Brier score — mean squared error between
    predicted probabilities and binary labels.

    Args:
        y_true (np.ndarray): Binary labels of shape (N,).
        y_prob (np.ndarray): Predicted probabilities of shape (N,).

    Returns:
        float: Brier score. Lower is better. 0 = perfect.
    """
    return round(float(np.mean((y_prob - y_true) ** 2)), 4)

def compute_calibration_metrics(y_true: np.ndarray,
                                 y_prob: np.ndarray,
                                 class_order: list,
                                 n_bins: int = 10) -> pd.DataFrame:
    """
    Compute ECE and Brier score per class and macro average.

    Args:
        y_true (np.ndarray): Binary label matrix of shape (N, n_classes).
        y_prob (np.ndarray): Probability matrix of shape (N, n_classes).
        class_order (list): Class names.
        n_bins (int): Number of calibration bins. Default 10.

    Returns:
        pd.DataFrame: ECE and Brier score per class with macro row.
    """
    results = []

    for i, cls in enumerate(class_order):
        prevalence    = y_true[:, i].mean()
        baseline_brier = prevalence * (1 - prevalence)  # naive model baseline

        ece    = compute_ece(y_true[:, i],   y_prob[:, i], n_bins)
        brier  = compute_brier(y_true[:, i], y_prob[:, i])

        results.append({
            'Class':           cls,
            'ECE':             ece,
            'Brier':           brier,
            'Baseline Brier':  round(baseline_brier, 4),
            'Brier Skill':     round(1 - brier / baseline_brier, 4),
        })

    df = pd.DataFrame(results).set_index('Class')

    macro = {
        'ECE':            round(df['ECE'].mean(),           4),
        'Brier':          round(df['Brier'].mean(),         4),
        'Baseline Brier': round(df['Baseline Brier'].mean(),4),
        'Brier Skill':    round(df['Brier Skill'].mean(),   4),
    }
    df = pd.concat([df, pd.DataFrame(macro, index=['Macro (mean)'])])

    return df

# -- Subgroup Analysis ----------------------------------------------------
def subgroup_auc(y_true: np.ndarray, y_prob: np.ndarray,
                 mask: np.ndarray, class_order: list,
                 n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Compute macro and per-class AUC with 95% bootstrap CI
    for a subgroup defined by a boolean mask.

    Args:
        y_true (np.ndarray): Binary label matrix (N, n_classes).
        y_prob (np.ndarray): Probability matrix (N, n_classes).
        mask (np.ndarray): Boolean mask of shape (N,) selecting subgroup.
        class_order (list): Class names.
        n_boot (int): Bootstrap resamples. Default 1000.
        seed (int): Random seed. Default 42.

    Returns:
        pd.DataFrame: AUC with 95% CI per class and macro.
    """
    rng        = np.random.default_rng(seed)
    y_sub      = y_true[mask]
    p_sub      = y_prob[mask]
    n          = len(y_sub)
    results    = []

    for i, cls in enumerate(class_order):
        # Skip if only one class present
        if y_sub[:, i].sum() == 0 or y_sub[:, i].sum() == n:
            results.append({
                'Class':    cls,
                'AUC':      float('nan'),
                'CI Lower': float('nan'),
                'CI Upper': float('nan'),
                'N':        n,
                'N_pos':    int(y_sub[:, i].sum()),
            })
            continue

        point_est = roc_auc_score(y_sub[:, i], p_sub[:, i])
        boot_aucs = []

        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            if y_sub[idx, i].sum() == 0:
                continue
            boot_aucs.append(roc_auc_score(y_sub[idx, i], p_sub[idx, i]))

        boot_aucs = np.array(boot_aucs)
        results.append({
            'Class':    cls,
            'AUC':      round(point_est, 4),
            'CI Lower': round(np.percentile(boot_aucs, 2.5),  4),
            'CI Upper': round(np.percentile(boot_aucs, 97.5), 4),
            'N':        n,
            'N_pos':    int(y_sub[:, i].sum()),
        })

    df_res = pd.DataFrame(results).set_index('Class')

    # Macro row
    valid_aucs = df_res['AUC'].dropna()
    df_res.loc['Macro (mean)'] = {
        'AUC':      round(valid_aucs.mean(), 4),
        'CI Lower': round(df_res['CI Lower'].dropna().mean(), 4),
        'CI Upper': round(df_res['CI Upper'].dropna().mean(), 4),
        'N':        n,
        'N_pos':    int(df_res['N_pos'].sum()),
    }

    return df_res

def run_subgroup_analysis(test_df: pd.DataFrame,
                           y_true: np.ndarray,
                           y_prob: np.ndarray,
                           class_order: list) -> dict:
    """
    Run subgroup analysis by sex, age group, and signal quality.

    Args:
        test_df (pd.DataFrame): Test set metadata aligned with y_true/y_prob.
        y_true (np.ndarray): Binary label matrix (N, n_classes).
        y_prob (np.ndarray): Probability matrix (N, n_classes).
        class_order (list): Class names.

    Returns:
        dict: Keys are subgroup names, values are AUC DataFrames.
    """
    results = {}

    # ── Sex subgroups ─────────────────────────────────────────────────────────
    for sex in ['Female', 'Male']:
        mask             = (test_df['sex_label'] == sex).values
        results[sex]     = subgroup_auc(y_true, y_prob, mask, class_order)

    # ── Age group subgroups ───────────────────────────────────────────────────
    for age_grp in ['<40', '40-60', '60-80', '>80']:
        mask                 = (test_df['age_group'] == age_grp).values
        results[age_grp]     = subgroup_auc(y_true, y_prob, mask, class_order)

    # ── Signal quality subgroups ──────────────────────────────────────────────
    for quality, label in [(False, 'Clean'), (True, 'Noisy')]:
        mask               = (test_df['has_artifact'] == quality).values
        results[label]     = subgroup_auc(y_true, y_prob, mask, class_order)

    return results

def plot_subgroup_comparison(subgroup_results: dict) -> plt.Figure:
    """
    Plot macro AUC with 95% CI across all subgroups.

    Args:
        subgroup_results (dict): Output from run_subgroup_analysis().

    Returns:
        plt.Figure
    """
    SUBGROUP_COLORS = {
        'Female':  '#D4537E',
        'Male':    '#378ADD',
        '<40':     '#1D9E75',
        '40-60':   '#EF9F27',
        '60-80':   '#7F77DD',
        '>80':     '#D85A30',
        'Clean':   '#1D9E75',
        'Noisy':   '#D4537E',
    }

    # Group subgroups for visual separation
    groups = {
        'Sex':           ['Female', 'Male'],
        'Age group':     ['<40', '40-60', '60-80', '>80'],
        'Signal quality':['Clean', 'Noisy'],
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, (group_name, subgroup_names) in zip(axes, groups.items()):
        y_pos = np.arange(len(subgroup_names))

        for i, name in enumerate(subgroup_names):
            df_res = subgroup_results[name]
            macro  = df_res.loc['Macro (mean)']
            est    = macro['AUC']
            lo     = macro['CI Lower']
            hi     = macro['CI Upper']
            n      = int(macro['N'])
            color  = SUBGROUP_COLORS.get(name, '#888780')

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
            ax.text(hi + 0.003, i,
                    f"{est:.3f} (n={n:,})",
                    va='center', fontsize=9, color=color)

        # Overall AUC reference line
        ax.axvline(0.9038, color='gray', linewidth=0.8,
                   linestyle='--', label='Overall AUC=0.904')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(subgroup_names)
        ax.set_xlabel('Macro-AUC')
        ax.set_title(group_name, fontsize=11, fontweight='bold')
        ax.set_xlim(0.78, 0.98)
        ax.legend(fontsize=8)
        sns.despine(ax=ax)

    fig.suptitle('Subgroup analysis — ResNet1D macro-AUC with 95% CI',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig

# -- Reliability ----------------------------------------------------------------
def plot_reliability_diagrams(y_true: np.ndarray,
                               model_probs: dict,
                               class_order: list,
                               n_bins: int = 10) -> plt.Figure:
    """
    Plot reliability diagrams (calibration curves) for all models,
    one subplot per class plus macro average.

    Args:
        y_true (np.ndarray): Binary label matrix of shape (N, n_classes).
        model_probs (dict): Keys are model names, values are prob matrices.
        class_order (list): Class names.
        n_bins (int): Number of calibration bins. Default 10.

    Returns:
        plt.Figure
    """
    # Define model colors
    MODEL_COLORS = {
        'Logistic Regression': '#888780',
        'Random Forest':       '#EF9F27',
        'ResNet1D':            '#378ADD',
    }
    # Define number of classes 
    n_classes = len(class_order)
    # Create figure containing subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes      = axes.flatten() # Flatten axes to iterate on them 

    for i, cls in enumerate(class_order):
        ax = axes[i] # Access on of the axes
        bin_edges = np.linspace(0, 1, n_bins + 1) 

        for model_name, probs in model_probs.items():
            bin_confs = []
            bin_accs  = []

            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                mask = (probs[:, i] >= lo) & (probs[:, i] < hi)
                if mask.sum() == 0:
                    continue
                bin_confs.append(probs[:, i][mask].mean())
                bin_accs.append(y_true[:, i][mask].mean())

            ece = compute_ece(y_true[:, i], probs[:, i], n_bins)
            ax.plot(bin_confs, bin_accs,
                    color     = MODEL_COLORS[model_name],
                    linewidth = 1.5,
                    marker    = 'o',
                    markersize= 4,
                    label     = f"{model_name} (ECE={ece:.3f})")

        ax.plot([0, 1], [0, 1], color='gray', linewidth=0.8,
                linestyle='--', label='Perfect calibration')
        ax.fill_between([0, 1], [0, 1], [0, 1],
                         alpha=0.05, color='gray')
        ax.set_title(cls, fontsize=11, fontweight='bold')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Actual positive rate')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc='upper left')
        sns.despine(ax=ax)

    # ── Macro reliability diagram — average across classes ────────────────────
    ax        = axes[5]
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for model_name, probs in model_probs.items():
        all_bin_confs = []
        all_bin_accs  = []

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            confs = []
            accs  = []
            for i in range(n_classes):
                mask = (probs[:, i] >= lo) & (probs[:, i] < hi)
                if mask.sum() == 0:
                    continue
                confs.append(probs[:, i][mask].mean())
                accs.append(y_true[:, i][mask].mean())
            if confs:
                all_bin_confs.append(np.mean(confs))
                all_bin_accs.append(np.mean(accs))

        macro_ece = np.mean([
            compute_ece(y_true[:, i], probs[:, i], n_bins)
            for i in range(n_classes)
        ])
        ax.plot(all_bin_confs, all_bin_accs,
                color     = MODEL_COLORS[model_name],
                linewidth = 1.5,
                marker    = 'o',
                markersize= 4,
                label     = f"{model_name} (ECE={macro_ece:.3f})")

    ax.plot([0, 1], [0, 1], color='gray', linewidth=0.8,
            linestyle='--', label='Perfect calibration')
    ax.set_title('Macro (mean)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Actual positive rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc='upper left')
    sns.despine(ax=ax)

    fig.suptitle('Reliability diagrams — calibration comparison',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig

def plot_calibration_comparison(y_true: np.ndarray,
                                 probs_before: np.ndarray,
                                 probs_after: np.ndarray,
                                 class_order: list,
                                 n_bins: int = 10) -> plt.Figure:
    """
    Plot reliability diagrams before and after temperature scaling
    for each class side by side.

    Args:
        y_true (np.ndarray): Binary label matrix (N, n_classes).
        probs_before (np.ndarray): Uncalibrated probabilities (N, n_classes).
        probs_after (np.ndarray): Calibrated probabilities (N, n_classes).
        class_order (list): Class names.
        n_bins (int): Number of bins. Default 10.

    Returns:
        plt.Figure
    """
    n_classes = len(class_order)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes      = axes.flatten()
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for i, cls in enumerate(class_order):
        ax = axes[i]

        for probs, label, color in [
            (probs_before, 'Before (ECE={:.3f})'.format(
                compute_ece(y_true[:, i], probs_before[:, i])), '#D4537E'),
            (probs_after,  'After  (ECE={:.3f})'.format(
                compute_ece(y_true[:, i], probs_after[:, i])),  '#378ADD'),
        ]:
            bin_confs = []
            bin_accs  = []
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                mask = (probs[:, i] >= lo) & (probs[:, i] < hi)
                if mask.sum() == 0:
                    continue
                bin_confs.append(probs[:, i][mask].mean())
                bin_accs.append(y_true[:, i][mask].mean())

            ax.plot(bin_confs, bin_accs, color=color,
                    linewidth=1.5, marker='o', markersize=4, label=label)

        ax.plot([0, 1], [0, 1], color='gray', linewidth=0.8,
                linestyle='--', label='Perfect calibration')
        ax.set_title(cls, fontsize=11, fontweight='bold')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Actual positive rate')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc='upper left')
        sns.despine(ax=ax)

    # ── Macro subplot ─────────────────────────────────────────────────────────
    ax = axes[5]
    for probs, label, color in [
        (probs_before, 'Before (ECE={:.3f})'.format(
            np.mean([compute_ece(y_true[:, i], probs_before[:, i])
                     for i in range(n_classes)])), '#D4537E'),
        (probs_after,  'After  (ECE={:.3f})'.format(
            np.mean([compute_ece(y_true[:, i], probs_after[:, i])
                     for i in range(n_classes)])),  '#378ADD'),
    ]:
        all_confs = []
        all_accs  = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            confs = []; accs = []
            for i in range(n_classes):
                mask = (probs[:, i] >= lo) & (probs[:, i] < hi)
                if mask.sum() == 0:
                    continue
                confs.append(probs[:, i][mask].mean())
                accs.append(y_true[:, i][mask].mean())
            if confs:
                all_confs.append(np.mean(confs))
                all_accs.append(np.mean(accs))

        ax.plot(all_confs, all_accs, color=color,
                linewidth=1.5, marker='o', markersize=4, label=label)

    ax.plot([0, 1], [0, 1], color='gray', linewidth=0.8,
            linestyle='--', label='Perfect calibration')
    ax.set_title('Macro (mean)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Actual positive rate')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc='upper left')
    sns.despine(ax=ax)

    fig.suptitle('Reliability diagrams — before vs after temperature scaling',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig

