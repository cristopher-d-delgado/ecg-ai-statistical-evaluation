# modules/eda_metadata.py

"""
Metadata EDA functions for PTB-XL ECG dataset.

Functions:
    cohort_summary_stats: Generate Table 1 cohort summary statistics.
    plot_cohort_summary_stats: Visualize cohort-level summary statistics.
    demographics_by_superclass: Compute age and sex statistics by superclass.
    plot_demographics_by_superclass: Visualize demographics by superclass.
    plot_cooccurrence: Plot label co-occurrence heatmap.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# Module-level constants
CLASS_ORDER = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
<<<<<<< HEAD
COLORS      = ['#378ADD', '#1D9E75', "#0C0C0B", '#D4537E', '#7F77DD']
=======
COLORS      = ['#378ADD', '#1D9E75', '#EF9F27', '#D4537E', '#7F77DD']
>>>>>>> 2c26a6724fe6b702fb05f8977181dc5d9620046f
SEX_PALETTE = {'Female': '#D4537E', 'Male': '#378ADD'}


def cohort_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a Table 1 cohort summary for the PTB-XL dataset.

    Args:
        df (pd.DataFrame): Dataframe containing 'patient_id', 'age', 'sex',
                           and 'superclass' columns.

    Returns:
        pd.DataFrame: Summary table with columns ['Characteristic', 'Value'].

    Example:
        >>> table = cohort_summary_stats(df[df['split'] == 'train'])
        >>> display(table)
    """
    df = df.copy()

    # -- Cohort size --
    n_records  = df.shape[0]
    n_patients = df['patient_id'].nunique()

    # -- Age --
    # Correct HIPAA-compliant encoding: age > 89 stored as 300 in PTB-XL
    df.loc[df['age'] >= 89, 'age'] = 90 # Create Assumption that they are 90 years old

    age_mean   = df['age'].mean()
    age_sd     = df['age'].std(ddof=1)
    age_median = df['age'].median()
    age_q1     = df['age'].quantile(0.25)
    age_q3     = df['age'].quantile(0.75)

    # -- Sex --
    # PTB-XL encodes 1 = Female, 0 = Male
    sex_counts  = (df['sex']
                   .value_counts()
                   .rename({1: 'Female', 0: 'Male'})
                   .reindex(['Female', 'Male'], fill_value=0))
    sex_percent = sex_counts / n_records * 100

    # -- Diagnostic superclasses --
    # Multi-label: % = records containing this class / total records
    all_classes   = [cls for sublist in df['superclass'] for cls in sublist]
    class_counts  = (pd.Series(all_classes)
                     .value_counts()
                     .reindex(CLASS_ORDER, fill_value=0))
    class_percent = class_counts / n_records * 100

    # -- Build table --
    rows = []
    rows.append(["Total ECG Recordings", f"{n_records:,}"])
    rows.append(["Unique Patients",       f"{n_patients:,}"])
    rows.append(["Age (years), mean ± SD",
                 f"{age_mean:.1f} ± {age_sd:.1f}"])
    rows.append(["Age (years), median (IQR)",
                 f"{age_median:.1f} ({age_q1:.1f}–{age_q3:.1f})"])

    for sex in sex_counts.index:
        rows.append([f"Sex: {sex}",
                     f"{sex_counts[sex]:,} ({sex_percent[sex]:.1f}%)"])

    for cls in class_counts.index:
        rows.append([f"Diagnosis: {cls}",
                     f"{class_counts[cls]:,} ({class_percent[cls]:.1f}%)"])

    return pd.DataFrame(rows, columns=["Characteristic", "Value"])


def plot_cohort_summary_stats(df: pd.DataFrame, cohort: str) -> dict:
    """
    Visualize cohort-level summary statistics.
    Companion plotting function to cohort_summary_stats().

    Args:
        df (pd.DataFrame): Raw dataframe containing 'age', 'sex',
                           and 'superclass' columns.
        cohort (str): Specify the cohort of 'Global', 'Train', 'Test', 'Val'

    Returns:
        dict: Dictionary of matplotlib figures keyed by plot name:
              'age_histogram', 'sex_bar', 'diagnosis_prevalence', 'age_by_sex_kde'.

    Example:
        >>> plots = plot_cohort_summary_stats(df[df['split'] == 'train'])
        >>> plots['age_histogram'].show()
    """
    cohort_plots = {}

    df = df.copy()
    df.loc[df['age'] >= 89, 'age'] = 90
    df['sex_label'] = df['sex'].map({1: 'Female', 0: 'Male'})
    n_records       = len(df)

    # ── 1. Age histogram with KDE ─────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    sns.histplot(
        data=df, x='age', bins=30,
        color='#378ADD', alpha=0.6,
        kde=True, line_kws={'linewidth': 2},
        ax=ax1
    )
    ax1.axvline(df['age'].median(), color='#D4537E', linewidth=1.5,
                linestyle='--', label=f"Median: {df['age'].median():.0f} yrs")
    ax1.set_title(f'Age distribution — {cohort} cohort', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Age (years)')
    ax1.set_ylabel('Number of records')
    ax1.legend()
    sns.despine(ax=ax1)
    fig1.tight_layout()
    cohort_plots['age_histogram'] = fig1

    # ── 2. Sex distribution bar ───────────────────────────────────────────────
    sex_counts  = df['sex_label'].value_counts().reindex(['Female', 'Male'],
                                                          fill_value=0)
    sex_percent = sex_counts / n_records * 100

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    bars = ax2.bar(
        sex_counts.index, sex_counts.values,
        color=[SEX_PALETTE[s] for s in sex_counts.index],
        edgecolor='none', width=0.5
    )
    for bar, (label, count) in zip(bars, sex_counts.items()):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 80,
                 f'{count:,}\n({sex_percent[label]:.1f}%)',
                 ha='center', va='bottom', fontsize=10)
    ax2.set_title(f'Sex distribution — {cohort} cohort', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Number of records')
    ax2.set_ylim(0, sex_counts.max() * 1.2)
    sns.despine(ax=ax2)
    fig2.tight_layout()
    cohort_plots['sex_bar'] = fig2

    # ── 3. Diagnosis prevalence horizontal bar ────────────────────────────────
    all_classes  = [cls for sublist in df['superclass'] for cls in sublist]
    class_counts = (pd.Series(all_classes)
                    .value_counts()
                    .reindex(CLASS_ORDER, fill_value=0))
    class_pct    = class_counts / n_records * 100

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    bars = ax3.barh(
        class_counts.index, class_counts.values,
        color=COLORS, edgecolor='none', height=0.55
    )
    for bar, (cls, count) in zip(bars, class_counts.items()):
        ax3.text(bar.get_width() + 80,
                 bar.get_y() + bar.get_height() / 2,
                 f'{count:,}  ({class_pct[cls]:.1f}%)',
                 va='center', fontsize=10)
    ax3.set_xlabel('Number of records')
    ax3.set_title(f'Diagnosis prevalence — {cohort} cohort\n'
                  '(multi-label: % of total records)',
                  fontsize=13, fontweight='bold')
    ax3.set_xlim(0, class_counts.max() * 1.25)
    sns.despine(ax=ax3)
    fig3.tight_layout()
    cohort_plots['diagnosis_prevalence'] = fig3

    # ── 4. Age distribution by sex (overlapping KDE) ──────────────────────────
    fig4, ax4 = plt.subplots(figsize=(9, 4))
    for sex, color in SEX_PALETTE.items():
        subset = df[df['sex_label'] == sex]
        sns.kdeplot(
            data=subset, x='age',
            color=color, linewidth=2,
            fill=True, alpha=0.25,
            label=f"{sex} (n={len(subset):,})",
            ax=ax4
        )
    ax4.set_title(f'Age distribution by sex — {cohort} cohort',
                  fontsize=13, fontweight='bold')
    ax4.set_xlabel('Age (years)')
    ax4.set_ylabel('Density')
    ax4.legend(title='Sex')
    sns.despine(ax=ax4)
    fig4.tight_layout()
    cohort_plots['age_by_sex_kde'] = fig4

    return cohort_plots


def demographics_by_superclass(df: pd.DataFrame) -> tuple:
    """
    Compute age and sex statistics stratified by diagnostic superclass.

    Args:
        df (pd.DataFrame): Dataframe containing 'superclass', 'age',
                           'sex', and 'patient_id' columns.

    Returns:
        tuple:
            stats (pd.DataFrame): Formatted age and sex statistics per superclass
                                  with columns ['N (label count)', 'Mean ± SD',
                                  'Median (IQR)', 'Female', 'Male', 'female_pct',
                                  'male_pct', 'n'].
            df_exploded (pd.DataFrame): Exploded dataframe used for downstream
                                        plotting, with added 'sex_label' column.

    Example:
        >>> stats, df_exploded = demographics_by_superclass(df[df['split'] == 'train'])
        >>> display(stats)
    """
    df = df.copy()
    df.loc[df['age'] >= 89, 'age'] = 90

    # Explode so each row = one superclass label
    df_exploded = (df[['patient_id', 'age', 'sex', 'superclass']]
                   .explode('superclass')
                   .dropna(subset=['superclass']))
    df_exploded['sex_label'] = df_exploded['sex'].map({1: 'Female', 0: 'Male'})

    # Age statistics
    age_stats = (
        df_exploded.groupby('superclass')['age']
        .agg(
            mean_age='mean',
            sd_age='std',
            median_age='median',
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
            n='count'
        )
        .round(1)
    )

    # Sex statistics — percentages
    sex_stats = (
        df_exploded.groupby('superclass')['sex']
        .value_counts(normalize=True)
        .unstack()
        .reindex(columns=[1, 0], fill_value=0)
        .rename(columns={1: 'female_pct', 0: 'male_pct'})
        .mul(100)
        .round(1)
    )

    # Sex statistics — counts
    sex_counts = (
        df_exploded.groupby('superclass')['sex']
        .value_counts()
        .unstack()
        .reindex(columns=[1, 0], fill_value=0)
        .rename(columns={1: 'female_n', 0: 'male_n'})
    )

    # Combine
    stats = age_stats.join(sex_stats).join(sex_counts)

    # Formatted display columns
    stats['N (label count)'] = stats['n'].apply(lambda x: f"{x:,}")
    stats['Mean ± SD']       = stats.apply(
        lambda r: f"{r['mean_age']} ± {r['sd_age']}", axis=1)
    stats['Median (IQR)']    = stats.apply(
        lambda r: f"{r['median_age']} ({r['q1']}–{r['q3']})", axis=1)
    stats['Female']          = stats.apply(
        lambda r: f"{r['female_n']:,.0f} ({r['female_pct']}%)", axis=1)
    stats['Male']            = stats.apply(
        lambda r: f"{r['male_n']:,.0f} ({r['male_pct']}%)", axis=1)

    return stats[['N (label count)', 'Mean ± SD', 'Median (IQR)',
                  'Female', 'Male', 'female_pct', 'male_pct', 'n']], df_exploded


def plot_demographics_by_superclass(stats: pd.DataFrame,
                                     df_exploded: pd.DataFrame) -> dict:
    """
    Visualize demographics stratified by diagnostic superclass.
    Consumes outputs directly from demographics_by_superclass().

    Args:
        stats (pd.DataFrame): Formatted stats table from demographics_by_superclass().
        df_exploded (pd.DataFrame): Exploded dataframe from demographics_by_superclass().

    Returns:
        dict: Dictionary of matplotlib figures keyed by plot name:
              'age_violin', 'sex_stacked_bar', 'class_counts', 'age_sex_interaction'.

    Example:
        >>> stats, df_exploded = demographics_by_superclass(df[df['split'] == 'train'])
        >>> plots = plot_demographics_by_superclass(stats, df_exploded)
        >>> plots['age_violin'].show()
    """
    demographic_stat_plots = {}
    PALETTE = dict(zip(CLASS_ORDER, COLORS))

    # ── 1. Age distribution by superclass (violin + boxplot) ──────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.violinplot(
        data=df_exploded, x='superclass', y='age', hue='superclass',
        order=CLASS_ORDER, palette=PALETTE,
        alpha=0.4, legend=False, ax=ax1
    )
    sns.boxplot(
        data=df_exploded, x='superclass', y='age',
        order=CLASS_ORDER, width=0.15,
        color='white', linewidth=1.2,
        ax=ax1
    )
    ax1.set_title('Age distribution by diagnostic superclass',
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel('Superclass')
    ax1.set_ylabel('Age (years)')
    ax1.set_ylim(0, 95)
    sns.despine(ax=ax1)
    fig1.tight_layout()
    demographic_stat_plots['age_violin'] = fig1

    # ── 2. Sex breakdown by superclass (stacked horizontal bar) ───────────────
    sex_pct = (
        stats.reindex(CLASS_ORDER)[['female_pct', 'male_pct']]
        .rename(columns={'female_pct': 'Female', 'male_pct': 'Male'})
    )
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sex_pct.plot(
        kind='barh', stacked=True,
        color=[SEX_PALETTE['Female'], SEX_PALETTE['Male']],
        ax=ax2, edgecolor='none'
    )
    ax2.axvline(50, color='white', linewidth=1.2, linestyle='--')
    ax2.set_xlabel('Percentage (%)')
    ax2.set_ylabel('')
    ax2.set_title('Sex distribution by diagnostic superclass',
                  fontsize=13, fontweight='bold')
    ax2.legend(title='Sex', bbox_to_anchor=(1.01, 1), loc='upper left')
    ax2.set_xlim(0, 100)
    sns.despine(ax=ax2)
    fig2.tight_layout()
    demographic_stat_plots['sex_stacked_bar'] = fig2

    # ── 3. Record counts per superclass ───────────────────────────────────────
    counts = stats.reindex(CLASS_ORDER)['n']
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    bars = ax3.barh(CLASS_ORDER, counts.values,
                    color=COLORS, edgecolor='none', height=0.55)
    for bar, val in zip(bars, counts.values):
        ax3.text(bar.get_width() + 80,
                 bar.get_y() + bar.get_height() / 2,
                 f'{val:,}', va='center', fontsize=10)
    ax3.set_xlabel('Number of records (label count)')
    ax3.set_title('Record counts per diagnostic superclass',
                  fontsize=13, fontweight='bold')
    ax3.set_xlim(0, counts.max() * 1.15)
    sns.despine(ax=ax3)
    fig3.tight_layout()
    demographic_stat_plots['class_counts'] = fig3

    # ── 4. Age × sex interaction per superclass (grouped boxplot) ─────────────
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=df_exploded, x='superclass', y='age', hue='sex_label',
        order=CLASS_ORDER, palette=SEX_PALETTE,
        width=0.5, fliersize=2, linewidth=1.0, ax=ax4
    )
    ax4.set_title('Age by superclass and sex',
                  fontsize=13, fontweight='bold')
    ax4.set_xlabel('Superclass')
    ax4.set_ylabel('Age (years)')
    ax4.legend(title='Sex', bbox_to_anchor=(1.01, 1), loc='upper left')
    sns.despine(ax=ax4)
    fig4.tight_layout()
    demographic_stat_plots['age_sex_interaction'] = fig4

    return demographic_stat_plots


def plot_cooccurrence(cooc_matrix: pd.DataFrame) -> plt.Figure:
    """
    Plot label co-occurrence heatmap with upper triangle masked.

    Normalizes co-occurrence counts by row class size so each cell represents
    the percentage of row-class records that also carry the column class label.
    The diagonal represents 100% by definition.

    Args:
        cooc_matrix (pd.DataFrame): Square co-occurrence matrix with superclass
                                     labels as both index and columns.

    Returns:
        plt.Figure

    Example:
        >>> fig = plot_cooccurrence(cooc_matrix)
        >>> fig.show()
    """
    # Normalize by row (% of row class records that also have column class)
    cooc_pct = cooc_matrix.div(cooc_matrix.values.diagonal(), axis=0) * 100

    # Mask upper triangle excluding diagonal
    mask = np.triu(np.ones_like(cooc_pct, dtype=bool), k=1)

    # Trim colormap to avoid near-white at low end
    from matplotlib.colors import LinearSegmentedColormap
    reds_trimmed = LinearSegmentedColormap.from_list(
        'reds_trimmed',
        plt.cm.Reds(np.linspace(0.15, 1.0, 256))
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cooc_pct,
        mask=mask,
        annot=True, fmt='.1f',
        cmap=reds_trimmed,
        vmin=0, vmax=100,
        linewidths=0.5,
        linecolor='white',
        ax=ax
    )
    ax.set_title('Label co-occurrence\n'
                 '(% of row class records also having column class)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    fig.tight_layout()
    return fig