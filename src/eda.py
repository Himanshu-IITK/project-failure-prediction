import os
import matplotlib
matplotlib.use('Agg')  # Prevents GUI popups when running from terminal
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from features import engineer_features


def run_eda(input_path: str = "data/raw/project_management_data.csv",
            output_dir: str = "reports/figures") -> None:
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # Load raw data and apply PM feature engineering
    df = pd.read_csv(input_path)
    df = engineer_features(df)

    print("=== Dataset Shape ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

    # 1. Target Class Distribution
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x='project_failed', hue='project_failed', legend=False, palette=['#2ecc71', '#e74c3c'])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Success (0)', 'Failure (1)'])
    ax.set_title('Project Outcome Distribution (Imbalanced Target)')
    ax.set_ylabel('Number of Projects')
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_class_distribution.png"), dpi=150)
    plt.close()

    # 2. Correlation Heatmap (Numerical & Engineered Features)
    numeric_cols = df.select_dtypes(include=['number']).columns
    corr = df[numeric_cols].corr()

    print("=== Top Correlations with Project Failure ===")
    print(corr['project_failed'].sort_values(ascending=False).round(3))

    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
    plt.title('Feature Correlation Matrix (Raw + Engineered PM Features)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_correlation_heatmap.png"), dpi=150)
    plt.close()

    # 3. CPI & SPI Distributions by Project Outcome
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Clip extreme CPI outliers only for cleaner visualization
    sns.boxplot(data=df, x='project_failed', y=df['cpi'].clip(0, 2.0),
                hue='project_failed', legend=False, palette=['#2ecc71', '#e74c3c'], ax=axes[0])
    axes[0].axhline(1.0, color='black', linestyle='--', linewidth=1, label='On Budget (CPI = 1.0)')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Success (0)', 'Failure (1)'])
    axes[0].set_title('Cost Performance Index (CPI) vs Outcome')
    axes[0].legend()

    sns.boxplot(data=df, x='project_failed', y='spi',
                hue='project_failed', legend=False, palette=['#2ecc71', '#e74c3c'], ax=axes[1])
    axes[1].axhline(1.0, color='black', linestyle='--', linewidth=1, label='On Schedule (SPI = 1.0)')
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Success (0)', 'Failure (1)'])
    axes[1].set_title('Schedule Performance Index (SPI) vs Outcome')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_cpi_spi_vs_failure.png"), dpi=150)
    plt.close()

    # 4. Failure Rate by Methodology
    plt.figure(figsize=(8, 4.5))
    meth_failure = df.groupby('methodology')['project_failed'].mean().sort_values(ascending=False) * 100
    ax = sns.barplot(x=meth_failure.index, y=meth_failure.values, hue=meth_failure.index, legend=False, palette="viridis")
    ax.set_title('Project Failure Rate (%) by Methodology')
    ax.set_ylabel('Failure Rate (%)')
    ax.set_xlabel('Methodology')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_methodology_failure_rate.png"), dpi=150)
    plt.close()

    print(f"\nAll 4 EDA figures saved successfully to: {output_dir}")


if __name__ == "__main__":
    run_eda()