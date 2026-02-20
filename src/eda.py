import matplotlib.pyplot as plt
import seaborn as sns
import os


def run_eda(df, save_path='outputs/plots/'):
    """
    Generate all exploratory charts and save them as PNG files.
    """
    os.makedirs(save_path, exist_ok=True)

    # --- Chart 1: Churn Distribution ---
    # Shows how many customers churned vs stayed
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Churn', data=df, palette='Set2')
    plt.title('Churn Distribution (0=Stayed, 1=Churned)')
    plt.savefig(f'{save_path}churn_distribution.png')
    plt.close()
    print("✅ Saved: churn_distribution.png")

    # --- Chart 2: Contract Type vs Churn ---
    # Month-to-month customers churn more — important insight
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Contract', hue='Churn', data=df, palette='Set1')
    plt.title('Contract Type vs Churn')
    plt.savefig(f'{save_path}contract_vs_churn.png')
    plt.close()
    print("✅ Saved: contract_vs_churn.png")

    # --- Chart 3: Monthly Charges Distribution ---
    # Higher charges = more churn risk
    plt.figure(figsize=(8, 5))
    sns.histplot(df['MonthlyCharges'], kde=True, color='steelblue')
    plt.title('Monthly Charges Distribution')
    plt.savefig(f'{save_path}monthly_charges.png')
    plt.close()
    print("✅ Saved: monthly_charges.png")

    # --- Chart 4: Tenure vs Churn (Boxplot) ---
    # Long-tenure customers churn less
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Churn', y='tenure', data=df, palette='coolwarm')
    plt.title('Tenure vs Churn')
    plt.savefig(f'{save_path}tenure_vs_churn.png')
    plt.close()
    print("✅ Saved: tenure_vs_churn.png")

    # --- Chart 5: Correlation Heatmap ---
    # Shows which features are related to each other and to Churn
    plt.figure(figsize=(14, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f'{save_path}correlation_heatmap.png')
    plt.close()
    print("✅ Saved: correlation_heatmap.png")