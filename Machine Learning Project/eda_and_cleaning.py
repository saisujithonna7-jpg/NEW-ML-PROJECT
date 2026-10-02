"""
Chronic Kidney Disease (CKD) Dataset - Data Cleaning & EDA
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, "kidney_disease.csv")
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    print(f"Loading raw dataset from: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"Initial shape: {df.shape}")
    
    # 1. Drop 'id' column as it is not a predictor
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)
        print("Dropped 'id' column.")

    # 2. Clean corrupted numeric columns (pcv, wc, rc) that were parsed as strings
    numeric_str_cols = ["pcv", "wc", "rc"]
    for col in numeric_str_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"\t|\?", "", regex=True).str.strip(), errors="coerce")
        print(f"Converted {col} to numeric (missing now: {df[col].isnull().sum()})")

    # 3. Clean string / categorical columns with stray tabs and whitespace
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in object_cols:
        df[col] = df[col].astype(str).str.replace(r"\t", "", regex=True).str.strip().str.lower()
        df[col] = df[col].replace({"nan": np.nan, "?": np.nan, "": np.nan})

    # Standardize specific known dirty values
    if "dm" in df.columns:
        df["dm"] = df["dm"].replace({" yes": "yes", "yes": "yes", "no": "no"})
    if "cad" in df.columns:
        df["cad"] = df["cad"].replace({"yes": "yes", "no": "no"})
    if "classification" in df.columns:
        df["classification"] = df["classification"].replace({"ckd": "ckd", "notckd": "notckd"})

    print("\nCleaned unique values in categorical columns:")
    for col in df.select_dtypes(include=["object"]).columns:
        print(f"  {col}: {df[col].dropna().unique()}")

    # 4. Handle Missing Values
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols and c != "classification"]

    print(f"\nNumeric columns ({len(num_cols)}): {num_cols}")
    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

    # Impute numeric features with median
    for col in num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Impute categorical features with mode
    for col in cat_cols:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)

    # Drop any rows where target is missing (if any)
    df.dropna(subset=["classification"], inplace=True)
    
    # Map target: ckd -> 1, notckd -> 0
    df["target"] = df["classification"].map({"ckd": 1, "notckd": 0})
    print("\nTarget distribution:")
    print(df["classification"].value_counts())

    # Save cleaned dataset
    cleaned_path = os.path.join(base_dir, "cleaned_kidney_disease.csv")
    df.to_csv(cleaned_path, index=False)
    print(f"\nSaved cleaned dataset to: {cleaned_path}")

    # 5. Visualizations
    sns.set_theme(style="whitegrid", palette="muted")

    # Plot 1: Target distribution
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x="classification", hue="classification", legend=False, palette=["#e74c3c", "#2ecc71"])
    plt.title("Chronic Kidney Disease - Class Distribution", fontsize=13, fontweight="bold")
    plt.xlabel("Diagnosis", fontsize=11)
    plt.ylabel("Patient Count", fontsize=11)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    plt.tight_layout()
    target_fig_path = os.path.join(figures_dir, "target_distribution.png")
    plt.savefig(target_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {target_fig_path}")

    # Plot 2: Correlation Heatmap of Numerical Features
    plt.figure(figsize=(12, 10))
    corr_cols = num_cols + ["target"]
    corr = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'shrink': .8}, linewidths=0.5)
    plt.title("Correlation Heatmap (Numeric Features and Target)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    corr_fig_path = os.path.join(figures_dir, "correlation_heatmap.png")
    plt.savefig(corr_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {corr_fig_path}")

    # Plot 3: Boxplots of Key Numerical Indicators by Diagnosis
    key_features = ["hemo", "sc", "bgr", "pcv", "al", "bu"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for idx, feature in enumerate(key_features):
        sns.boxplot(data=df, x="classification", y=feature, ax=axes[idx], palette=["#e74c3c", "#2ecc71"], hue="classification", legend=False)
        axes[idx].set_title(f"Distribution of {feature.upper()}", fontweight="bold")
        axes[idx].set_xlabel("")
    plt.suptitle("Key Biomarkers vs. Kidney Disease Status", fontsize=15, fontweight="bold")
    plt.tight_layout()
    key_fig_path = os.path.join(figures_dir, "key_features_distribution.png")
    plt.savefig(key_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {key_fig_path}")
    print("\nEDA and Cleaning finished successfully!")

if __name__ == "__main__":
    main()
