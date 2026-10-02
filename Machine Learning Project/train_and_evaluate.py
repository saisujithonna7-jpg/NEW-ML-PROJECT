"""
Chronic Kidney Disease (CKD) - Model Training, Evaluation & Comparison
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cleaned_path = os.path.join(base_dir, "cleaned_kidney_disease.csv")
    figures_dir = os.path.join(base_dir, "figures")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading cleaned data from: {cleaned_path}")
    df = pd.read_csv(cleaned_path)

    # Separate features and target
    y = df["target"]
    X = df.drop(columns=["target", "classification"])

    # Categorical columns to encode
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"Categorical features to encode: {cat_cols}")

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)
    feature_names = list(X_encoded.columns)
    print(f"Total features after encoding: {len(feature_names)}")

    # Train / Test split (80% train, 20% test, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save feature names and scaler
    with open(os.path.join(models_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))

    # Define Candidate Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
    }

    results = []
    trained_models = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n--- Training and Evaluating Models ---")
    for name, model in models.items():
        # Cross-validation on training data
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="f1")
        cv_f1_mean = cv_scores.mean()
        cv_f1_std = cv_scores.std()

        # Fit on full training set
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model

        # Test set evaluation
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            "Model": name,
            "CV F1 (Mean)": round(cv_f1_mean, 4),
            "CV F1 (Std)": round(cv_f1_std, 4),
            "Test Accuracy": round(acc, 4),
            "Test Precision": round(prec, 4),
            "Test Recall": round(rec, 4),
            "Test F1-Score": round(f1, 4),
            "Test ROC-AUC": round(auc, 4)
        })
        print(f"[{name}] Test Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

    results_df = pd.DataFrame(results)
    csv_results_path = os.path.join(base_dir, "model_comparison_results.csv")
    results_df.to_csv(csv_results_path, index=False)
    print(f"\nSaved model results to: {csv_results_path}")
    print("\n--- Model Performance Comparison Table ---")
    print(results_df.to_string(index=False))

    # Identify Best Model by Test F1-Score (and Accuracy)
    best_row = results_df.sort_values(by=["Test F1-Score", "Test ROC-AUC"], ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]
    print(f"\nBest Performing Model: {best_model_name}")

    # Save Best Model
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model, best_model_path)
    print(f"Saved best model to: {best_model_path}")

    # Visualizations
    sns.set_theme(style="whitegrid")

    # Plot 1: Model Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    melted_df = results_df.melt(id_vars=["Model"], value_vars=["Test Accuracy", "Test F1-Score", "Test ROC-AUC"],
                                var_name="Metric", value_name="Score")
    ax = sns.barplot(data=melted_df, x="Model", y="Score", hue="Metric", palette="deep")
    plt.title("Model Comparison on Test Set", fontsize=14, fontweight="bold")
    plt.ylim(0.8, 1.05)
    plt.xticks(rotation=15)
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height + 0.005),
                        ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    comparison_fig_path = os.path.join(figures_dir, "model_comparison.png")
    plt.savefig(comparison_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {comparison_fig_path}")

    # Plot 2: Confusion Matrices Subplots for All Models
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for idx, (name, model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx], cbar=False,
                    xticklabels=["Not CKD (0)", "CKD (1)"],
                    yticklabels=["Not CKD (0)", "CKD (1)"])
        axes[idx].set_title(f"{name}", fontweight="bold")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")
    # Hide 6th empty subplot
    axes[5].axis("off")
    plt.suptitle("Confusion Matrices Across Models (Test Set)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    cm_fig_path = os.path.join(figures_dir, "confusion_matrices.png")
    plt.savefig(cm_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_fig_path}")

    # Plot 3: Feature Importance (from Random Forest)
    rf_model = trained_models["Random Forest"]
    importances = rf_model.feature_importances_
    feat_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False).head(12)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp_df, x="Importance", y="Feature", palette="viridis")
    plt.title("Top 12 Most Influential Clinical Features (Random Forest)", fontsize=14, fontweight="bold")
    plt.xlabel("Gini Feature Importance", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.tight_layout()
    feat_fig_path = os.path.join(figures_dir, "feature_importance.png")
    plt.savefig(feat_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {feat_fig_path}")
    print("\nModel Training and Evaluation finished successfully!")

if __name__ == "__main__":
    main()
