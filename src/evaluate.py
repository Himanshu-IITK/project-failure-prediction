import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
from sklearn.inspection import permutation_importance


MODEL_NAMES = [
    "logistic_regression",
    "svm",
    "decision_tree",
    "random_forest",
    "xgboost"
]


def evaluate_models(
    test_path: str = "data/processed/test_processed.csv",
    model_dir: str = "models",
    reports_dir: str = "reports",
    figures_dir: str = "reports/figures"
) -> None:
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Load untouched test dataset
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=["project_failed"])
    y_test = test_df["project_failed"]

    results = []
    roc_data = {}
    loaded_models = {}

    # 2. Evaluate all 5 models on the test set
    for name in MODEL_NAMES:
        model_path = os.path.join(model_dir, f"{name}.joblib")
        model = joblib.load(model_path)
        loaded_models[name] = model

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        roc_data[name] = (fpr, tpr, auc_score)

        results.append({
            "model": name,
            "test_roc_auc": round(auc_score, 4),
            "test_f1": round(f1_score(y_test, y_pred), 4),
            "test_recall": round(recall_score(y_test, y_pred), 4),
            "test_precision": round(precision_score(y_test, y_pred), 4),
            "test_accuracy": round(accuracy_score(y_test, y_pred), 4)
        })

    # 3. Rank models and save the winning model
    eval_df = pd.DataFrame(results).sort_values(by="test_roc_auc", ascending=False)
    eval_path = os.path.join(reports_dir, "test_evaluation.csv")
    eval_df.to_csv(eval_path, index=False)

    best_model_name = eval_df.iloc[0]["model"]
    best_model = loaded_models[best_model_name]
    best_model_path = os.path.join(model_dir, "best_model.joblib")
    joblib.dump(best_model, best_model_path)

    print("=== Unbiased Test Set Evaluation Summary ===")
    print(eval_df.to_string(index=False))
    print(f"\nBest Performing Model: {best_model_name} (saved to '{best_model_path}')")

    # 4. Plot 1: Overlaid ROC Curves for All 5 Models
    plt.figure(figsize=(8, 6))
    for name in eval_df["model"]:
        fpr, tpr, auc_val = roc_data[name]
        label = f"{name.replace('_', ' ').title()} (AUC = {auc_val:.3f})"
        plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Baseline (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve Comparison on Unbiased Test Set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "05_roc_curves.png"), dpi=150)
    plt.close()

    # 5. Plot 2: Confusion Matrix for the Best Model
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Success (0)", "Predicted Failure (1)"],
        yticklabels=["Actual Success (0)", "Actual Failure (1)"]
    )
    plt.title(f"Confusion Matrix — {best_model_name.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "06_confusion_matrix_best_model.png"), dpi=150)
    plt.close()

    # 6. Plot 3: Top 10 Feature Importances for the Best Model
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        imp_label = "Feature Importance Score"
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])
        imp_label = "Absolute Coefficient Magnitude"
    else:
        perm = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc")
        importances = perm.importances_mean
        imp_label = "Permutation Importance (ROC-AUC Drop)"

    feat_imp = pd.Series(importances, index=X_test.columns).sort_values(ascending=False).head(10)

    plt.figure(figsize=(9, 5.5))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, hue=feat_imp.index, legend=False, palette="magma")
    plt.title(f"Top 10 Predictive Features — {best_model_name.replace('_', ' ').title()}")
    plt.xlabel(imp_label)
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "07_feature_importance.png"), dpi=150)
    plt.close()

    print(f"Evaluation figures saved successfully to '{figures_dir}/'")


if __name__ == "__main__":
    evaluate_models()