import os
import warnings
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def get_model_configs() -> dict:
    """
    Defines the 5 ML algorithms and their hyperparameter search grids.
    """
    return {
        "logistic_regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0]
            }
        },
        "svm": {
            "model": SVC(probability=True, random_state=42),
            "params": {
                "C": [0.5, 1.0, 5.0],
                "kernel": ["linear", "rbf"]
            }
        },
        "decision_tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                "max_depth": [3, 5, 8, 12],
                "min_samples_split": [2, 10, 20]
            }
        },
        "random_forest": {
            "model": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [6, 12, None],
                "min_samples_split": [2, 10]
            }
        },
        "xgboost": {
            "model": XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            ),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1]
            }
        }
    }


def train_models(
    train_path: str = "data/processed/train_processed.csv",
    model_dir: str = "models",
    reports_dir: str = "reports"
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load preprocessed, SMOTE-balanced training data
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(columns=["project_failed"])
    y_train = train_df["project_failed"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"}

    configs = get_model_configs()
    results = []

    print("Starting 5-Fold Cross-Validation & Hyperparameter Tuning...\n")

    for name, cfg in configs.items():
        print(f"Training {name}...")
        grid = GridSearchCV(
            estimator=cfg["model"],
            param_grid=cfg["params"],
            cv=cv,
            scoring=scoring,
            refit="roc_auc",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        best_idx = grid.best_index_
        best_model = grid.best_estimator_

        # Save each tuned model
        model_path = os.path.join(model_dir, f"{name}.joblib")
        joblib.dump(best_model, model_path)

        results.append({
            "model": name,
            "cv_roc_auc": round(grid.cv_results_["mean_test_roc_auc"][best_idx], 4),
            "cv_f1": round(grid.cv_results_["mean_test_f1"][best_idx], 4),
            "cv_accuracy": round(grid.cv_results_["mean_test_accuracy"][best_idx], 4),
            "best_params": str(grid.best_params_)
        })

    # 2. Rank models by CV ROC-AUC and save summary
    summary_df = pd.DataFrame(results).sort_values(by="cv_roc_auc", ascending=False)
    summary_path = os.path.join(reports_dir, "cv_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n=== 5-Fold Cross-Validation Summary ===")
    print(summary_df[["model", "cv_roc_auc", "cv_f1", "cv_accuracy"]].to_string(index=False))
    print(f"\nAll 5 tuned models saved to '{model_dir}/' and CV summary saved to '{summary_path}'")


if __name__ == "__main__":
    train_models()