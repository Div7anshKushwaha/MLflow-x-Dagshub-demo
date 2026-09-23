import json
import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt
import pandas as pd
import dagshub

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# ============================================================
# DAGSHUB + MLFLOW
# ============================================================

dagshub.init(
    repo_owner="Div7anshKushwaha",
    repo_name="MLflow-x-Dagshub-demo",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/Div7anshKushwaha/MLflow-x-Dagshub-demo.mlflow"
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
class_names = iris.target_names


# ============================================================
# 2. TRAIN-TEST SPLIT
# ============================================================

TEST_SIZE = 0.2
RANDOM_STATE = 42

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# ============================================================
# 3. CREATE TRAIN / TEST DATASETS
# ============================================================

train_df = pd.DataFrame(
    X_train,
    columns=feature_names
)

train_df["target"] = y_train


test_df = pd.DataFrame(
    X_test,
    columns=feature_names
)

test_df["target"] = y_test


# Save datasets separately
train_df.to_csv(
    "train_dataset.csv",
    index=False
)

test_df.to_csv(
    "test_dataset.csv",
    index=False
)


# ============================================================
# 4. CREATE MLFLOW DATASETS
# ============================================================

train_dataset = mlflow.data.from_pandas(
    train_df,
    source="sklearn.datasets.load_iris",
    name="iris_train_dataset"
)

test_dataset = mlflow.data.from_pandas(
    test_df,
    source="sklearn.datasets.load_iris",
    name="iris_test_dataset"
)


# ============================================================
# 5. MODEL PARAMETERS
# ============================================================

N_ESTIMATORS = 100
MAX_DEPTH = 5
MAX_FEATURES = 2


# ============================================================
# 6. MLFLOW EXPERIMENT
# ============================================================

mlflow.set_experiment(
    "Iris Random Forest Full Tracking"
)


# ============================================================
# 7. START RUN
# ============================================================

with mlflow.start_run(
    run_name="Random Forest - Full Tracking"
):

    # ========================================================
    # TAGS
    # ========================================================

    mlflow.set_tag(
        "model_type",
        "Random Forest"
    )

    mlflow.set_tag(
        "dataset",
        "Iris"
    )

    mlflow.set_tag(
        "framework",
        "scikit-learn"
    )

    mlflow.set_tag(
        "task",
        "multiclass classification"
    )

    mlflow.set_tag(
        "developer",
        "Divyansh"
    )


    # ========================================================
    # PARAMETERS
    # ========================================================

    mlflow.log_param(
        "n_estimators",
        N_ESTIMATORS
    )

    mlflow.log_param(
        "max_depth",
        MAX_DEPTH
    )

    mlflow.log_param(
        "max_features",
        MAX_FEATURES
    )

    mlflow.log_param(
        "test_size",
        TEST_SIZE
    )

    mlflow.log_param(
        "random_state",
        RANDOM_STATE
    )

    # Dataset sizes
    mlflow.log_param(
        "train_samples",
        len(train_df)
    )

    mlflow.log_param(
        "test_samples",
        len(test_df)
    )


    # ========================================================
    # DATASET TRACKING
    # ========================================================

    mlflow.log_input(
        train_dataset,
        context="training"
    )

    mlflow.log_input(
        test_dataset,
        context="testing"
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        max_features=MAX_FEATURES,
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train
    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    y_pred = model.predict(
        X_test
    )


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )


    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot()

    plt.title(
        "Iris Random Forest - Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        "confusion_matrix.png"
    )

    plt.close()


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        feature_names,
        model.feature_importances_
    )

    plt.xlabel(
        "Features"
    )

    plt.ylabel(
        "Importance"
    )

    plt.title(
        "Random Forest Feature Importance"
    )

    plt.xticks(
        rotation=30
    )

    plt.tight_layout()

    plt.savefig(
        "feature_importance.png"
    )

    plt.close()


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names
    )

    with open(
        "classification_report.txt",
        "w"
    ) as f:

        f.write(report)


    # ========================================================
    # METRICS JSON
    # ========================================================

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    with open(
        "metrics.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )


    # ========================================================
    # LOG DATASETS
    # ========================================================

    mlflow.log_artifact(
        "train_dataset.csv",
        artifact_path="dataset/train"
    )

    mlflow.log_artifact(
        "test_dataset.csv",
        artifact_path="dataset/test"
    )


    # ========================================================
    # LOG PLOTS
    # ========================================================

    mlflow.log_artifact(
        "confusion_matrix.png",
        artifact_path="plots"
    )

    mlflow.log_artifact(
        "feature_importance.png",
        artifact_path="plots"
    )


    # ========================================================
    # LOG REPORTS
    # ========================================================

    mlflow.log_artifact(
        "classification_report.txt",
        artifact_path="reports"
    )

    mlflow.log_artifact(
        "metrics.json",
        artifact_path="reports"
    )


    # ========================================================
    # LOG MODEL
    # ========================================================

    mlflow.sklearn.log_model(
        model,
        name="random_forest_model",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\nDataset Information")
    print("-------------------")

    print(
        f"Training samples : {len(train_df)}"
    )

    print(
        f"Testing samples  : {len(test_df)}"
    )

    print("\nModel Parameters")
    print("----------------")

    print(
        f"n_estimators : {N_ESTIMATORS}"
    )

    print(
        f"max_depth    : {MAX_DEPTH}"
    )

    print(
        f"max_features : {MAX_FEATURES}"
    )

    print(
        f"test_size    : {TEST_SIZE}"
    )

    print(
        f"random_state : {RANDOM_STATE}"
    )

    print("\nModel Performance")
    print("-----------------")

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        "\nMLflow Run completed successfully."
    )