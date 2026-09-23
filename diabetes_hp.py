import mlflow
import mlflow.sklearn

import pandas as pd

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import dagshub


# ============================================================
# 1. DAGSHUB + MLFLOW
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
# 2. LOAD DATA
# ============================================================

URL = (
    "https://raw.githubusercontent.com/"
    "npradaschnor/Pima-Indians-Diabetes-Dataset/"
    "refs/heads/master/diabetes.csv"
)

df = pd.read_csv(URL)

print("Dataset shape:", df.shape)


# ============================================================
# 3. FEATURES / TARGET
# ============================================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. HYPERPARAMETER GRID
# ============================================================

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5],
    "min_samples_split": [2, 5]
}


# ============================================================
# 6. MLFLOW EXPERIMENT
# ============================================================

mlflow.set_experiment(
    "Pima Diabetes - Hyperparameter Tuning"
)


# ============================================================
# 7. PARENT RUN
# ============================================================

with mlflow.start_run(
    run_name="Random Forest - Hyperparameter Tuning"
) as parent_run:

    parent_run_id = parent_run.info.run_id

    mlflow.set_tag(
        "run_type",
        "parent"
    )

    mlflow.set_tag(
        "model",
        "Random Forest"
    )

    mlflow.set_tag(
        "dataset",
        "Pima Indians Diabetes"
    )

    mlflow.log_param(
        "test_size",
        0.2
    )

    mlflow.log_param(
        "random_state",
        42
    )

    mlflow.log_param(
        "total_trials",
        len(list(ParameterGrid(param_grid)))
    )


    # ========================================================
    # 8. HYPERPARAMETER SEARCH
    # ========================================================

    best_score = -1
    best_params = None
    best_model = None


    for trial_number, params in enumerate(
        ParameterGrid(param_grid),
        start=1
    ):

        # ====================================================
        # CHILD RUN
        # ====================================================

        with mlflow.start_run(
            run_name=f"Trial {trial_number}",
            nested=True
        ) as child_run:

            # -----------------------------------------------
            # Parameters
            # -----------------------------------------------

            mlflow.log_param(
                "n_estimators",
                params["n_estimators"]
            )

            mlflow.log_param(
                "max_depth",
                params["max_depth"]
            )

            mlflow.log_param(
                "min_samples_split",
                params["min_samples_split"]
            )

            mlflow.log_param(
                "trial_number",
                trial_number
            )


            # -----------------------------------------------
            # Model
            # -----------------------------------------------

            model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                random_state=42
            )


            # -----------------------------------------------
            # Train
            # -----------------------------------------------

            model.fit(
                X_train,
                y_train
            )


            # -----------------------------------------------
            # Validation / Test prediction
            # -----------------------------------------------

            y_pred = model.predict(
                X_test
            )


            # -----------------------------------------------
            # Metrics
            # -----------------------------------------------

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred
            )

            recall = recall_score(
                y_test,
                y_pred
            )

            f1 = f1_score(
                y_test,
                y_pred
            )


            # -----------------------------------------------
            # Log metrics
            # -----------------------------------------------

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


            # -----------------------------------------------
            # Check best model
            # -----------------------------------------------

            if accuracy > best_score:

                best_score = accuracy
                best_params = params
                best_model = model

                mlflow.set_tag(
                    "best_trial",
                    "true"
                )

            else:

                mlflow.set_tag(
                    "best_trial",
                    "false"
                )


            print(
                f"Trial {trial_number}: "
                f"{params} → Accuracy = {accuracy:.4f}"
            )


    # ========================================================
    # 9. LOG BEST RESULT TO PARENT RUN
    # ========================================================

    mlflow.log_param(
        "best_n_estimators",
        best_params["n_estimators"]
    )

    mlflow.log_param(
        "best_max_depth",
        best_params["max_depth"]
    )

    mlflow.log_param(
        "best_min_samples_split",
        best_params["min_samples_split"]
    )

    mlflow.log_metric(
        "best_accuracy",
        best_score
    )


    # ========================================================
    # 10. SAVE BEST MODEL
    # ========================================================

    mlflow.sklearn.log_model(
        best_model,
        name="best_random_forest_model",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )


    # ========================================================
    # 11. PARENT RUN TAGS
    # ========================================================

    mlflow.set_tag(
        "best_model",
        "Random Forest"
    )

    mlflow.set_tag(
        "status",
        "completed"
    )


    # ========================================================
    # 12. PRINT BEST RESULT
    # ========================================================

    print("\n" + "=" * 50)
    print("BEST RESULT")
    print("=" * 50)

    print(
        "Best Parameters:",
        best_params
    )

    print(
        f"Best Accuracy: {best_score:.4f}"
    )

    print(
        "\nParent Run ID:",
        parent_run_id
    )

    print(
        "\nMLflow Hyperparameter Tuning Completed."
    )