import mlflow
import mlflow.sklearn
import dagshub

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


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
# 2. ENABLE AUTO LOGGING
# ============================================================

mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    log_datasets=True
)


# ============================================================
# 3. LOAD DATASET
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target


# ============================================================
# 4. TRAIN-TEST SPLIT
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
# 5. MLFLOW EXPERIMENT
# ============================================================

mlflow.set_experiment(
    "Iris Random Forest Auto Logging"
)


# ============================================================
# 6. START RUN
# ============================================================

with mlflow.start_run(
    run_name="Random Forest - Auto Logging"
):

    # Optional custom tags
    mlflow.set_tag(
        "model_type",
        "Random Forest"
    )

    mlflow.set_tag(
        "dataset",
        "Iris"
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
    # 7. MODEL
    # ========================================================

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        max_features=2,
        random_state=42
    )


    # ========================================================
    # 8. TRAIN
    # ========================================================

    model.fit(
        X_train,
        y_train
    )


    # ========================================================
    # 9. PREDICTION
    # ========================================================

    y_pred = model.predict(
        X_test
    )


    print("\nPredictions completed.")

    print(
        f"Training samples : {len(X_train)}"
    )

    print(
        f"Testing samples  : {len(X_test)}"
    )

    print(
        "\nMLflow Auto Logging completed."
    )