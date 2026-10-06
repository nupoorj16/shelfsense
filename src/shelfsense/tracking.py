"""MLflow experiment tracking: one run per training, so results are comparable and reproducible."""

import mlflow

from shelfsense.config import ARTIFACTS_DIR, SEED

# SQLite file store (the plain-folder store is deprecated). View with: uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "shelfsense-ranker"


def log_run(
    ranker, params: dict, results: dict[str, dict[str, float]], data_info: dict
) -> str:
    """Log params, metrics for every model, feature importances and artifacts. Returns the run id."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.log_params({**params, "seed": SEED, **data_info})
        mlflow.log_metric("best_iteration", ranker.best_iteration)
        for model, scores in results.items():
            # MLflow metric names can't contain "@".
            mlflow.log_metrics(
                {
                    f"{model}_{name.replace('@', '_at_')}": v
                    for name, v in scores.items()
                }
            )
        importance = dict(
            zip(ranker.feature_name(), ranker.feature_importance("gain"), strict=True)
        )
        mlflow.log_dict(
            dict(sorted(importance.items(), key=lambda kv: -kv[1])),
            "feature_importance_gain.json",
        )
        mlflow.log_artifact(str(ARTIFACTS_DIR / "ranker.txt"))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "results.json"))
        return run.info.run_id
