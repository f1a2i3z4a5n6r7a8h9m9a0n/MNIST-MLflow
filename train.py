"""Assignment 3 Part C: four MNIST experiments with MLflow and DagsHub.

Run `python train.py --help`. Defaults always mean all four 50-epoch runs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ASYNC_LOGGING", "false")
os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parent / ".cache" / "keras"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow
    import mlflow.tensorflow
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from dotenv import load_dotenv
except ImportError as exc:
    raise SystemExit("Missing dependency. Use Python 3.11 and run: python -m pip install -r requirements.txt") from exc

ROOT = Path(__file__).resolve().parent
SEED = 42
EXPERIMENT = "Assignment3-PartC-MNIST"
CONFIGS = [
    {"name": "baseline_128_adam_b64", "neurons": [128], "optimizer": "Adam", "batch_size": 64, "epochs": 50},
    {"name": "experiment2_256_adam_b64", "neurons": [256], "optimizer": "Adam", "batch_size": 64, "epochs": 50},
    {"name": "experiment3_128_64_adam_b64", "neurons": [128, 64], "optimizer": "Adam", "batch_size": 64, "epochs": 50},
    {"name": "experiment4_128_sgd_b32", "neurons": [128], "optimizer": "SGD", "batch_size": 32, "epochs": 50},
]


def configure_runtime():
    """Seeds improve repeatability; exact results still depend on hardware/packages."""
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()


def configure_tracking(mode="dagshub", smoke=False):
    load_dotenv(ROOT / ".env", override=False)
    if mlflow.active_run() is not None:
        raise RuntimeError("Finish the active MLflow run before starting this project.")
    if mode == "dagshub":
        owner = os.getenv("DAGSHUB_REPO_OWNER", "").strip()
        repo = os.getenv("DAGSHUB_REPO_NAME", "").strip()
        if not owner or not repo:
            raise ValueError("Set DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME in .env. See README.md, or use --tracking local for local verification.")
        token = os.getenv("DAGSHUB_USER_TOKEN") or os.getenv("MLFLOW_TRACKING_PASSWORD")
        if not token:
            raise ValueError("Set DAGSHUB_USER_TOKEN in your private .env. Never put the token in code or Git.")
        # The token must be available to both the DagsHub client and MLflow.
        os.environ["DAGSHUB_USER_TOKEN"] = token
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME") or owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        try:
            import dagshub
        except ImportError as exc:
            raise RuntimeError("DagsHub client missing: python -m pip install -r requirements.txt") from exc
        dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)
        expected_uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
        configured_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
        if configured_uri and configured_uri.rstrip("/") != expected_uri:
            raise ValueError("MLFLOW_TRACKING_URI must match the configured DagsHub repository. Copy it from the repository Remote panel.")
        mlflow.set_tracking_uri(expected_uri)
    elif mode == "local":
        (ROOT / ".tracking").mkdir(exist_ok=True)
        mlflow.set_tracking_uri("sqlite:///" + (ROOT / ".tracking" / "mlflow.db").as_posix())
    else:
        raise ValueError("Tracking must be 'local' or 'dagshub'.")
    name = EXPERIMENT + ("-SMOKE-ONLY" if smoke else "")
    if mode == "local" and mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ROOT / ".tracking" / "artifacts").as_uri())
    mlflow.set_experiment(name)
    # Keep epoch metrics and architecture autologging. Store .keras files with
    # log_artifact below so DagsHub does not need MLflow 3 logged-model endpoints.
    mlflow.tensorflow.autolog(log_models=False, log_datasets=False, checkpoint=False,
                              log_every_epoch=True, exclusive=False)
    print(f"Tracking mode: {mode}; experiment: {name}")
    return {"mode": mode, "experiment": name, "tracking_uri": mlflow.get_tracking_uri()}


def load_data(smoke=False):
    """Split only original training data; test images never enter training/selection."""
    try:
        (images, labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    except Exception as exc:
        raise RuntimeError("MNIST could not load. Check internet access or place mnist.npz in .cache/keras/datasets/ (or KERAS_HOME/datasets if overridden).") from exc
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(images))
    val_indices, train_indices = indices[:6000], indices[6000:]
    if smoke:
        train_indices, val_indices = train_indices[:1024], val_indices[:256]
    data = {
        "x_train": images[train_indices].astype("float32") / 255.0,
        "y_train": labels[train_indices],
        "x_val": images[val_indices].astype("float32") / 255.0,
        "y_val": labels[val_indices],
        "x_test": test_images.astype("float32") / 255.0,
        "y_test": test_labels,
        "train_indices": train_indices, "val_indices": val_indices,
    }
    for key in ["x_train", "y_train", "x_val", "y_val", "x_test", "y_test"]:
        print(f"{key}: {data[key].shape}, {data[key].dtype}")
    return data


def build_model(config):
    model = tf.keras.Sequential(name=config["name"])
    model.add(tf.keras.layers.Input(shape=(28, 28)))
    model.add(tf.keras.layers.Flatten(name="flatten"))
    for i, units in enumerate(config["neurons"], start=1):
        model.add(tf.keras.layers.Dense(units, activation="relu", name=f"hidden_{i}"))
    model.add(tf.keras.layers.Dense(10, activation="softmax", name="digits"))
    optimizer = tf.keras.optimizers.Adam() if config["optimizer"] == "Adam" else tf.keras.optimizers.SGD()
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def make_dataset(images, labels, batch_size, training=False):
    """Explicit batching also keeps CPU thread use bounded on student laptops."""
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        ds = ds.shuffle(len(images), seed=SEED, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 2
    return ds.batch(batch_size).with_options(options).prefetch(1)


def create_output_dir(smoke=False):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = ROOT / "outputs" / (("smoke_" if smoke else "full_") + stamp)
    path.mkdir(parents=True)
    return path


def run_experiment(config, data, output_dir, smoke=False):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = build_model(config)
    epochs = 1 if smoke else config["epochs"]
    run_dir = Path(output_dir) / config["name"]
    run_dir.mkdir()
    with mlflow.start_run(run_name=("SMOKE_" if smoke else "") + config["name"]) as run:
        mlflow.set_tags({"assignment_part": "C", "smoke_test": str(smoke).lower(),
                         "dataset": "MNIST", "selection_rule": "final validation accuracy, then final validation loss"})
        params = {"hidden_layers": len(config["neurons"]), "neurons_per_layer": json.dumps(config["neurons"]),
                  "optimizer": config["optimizer"], "configured_batch_size": config["batch_size"],
                  "configured_epochs": config["epochs"], "actual_epochs": epochs, "seed": SEED,
                  "train_size": len(data["x_train"]), "validation_size": len(data["x_val"]),
                  "loss_function": "sparse_categorical_crossentropy",
                  "learning_rate": float(tf.keras.backend.get_value(model.optimizer.learning_rate)),
                  "split_sha256": hashlib.sha256(data["val_indices"].tobytes()).hexdigest()}
        params.update({f"hidden_layer_{i}_neurons": units for i, units in enumerate(config["neurons"], 1)})
        # Explicit names avoid conflicts with autolog's fit arguments for tf.data.
        mlflow.log_params(params)
        summary_lines = []
        model.summary(print_fn=lambda text, **kwargs: summary_lines.append(text))
        (run_dir / "model_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
        history = model.fit(
            make_dataset(data["x_train"], data["y_train"], config["batch_size"], training=True),
            validation_data=make_dataset(data["x_val"], data["y_val"], config["batch_size"]),
            epochs=epochs, verbose=2, shuffle=False,
        ).history
        pd.DataFrame(history, index=pd.RangeIndex(1, epochs + 1, name="epoch")).to_csv(run_dir / "history.csv")
        # Explicitly preserve all four epoch series even across autolog versions.
        for step in range(epochs):
            mlflow.log_metrics({f"curve_{key}": float(history[key][step])
                                for key in ["accuracy", "val_accuracy", "loss", "val_loss"]}, step=step)
        final_metrics = {"training_accuracy": history["accuracy"][-1], "validation_accuracy": history["val_accuracy"][-1],
                         "training_loss": history["loss"][-1], "validation_loss": history["val_loss"][-1]}
        mlflow.log_metrics(final_metrics)
        model.save(run_dir / "model.keras")
        mlflow.log_artifacts(str(run_dir), artifact_path="training")
        # Verify the values arrived before declaring success (also works remotely).
        recorded = mlflow.MlflowClient().get_run(run.info.run_id)
        for key, value in final_metrics.items():
            if not np.isclose(recorded.data.metrics.get(key, np.nan), value):
                raise RuntimeError(f"Tracking verification failed for {key}.")
        if "accuracy" not in recorded.data.metrics or "val_accuracy" not in recorded.data.metrics:
            raise RuntimeError("TensorFlow autologging did not record epoch accuracy metrics.")
        return {"name": config["name"], "run_id": run.info.run_id, **final_metrics,
                "hidden_layers": len(config["neurons"]), "neurons_per_layer": json.dumps(config["neurons"]),
                "optimizer": config["optimizer"], "batch_size": config["batch_size"], "epochs": epochs,
                "configured_epochs": config["epochs"], "smoke_test": smoke}


def plot_comparison(results, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for result in results:
        history = pd.read_csv(Path(output_dir) / result["name"] / "history.csv")
        for ax, key, title in zip(axes.flat, ["accuracy", "val_accuracy", "loss", "val_loss"],
                                 ["Training accuracy", "Validation accuracy", "Training loss", "Validation loss"]):
            ax.plot(history["epoch"], history[key], label=result["name"])
            ax.set(xlabel="Epoch", ylabel=title, title=title)
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "comparison.png", dpi=150)
    plt.close(fig)


def finalize_results(results, data, output_dir, tracking, smoke=False):
    if len(results) != 4 or {r["name"] for r in results} != {c["name"] for c in CONFIGS}:
        raise ValueError("Complete all four different experiments before comparison.")
    if not smoke and any(r["epochs"] != 50 or r["smoke_test"] for r in results):
        raise ValueError("Full comparison requires four complete 50-epoch runs.")
    frame = pd.DataFrame(results).sort_values(["validation_accuracy", "validation_loss", "name"],
                                            ascending=[False, True, True]).reset_index(drop=True)
    frame.to_csv(Path(output_dir) / "comparison.csv", index=False)
    plot_comparison(results, output_dir)
    best = frame.iloc[0].to_dict()
    gap = best["training_accuracy"] - best["validation_accuracy"]
    history = pd.read_csv(Path(output_dir) / best["name"] / "history.csv")
    best_loss_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
    loss_rise = float(history["val_loss"].iloc[-1] - history["val_loss"].min())
    train_loss_change = float(history["loss"].iloc[-1] - history.loc[history["val_loss"].idxmin(), "loss"])
    runner_up = frame.iloc[1]
    report = ["# " + ("SMOKE TEST ONLY" if smoke else "MNIST experiment results"), "",
              "Tracking mode: " + tracking["mode"] + ". Local runs are not DagsHub evidence.", "",
              f"Selected final-epoch model: **{best['name']}**, using validation accuracy first and lower validation loss to break ties.",
              f"Validation accuracy: {best['validation_accuracy']:.6f}; validation loss: {best['validation_loss']:.6f}.",
              f"The next-ranked model, {runner_up['name']}, has validation accuracy {runner_up['validation_accuracy']:.6f} "
              f"and validation loss {runner_up['validation_loss']:.6f}. These observed validation values determine the ranking, "
              "rather than model size or training accuracy.",
              f"Training minus validation accuracy: {gap:.6f}. Minimum validation loss occurred at epoch {best_loss_epoch}.", "",
              f"Final validation loss is {loss_rise:.6f} above its minimum. Since that epoch, training loss changed by {train_loss_change:.6f}. "
              + ("Validation loss increased while training loss decreased, suggesting overfitting after that epoch."
                 if loss_rise > 0 and train_loss_change < 0 else
                 "This particular loss comparison does not show simultaneous increasing validation loss and decreasing training loss; inspect all curves and the accuracy gap as well."),
              "A training/validation gap, especially when training loss falls while validation loss rises, is evidence of possible overfitting. "
              "The curves describe this sample; one seed does not establish that an architecture is always superior. "
              "Training metrics are the last fit epoch's batch aggregates; validation metrics evaluate the final epoch weights.", "",
              "All models use final epoch weights; there is no early stopping or best-epoch restoration. "
              "The test dataset was excluded from model selection."]
    test_result = None
    if not smoke:
        # This is the ONLY test evaluation, after ranking has been fixed.
        model = tf.keras.models.load_model(Path(output_dir) / best["name"] / "model.keras")
        loss, accuracy = model.evaluate(make_dataset(data["x_test"], data["y_test"], 64), verbose=0)
        test_result = {"selected_model": best["name"], "test_loss": float(loss), "test_accuracy": float(accuracy),
                       "test_size": len(data["y_test"]), "run_id": best["run_id"]}
        (Path(output_dir) / "test_results.json").write_text(json.dumps(test_result, indent=2), encoding="utf-8")
        report.extend(["", f"Untouched test set: accuracy {accuracy:.6f}, loss {loss:.6f} ({len(data['y_test'])} images)."])
    else:
        report.extend(["", "Smoke runs use 1,024 training images, 256 validation images and one epoch. Test evaluation is skipped; these are not assignment results."])
    versions = {p: importlib.metadata.version(p) for p in ["tensorflow", "keras", "mlflow", "numpy", "pandas", "matplotlib"]}
    manifest = {"tracking": tracking, "smoke_test": smoke, "seed": SEED, "python": platform.python_version(),
                "versions": versions, "configurations": CONFIGS, "selected_model": best["name"],
                "runs": results, "selection": "final validation accuracy descending; final validation loss ascending",
                "test_result": test_result}
    (Path(output_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (Path(output_dir) / "analysis.md").write_text("\n\n".join(report), encoding="utf-8")
    # Reopen existing runs; never create a fifth training run for the comparison.
    for result in results:
        with mlflow.start_run(run_id=result["run_id"]):
            mlflow.set_tag("selected_model", str(result["name"] == best["name"]).lower())
            for name in ["comparison.csv", "comparison.png", "analysis.md", "manifest.json"]:
                mlflow.log_artifact(str(Path(output_dir) / name), artifact_path="comparison")
            if test_result and result["name"] == best["name"]:
                mlflow.log_metrics({k: test_result[k] for k in ["test_accuracy", "test_loss"]})
                mlflow.log_artifact(str(Path(output_dir) / "test_results.json"), artifact_path="comparison")
    print(frame.to_string(index=False))
    print(f"Outputs: {output_dir}")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", choices=["dagshub", "local"], default="dagshub")
    parser.add_argument("--smoke", action="store_true", help="Separate 1-epoch diagnostic experiment; never assignment results")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("This project targets Python 3.11. Create a Python 3.11 environment first.")
    configure_runtime()
    tracking = configure_tracking(args.tracking, args.smoke)
    data = load_data(args.smoke)
    output_dir = create_output_dir(args.smoke)
    results = [run_experiment(config, data, output_dir, args.smoke) for config in CONFIGS]
    finalize_results(results, data, output_dir, tracking, args.smoke)


if __name__ == "__main__":
    main()
