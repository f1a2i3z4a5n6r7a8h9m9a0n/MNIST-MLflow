"""Check real data, configurations and the integrity of a completed execution."""
import argparse
import json
from pathlib import Path

import train
import numpy as np
import mlflow
import tensorflow as tf


def verify_data_and_models():
    train.configure_runtime()
    data = train.load_data()
    assert data["x_train"].shape == (54000, 28, 28)
    assert data["x_val"].shape == (6000, 28, 28)
    assert data["x_test"].shape == (10000, 28, 28)
    assert not np.intersect1d(data["train_indices"], data["val_indices"]).size
    assert len(np.unique(np.concatenate([data["train_indices"], data["val_indices"]]))) == 60000
    for key in ["x_train", "x_val", "x_test"]:
        assert data[key].dtype == np.float32
        assert 0 <= data[key].min() <= data[key].max() <= 1
    expected = [([128], "Adam", 64), ([256], "Adam", 64), ([128, 64], "Adam", 64), ([128], "SGD", 32)]
    for config, (neurons, optimizer, batch_size) in zip(train.CONFIGS, expected):
        assert (config["neurons"], config["optimizer"], config["batch_size"], config["epochs"]) == (neurons, optimizer, batch_size, 50)
        model = train.build_model(config)
        assert isinstance(model.layers[0], tf.keras.layers.Flatten)
        assert [layer.units for layer in model.layers[1:]] == neurons + [10]
        assert all(layer.activation.__name__ == "relu" for layer in model.layers[1:-1])
        assert model.layers[-1].activation.__name__ == "softmax"
        assert type(model.optimizer).__name__ == optimizer
        probs = model(data["x_train"][:2], training=False).numpy()
        assert probs.shape == (2, 10)
        np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6)
    print("PASS: full MNIST preprocessing, split isolation and all four architectures/configurations")


def verify_execution(path):
    import pandas as pd
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(manifest["tracking"]["tracking_uri"])
    client = mlflow.MlflowClient()
    assert len(manifest["runs"]) == 4
    for row in manifest["runs"]:
        run = client.get_run(row["run_id"])
        assert run.info.status == "FINISHED"
        for name in ["training_accuracy", "validation_accuracy", "training_loss", "validation_loss"]:
            assert np.isclose(run.data.metrics[name], row[name])
        for name in ["accuracy", "val_accuracy", "loss", "val_loss"]:
            assert len(client.get_metric_history(row["run_id"], name)) == row["epochs"]
        assert len(pd.read_csv(path / row["name"] / "history.csv")) == row["epochs"]
        artifacts = {item.path for item in client.list_artifacts(row["run_id"], "training")}
        assert "training/model.keras" in artifacts
        assert "training/history.csv" in artifacts
    table = pd.read_csv(path / "comparison.csv")
    assert table.iloc[0]["name"] == manifest["selected_model"]
    if manifest["smoke_test"]:
        assert manifest["test_result"] is None and not (path / "test_results.json").exists()
    else:
        assert all(row["epochs"] == 50 for row in manifest["runs"])
        test = manifest["test_result"]
        assert test["test_size"] == 10000
        assert sum("test_accuracy" in client.get_run(row["run_id"]).data.metrics for row in manifest["runs"]) == 1
    assert mlflow.active_run() is None
    print("PASS: four finished runs, autologged epoch histories, explicit metrics, saved artifacts and selection/test protocol")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path)
    args = parser.parse_args()
    verify_data_and_models()
    if args.outputs:
        verify_execution(args.outputs)
