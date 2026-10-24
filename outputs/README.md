# Actual experiment outputs

Each execution creates a separate `full_<UTC timestamp>` or `smoke_<UTC timestamp>` directory.
Full runs contain real comparison.csv, comparison.png, analysis.md, manifest.json,
test_results.json, and a model directory per run with history.csv, model_summary.txt,
and model.keras. Models are excluded from Git to keep the repository small; MLflow
stores them as artifacts. Smoke results are diagnostics and cannot satisfy Part C.
