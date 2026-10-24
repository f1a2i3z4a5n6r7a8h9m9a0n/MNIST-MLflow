# Assignment 3 Part C — Tracking MNIST experiments

This Python 3.11 project trains and compares four TensorFlow/Keras digit classifiers,
records experiments with MLflow, and supports DagsHub remote tracking. It implements
Part C only. No URLs, screenshots or remote results are prefilled.

## Files

- `Assignment3_PartC.ipynb`: educational notebook; run cells from top to bottom.
- `train.py`: complete standalone training and comparison implementation.
- `requirements.txt`, `.env.example`, `.gitignore`: setup and credential hygiene.
- `verify_project.py`: checks preprocessing, architectures, metrics and artifacts.
- `verify_notebook.py`: executes a diagnostic notebook copy without altering full-run defaults.
- `outputs/`: timestamped actual results; `smoke_` directories are diagnostics only.
- `SUBMISSION_CHECKLIST.md`: final evidence and submission steps.
- `VERIFICATION.md`: what was actually executed in the development environment.

## Install on Windows with Python 3.11

From PowerShell in this project directory (confirm `python --version` is 3.11):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ipykernel install --user --name assignment3 --display-name "Python 3.11 (Assignment 3)"
```

Activation is optional because the commands use the environment's interpreter
explicitly. On macOS/Linux use `python3.11` and `.venv/bin/python`. Native Windows
uses CPU with this TensorFlow version. Internet is needed for initial installation,
MNIST download and DagsHub tracking. MNIST is cached under `.cache/keras/datasets`
inside the project. CPU training may take several minutes or longer.

## Create DagsHub repository and configure authentication

1. Sign in to GitHub and create a repository for the project. Copy its actual browser URL.
2. Sign in at [DagsHub](https://dagshub.com). Use **Create → New Repository** or
   **Connect a Repository** to connect the GitHub repository. Choose the owner
   and repository name and complete creation. The DagsHub and GitHub names may differ.
3. In DagsHub account settings, open **Tokens** and create/copy an access token
   with access to the chosen repository. For organization repositories, use a
   personal username for MLflow authentication and the organization as repo owner.
4. Copy `.env.example` to `.env` and privately fill `DAGSHUB_REPO_OWNER`,
   `DAGSHUB_REPO_NAME`, `DAGSHUB_USER_TOKEN` and your `MLFLOW_TRACKING_USERNAME`.
   Do not paste a token into a notebook cell, terminal command, screenshot or Git.
5. On the repository page open the **Remote** panel and find the **Experiments /
   MLflow** connection information. Copy its tracking URI to the optional
   `MLFLOW_TRACKING_URI` field. The standard form is
   `https://dagshub.com/<owner>/<repository>.mlflow`; this is a pattern, not an actual URL for this assignment.

```powershell
Copy-Item .env.example .env
```

The code reads `.env` using python-dotenv without replacing pre-existing environment
variables. `dagshub.init(repo_owner=..., repo_name=..., mlflow=True)` configures
DagsHub's MLflow connection. The code also sets the MLflow URI and username/token
environment variables explicitly. It fails clearly on missing configuration.
The client can create a missing repository; create it yourself first to control
ownership and visibility. Authentication/network failures stop training instead
of silently falling back to local tracking.

See the current official [DagsHub MLflow guide](https://dagshub.com/docs/integration_guide/mlflow_tracking/)
and [client initialization reference](https://dagshub.com/docs/client/reference/setup).

## Run the required experiments

```powershell
.\.venv\Scripts\python.exe train.py
```

The default is DagsHub, all four runs and all 50 epochs. To use the notebook:

```powershell
.\.venv\Scripts\python.exe -m notebook Assignment3_PartC.ipynb
```

Select **Python 3.11 (Assignment 3)** and run in order. Leave `TRACKING='dagshub'`
and `SMOKE=False` for the submission. Keep the notebook in the same directory as
train.py and verify_project.py. Training cells can take time; do not execute them
twice while still running. After failure restart the kernel and run from the top;
each execution gets a fresh output directory and separate MLflow runs.

| Model | Hidden neurons | Optimizer | Batch size | Epochs |
|---|---|---|---|---|
| Baseline | 128 | Adam | 64 | 50 |
| Experiment 2 | 256 | Adam | 64 | 50 |
| Experiment 3 | 128, 64 | Adam | 64 | 50 |
| Experiment 4 | 128 | SGD | 32 | 50 |

All models start with Flatten and end with Dense(10, softmax). All hidden layers
use ReLU. Sparse categorical cross-entropy works with integer labels; accuracy
measures the fraction of correctly classified digits. Adam's default learning
rate is 0.001; SGD's is 0.01, without momentum. Exact rates are logged.
The last experiment changes two factors, so it does not isolate optimizer effects.

## Data, fairness and selection

MNIST has 60,000 training and 10,000 test images. We normalize to float32 [0,1].
A seeded random permutation splits the original training set into 54,000 training
and 6,000 validation images; arrays are reused unchanged across all four models.
Shapes are `(54000,28,28)`, `(6000,28,28)`, `(10000,28,28)` and corresponding
one-dimensional label arrays. No augmentation or test-based tuning is used.
Seed 42 controls NumPy splitting, TensorFlow initialization and training shuffles.
Deterministic TensorFlow operations improve repeatability; different hardware,
libraries and numerical execution can still affect results.

Each model completes 50 epochs; there is no early stopping or weight restoration.
The comparison uses **final-epoch validation accuracy**, with lower final validation
loss breaking exact ties. After fixing this choice, only the selected final model
is evaluated on the untouched 10,000-image test set. Training metrics are fit's
last-epoch batch aggregates, while validation metrics use final-epoch weights.

The generated analysis states the winner, actual validation metrics, the accuracy
gap and the epoch of lowest validation loss. Review the plotted curves to explain
why this model won in this experiment and possible overfitting: rising validation
loss alongside decreasing training loss is a useful sign. A higher training
accuracy alone is not evidence of better generalization. One seed is not proof
that the winning architecture is universally best.

## What MLflow records

All full runs share `Assignment3-PartC-MNIST`; diagnostic runs use a separate
`Assignment3-PartC-MNIST-SMOKE-ONLY` experiment. Meaningful names identify each model.
The run context manager finishes successful runs and marks exceptions FAILED.
TensorFlow autologging records epoch `accuracy`, `val_accuracy`, `loss`, `val_loss`
and model information. Explicit `curve_*` histories and final `training_accuracy`,
`validation_accuracy`, `training_loss`, `validation_loss` provide stable comparisons.
Hyperparameters include `hidden_layers`, `neurons_per_layer`, per-layer neuron
counts, `optimizer`, `configured_batch_size`, `configured_epochs`, `actual_epochs`,
learning rate, seed and a validation-index fingerprint.

Because fit uses explicitly batched tf.data datasets, use **configured_batch_size**
for comparison; autolog's fit argument can be `None`. `actual_epochs=50` confirms
completion; diagnostic runs retain `configured_epochs=50` but `actual_epochs=1`.

Autologging is enabled with model/dataset logging and checkpoints disabled; epoch
metrics and model summaries remain enabled. Every final model is explicitly saved
as `training/model.keras`, alongside its summary and history, using MLflow's
artifact API. This avoids requiring MLflow 3 logged-model endpoints on DagsHub.
Download it and call `tf.keras.models.load_model(path)` to reload. Comparisons and
analysis are attached to each existing run; no fifth training run is introduced.
The [official TensorFlow autolog reference](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.tensorflow.html)
documents supported versions and options.

## Results and checks

Each timestamped output folder contains:

- `comparison.csv`: real final metrics and configurations for all four models.
- `comparison.png`: four panels showing training/validation accuracy and loss.
- `analysis.md`: validation-based selection, overfitting discussion and test outcome.
- `manifest.json`: actual tracking URI, run IDs, versions, configuration and selection.
- `test_results.json`: selected model's test metrics, omitted for smoke tests.
- Per-model folders: `history.csv`, `model_summary.txt`, `model.keras`.

For quick diagnostics, without changing full-run configurations:

```powershell
.\.venv\Scripts\python.exe train.py --tracking local --smoke
.\.venv\Scripts\python.exe verify_project.py
# Replace the example directory below with the actual path printed by training:
.\.venv\Scripts\python.exe verify_project.py --outputs outputs/full_ACTUAL_TIMESTAMP
```

Smoke mode trains all four architectures for one epoch on 1,024 training images
and 256 validation images. It skips test evaluation. It is never assignment evidence.
To run all 50 epochs locally use `train.py --tracking local`; those results are
valid local experiments but are **not uploaded to DagsHub**. Configure DagsHub and
rerun the full experiments remotely to meet the evidence requirement.

To browse local MLflow, from this project directory:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///.tracking/mlflow.db --host 127.0.0.1 --port 5000
```

Open the printed localhost URL. Local tracking data and model artifacts are under
`.tracking/` and excluded from Git. Keep them if you want to revisit local runs.

## Capture the required DagsHub screenshots

1. Open your DagsHub repository and click **Experiments**. Select
   `Assignment3-PartC-MNIST` and open the MLflow view (the **MLflow** / **Go to MLflow**
   control may be available). Clear stale search filters.
2. **Run list:** show the four FINISHED runs with their meaningful names and metrics.
   Capture the repository/experiment context in the screenshot.
3. **Parameters:** click a run name, expand **Parameters**, and capture hidden-layer
   count, neuron counts, optimizer, configured batch size and epoch counts.
4. **Metrics:** in the same run expand **Metrics**, showing training/validation
   accuracy and loss. Click a metric to show its epoch curve if useful.
5. **Comparison:** return to the run list, check boxes beside at least two full
   runs, then click **Compare**. Show the metric/parameter comparison; choose
   validation accuracy/loss and include training curves if the UI offers them.
6. Use Windows **Win+Shift+S** to capture each view. Save PNG files in a new
   `evidence/` directory and insert them into your assignment Word document.
   Do not capture the token settings or credential fields.

If labels vary with a DagsHub UI update, the repository's **Experiments** tab is
the entry point. See [DagsHub experiment tracking](https://dagshub.com/docs/feature_guide/experiment_tracking/).
The project verifies required metric values by reading them back from MLflow,
but you must still visually confirm the remote runs and collect screenshots.

Copy your **GitHub repository URL** from its browser address bar; copy the
**DagsHub repository URL** from that repository's address bar. Copy the **MLflow
Tracking URI** from DagsHub's Remote panel and cross-check against
`outputs/<your actual run>/manifest.json` → `tracking.tracking_uri`. For remote
runs it must be the DagsHub HTTPS endpoint, not a local SQLite path or the MLflow
UI's browser address. None of these values are fabricated by this project.

## Publish code to GitHub and submit

Review `git status` and `.gitignore` before staging. Ensure `.env` is ignored.
Create an empty GitHub repository if you have not already done so. In this existing
Git checkout, use the real URL copied from GitHub:

```powershell
git add train.py Assignment3_PartC.ipynb requirements.txt README.md .env.example .gitignore verify_project.py verify_notebook.py build_notebook.py SUBMISSION_CHECKLIST.md VERIFICATION.md outputs
git commit -m "Implement Assignment 3 Part C MNIST experiments"
# If origin is already configured, inspect it with git remote -v first.
git remote add origin <YOUR_ACTUAL_GITHUB_REPOSITORY_GIT_URL>
git branch -M main
git push -u origin main
```

The angle-bracket URL is an instruction placeholder; replace it before executing.
Commit screenshots after capturing them if appropriate. Model files are ignored
because they can exceed ordinary Git hosting limits; they are stored in MLflow.
Submit Python code (`train.py`) on DC Connect. The assignment also requests a Word
document with the links, URI and DagsHub screenshots; insert this Part C evidence
in your own document alongside separately prepared Parts A and B. This project
does not write Parts A or B. Follow SUBMISSION_CHECKLIST.md before submitting.

## Troubleshooting

- Missing credentials: fill the private `.env` or use local tracking for diagnostics.
- HTTP 401/403: verify your personal username/token and repository permissions.
- HTTP 404: check the DagsHub owner, repository name and Remote-panel URI.
- Download failure: check internet or put `mnist.npz` in `.cache/keras/datasets/`.
- Import/package errors: confirm the notebook's Python kernel, then reinstall
  requirements in a fresh Python 3.11 environment. Do not mix unrelated coursework
  packages into the environment. See VERIFICATION.md for development-environment limits.
- Training error: inspect FAILED runs, fix the cause and start a fresh execution.
  Partial output folders are not complete submissions.
