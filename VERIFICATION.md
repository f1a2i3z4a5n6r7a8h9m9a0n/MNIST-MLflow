# Verification record

Verified on October 23, 2026 using Python 3.11.9 on native Windows CPU.
TensorFlow 2.21.0, Keras 3.15.1, MLflow 3.17.1 and DagsHub client 0.7.3.

## Completed

- Read the assignment's actual Part C requirements and rubric. Implemented Part C only.
- Installed the missing DagsHub client in the project `.venv`; verified import and
  the supported `dagshub.init(repo_owner=..., repo_name=..., mlflow=True)` signature.
- Verified the full requirements list is already satisfied using pip's offline dry run.
- Generated and schema-validated the notebook; compiled every notebook code cell
  and all project Python scripts.
- Downloaded real MNIST through Keras. Verified the 54,000/6,000/10,000 image shapes,
  float32 normalization to [0,1], disjoint train/validation indices and full coverage
  of the original 60,000 training examples.
- Verified all four layer architectures, ReLU/softmax activations, optimizers,
  batch sizes and configured 50 epochs; checked valid ten-class probability outputs.
- Completed a four-model, one-epoch diagnostic execution with local MLflow tracking.
- **Completed all four required 50-epoch experiments on the full training split**
  using the standalone script with local MLflow tracking. No configuration was
  reduced for these full runs.
- Verified four FINISHED full runs, 50 autologged points per training/validation
  metric per run, matching explicit final metrics, saved histories and uploaded
  model artifacts in the local MLflow artifact store.
- Verified selection order and that only the selected run has test metrics.
  Test evaluation used all 10,000 original test examples after selection.
- Executed every notebook code cell in order in a real Jupyter kernel using a
  diagnostic copy with `TRACKING='local'`, `SMOKE=True`. Its final tracking checks
  passed. The submission notebook retains remote tracking and full-run defaults.
- Opened and visually inspected the generated full comparison figure.

The notebook kernel stalled inside the execution sandbox. Its verification
succeeded when rerun with permission to start a local Jupyter kernel.
The executed diagnostic notebook is in `.cache/Assignment3_PartC.smoke.executed.ipynb`;
it is excluded from Git so diagnostic results are not confused with the submission.

## Actual full results

Output directory: `outputs/full_20261023T164119831874Z`.
All values below are observed final-epoch metrics, rounded for display.

| Model | Training accuracy | Validation accuracy | Training loss | Validation loss |
|---|---:|---:|---:|---:|
| Baseline 128 Adam batch 64 | 100.0000% | 98.0000% | 0.000138 | 0.134100 |
| Experiment 2 256 Adam batch 64 | 100.0000% | **98.2000%** | 0.000015 | 0.137461 |
| Experiment 3 128,64 Adam batch 64 | 99.8759% | 97.7833% | 0.004680 | 0.173294 |
| Experiment 4 128 SGD batch 32 | 98.7278% | 97.3167% | 0.049850 | 0.090396 |

Experiment 2 wins under the specified accuracy-first rule, although the baseline
and SGD model have lower final validation losses. The selected model's test
accuracy was **98.36%** and test loss was **0.108697**. It classified 9,836 of 10,000
test images correctly. The original unrounded metrics and run IDs are preserved
in comparison.csv, manifest.json and test_results.json.

Experiment 2's minimum validation loss occurred at epoch 6; its final validation
loss was 0.062214 above that minimum while training loss fell. This supports
possible overfitting after that epoch. It does not invalidate the stipulated
final-epoch comparison, but should be discussed in the student's explanation.

## Environment limitation

To reuse the already installed TensorFlow/MLflow packages, the development `.venv`
was created with `--system-site-packages`. `pip check` reports one incompatibility
in an unrelated inherited package: `ydata-profiling 4.18.0` wants `dacite>=1.9`,
while DagsHub requires `dacite~=1.6.0`. This project never imports ydata-profiling;
all imports, training and tracking checks above passed. The README instructs a
fresh isolated Python 3.11 environment for normal installation, without that
unrelated package. A fresh isolated installation was not performed here.

The initially installed gql 4 dependency also conflicted with MLflow's Graphene
dependency. The requirements pin gql 3.5.3 and graphql-core below 3.3; the resolved
GraphQL packages imported and all MLflow operations passed afterward.

## Not completed and remaining manual actions

The user confirmed that no DagsHub repository exists yet. No DagsHub credentials
were configured. Therefore remote authentication, metric/artifact upload and
appearance in the DagsHub interface were **not tested**, and no DagsHub screenshot,
GitHub URL, DagsHub repository URL or remote tracking URI is claimed as evidence.

Create the GitHub and DagsHub repositories, configure `.env` privately, rerun the
four full experiments with `python train.py` (DagsHub is the default), inspect the
remote FINISHED runs, and capture the four required screenshot views. Publish the
code, then include real links, the actual tracking URI and screenshots in the
assignment Word document. Follow README.md and SUBMISSION_CHECKLIST.md.

The local full results are genuine experiments, but do not satisfy the assignment's
remote evidence requirement. Parts A and B and the final evidence Word document
were not authored as part of this Part C project.
