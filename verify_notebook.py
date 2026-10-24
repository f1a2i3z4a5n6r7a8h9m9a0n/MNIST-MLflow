"""Execute a local smoke copy, preserving the submission notebook's full defaults."""
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
cache = root / ".cache"
kernel = cache / "jupyter" / "kernels" / "assignment3-test"
kernel.mkdir(parents=True, exist_ok=True)
(kernel / "kernel.json").write_text(json.dumps({
    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "Assignment3 local verification", "language": "python",
}), encoding="utf-8")
os.environ["JUPYTER_PATH"] = str(cache / "jupyter")
os.environ["JUPYTER_RUNTIME_DIR"] = str(cache / "jupyter-runtime")
os.environ["IPYTHONDIR"] = str(cache / "ipython")
notebook = nbformat.read(root / "Assignment3_PartC.ipynb", as_version=4)
for cell in notebook.cells:
    if cell.cell_type == "code":
        cell.source = cell.source.replace("TRACKING = 'dagshub'", "TRACKING = 'local'").replace("SMOKE = False", "SMOKE = True")
client = NotebookClient(notebook, timeout=600, kernel_name="assignment3-test",
                        resources={"metadata": {"path": str(root)}})
client.execute()
nbformat.write(notebook, cache / "Assignment3_PartC.smoke.executed.ipynb")
assert all(cell.execution_count is not None for cell in notebook.cells if cell.cell_type == "code")
assert not any(out.output_type == "error" for cell in notebook.cells if cell.cell_type == "code" for out in cell.outputs)
print("PASS: every notebook code cell executed in order using a local smoke copy.")
print("Submission notebook retains DagsHub tracking and full 50-epoch defaults.")
