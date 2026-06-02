# Qwen3-Embedding-0.6B Binary Classifier

A custom MLflow `PythonModel` that combines:

- **Qwen3-Embedding-0.6B** loaded through the **vLLM** engine on GPU (falls back to `sentence-transformers` on CPU)
- A **trainable linear classification head** (`nn.Linear(1024, 2)`) stored as a separate PyTorch weight file
- A single **`predict(text)`** interface returning binary labels (`0` or `1`)

The model is logged to **Databricks Unity Catalog MLflow Registry** and deployed as a **GPU Model Serving** endpoint, orchestrated from a single notebook.

---

## Repository layout

```
embedding-models/
├── qwen3_classifier_model.py          # Custom MLflow PythonModel (vLLM + classifier head)
├── requirements.txt                   # Pinned runtime dependencies
├── 01_train_register_deploy.ipynb     # End-to-end: train → register → deploy
├── .gitignore
└── README.md
```

---

## Quick-start

### 1. Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| CUDA (GPU path) | 12.x |
| Databricks Runtime | 15.4 ML GPU (on cluster) |

Install local dependencies (for development / linting only – the cluster installs its own):

```bash
pip install -r requirements.txt
```

### 2. Run the notebook

Open `01_train_register_deploy.ipynb` in your Databricks workspace and run all cells.

---

## Model design

### `Qwen3ClassifierModel` ([qwen3_classifier_model.py](qwen3_classifier_model.py))

| Method | Description |
|---|---|
| `load_context(context)` | Detects GPU; loads vLLM engine **or** sentence-transformers; loads classification head weights from `classification_weights.pt` |
| `predict(context, model_input)` | Accepts `str`, `list[str]`, or `DataFrame(text=…)`; returns `list[int]` of binary labels |

#### GPU path (preferred)

```
text → vLLM (Qwen3-Embedding-0.6B, task=embed) → float16 embedding → nn.Linear → argmax → label
```

#### CPU fallback

```
text → sentence-transformers (Qwen3-Embedding-0.6B) → float32 embedding → nn.Linear → argmax → label
```

### Classification head

A single `nn.Linear(1024, 2)` layer whose weights are stored as a separate `classification_weights.pt` file.
To use your own trained weights:

1. Train a linear probe on real embeddings from Qwen3-Embedding-0.6B.
2. Save with `torch.save(head.state_dict(), "classification_weights.pt")`.
3. Place the file in `WEIGHTS_LOCAL_DIR` before running the notebook.

---

## Notebook walkthrough ([01_train_register_deploy.ipynb](01_train_register_deploy.ipynb))

| Cell | Action |
|---|---|
| 0 | Install pip dependencies on the cluster |
| 1 | Set UC catalog / schema / endpoint name constants |
| 2 | Train a demo classification head on synthetic data (replace with real data) |
| 3 | Set MLflow experiment and pip requirements |
| 4 | Log the `Qwen3ClassifierModel` to UC MLflow Registry |
| 5 | Smoke-test: load model locally and run predictions |
| 6 | Create / update a GPU Model Serving endpoint via Databricks SDK |
| 7 | Poll until endpoint is `READY` |
| 8 | Send a test scoring request to the live endpoint |

---

## Notes

- The model downloads Qwen3-Embedding-0.6B from Hugging Face on first load. In air-gapped environments point `MODEL_NAME` to a DBFS / Volumes path.
- `gpu_memory_utilization=0.50` leaves headroom for the classification head and avoids OOM on small T4 GPUs. Increase for larger GPUs.
- The demo training loop in the notebook uses **random synthetic embeddings**. Replace it with real embeddings produced by the Qwen3 model for meaningful classification accuracy.
