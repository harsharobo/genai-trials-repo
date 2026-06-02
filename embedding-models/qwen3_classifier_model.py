"""
Custom MLflow Model: Qwen3 Embedding (0.6B) + Binary Classification Head

This module defines a custom MLflow PythonModel that:
1. Loads the Qwen3-Embedding-0.6B model via vLLM engine on GPU (if available)
2. Attaches a trained binary classification head (linear layer) on top of embeddings
3. Exposes a predict() interface that takes text input and returns binary class labels

The classification weights are expected as a separate file (classification_weights.pt)
loaded alongside this model during MLflow artifact retrieval.
"""

import os
import logging
from pathlib import Path
import mlflow
import mlflow.pyfunc
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
FMAPI_ENDPOINT = "databricks-qwen3-embedding-0-6b"
WEIGHTS_FILENAME = "classification_weights.pt"
EMBEDDING_DIM = 1024  # Qwen3-Embedding-0.6B hidden size


# ---------------------------------------------------------------------------
# Helper: create a fresh classification head
# ---------------------------------------------------------------------------

def _build_classification_head(embedding_dim: int = EMBEDDING_DIM):
    """Return a simple linear classification head (embedding_dim -> 2)."""
    import torch.nn as nn  # local import — torch may not be present at parse time

    return nn.Linear(embedding_dim, 2)


# ---------------------------------------------------------------------------
# Custom MLflow PythonModel
# ---------------------------------------------------------------------------

class Qwen3ClassifierModel(mlflow.pyfunc.PythonModel):
    """
    MLflow PythonModel wrapping:
      • vLLM-served Qwen3-Embedding-0.6B  (GPU path)
      • Databricks FMAPI fallback           (CPU / serverless path)
      • A binary classification head       (nn.Linear)

    Artifacts expected in the MLflow model directory
    ─────────────────────────────────────────────────
    classification_weights.pt  – PyTorch state-dict for the linear head.
                                  If absent a random-weight head is used
                                  (useful during initial testing).
    """

    def load_context(self, context):
        """
        Called once when the model is loaded.

        GPU path  → vLLM LLM for Qwen3 embedding (local)
        CPU path  → Databricks FMAPI for Qwen3 embedding (remote)
        In both cases the classification head is loaded from the artifact.
        """
        import torch

        # ------------------------------------------------------------------ #
        # 1. Detect hardware
        # ------------------------------------------------------------------ #
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        logger.info("Device selected: %s", self.device)

        # ------------------------------------------------------------------ #
        # 2. Load classification head weights
        # ------------------------------------------------------------------ #
        weights_path = os.path.join(context.artifacts["model_dir"], WEIGHTS_FILENAME)

        self.head = _build_classification_head(EMBEDDING_DIM)

        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            self.head.load_state_dict(state_dict)
            logger.info("Loaded classification weights from %s", weights_path)
        else:
            logger.warning(
                "classification_weights.pt not found – using random weights. "
                "Fine-tune the head before production use."
            )

        self.head.to(self.device)
        self.head.eval()

        # ------------------------------------------------------------------ #
        # 3. Load embedding model
        # ------------------------------------------------------------------ #
        if self.use_gpu:
            self._load_vllm()
        else:
            self._load_fmapi()

    # ---------------------------------------------------------------------- #
    # Private loaders
    # ---------------------------------------------------------------------- #

    def _load_vllm(self):
        """Load Qwen3-Embedding-0.6B via vLLM on GPU."""
        from vllm import LLM

        logger.info("Loading %s via vLLM on GPU …", MODEL_NAME)
        self.vllm_engine = LLM(
            model=MODEL_NAME,
            # task="embed",                   # embedding task
            dtype="float16",
            gpu_memory_utilization=0.50,    # leave headroom for classification head
            trust_remote_code=True,
            max_model_len=512,
        )
        self.embedding_backend = "vllm"
        logger.info("vLLM engine ready.")

    def _load_fmapi(self):
        """Initialise the Databricks FMAPI deploy client for embeddings."""
        from mlflow.deployments import get_deploy_client

        logger.info(
            "Initialising Databricks FMAPI client for endpoint '%s' …",
            FMAPI_ENDPOINT,
        )
        self.deploy_client = get_deploy_client("databricks")
        self.embedding_backend = "fmapi"
        logger.info("Databricks FMAPI client ready.")

    # ---------------------------------------------------------------------- #
    # Embedding helpers
    # ---------------------------------------------------------------------- #

    def _embed_vllm(self, texts: list) -> "np.ndarray":
        """Return (N, D) float32 array of embeddings using vLLM."""
        outputs = self.vllm_engine.embed(texts, use_tqdm=False)
        embeddings = np.array(
            [o.outputs.embedding for o in outputs], dtype=np.float32
        )
        return embeddings

    def _embed_fmapi(self, texts: list) -> "np.ndarray":
        """Return (N, D) float32 array of embeddings using Databricks FMAPI."""
        response = self.deploy_client.predict(
            endpoint=FMAPI_ENDPOINT,
            inputs={"input": texts},
        )
        # response["data"] is a list of {"embedding": [...], "index": i, ...}
        sorted_data = sorted(response["data"], key=lambda x: x["index"])
        embeddings = np.array(
            [item["embedding"] for item in sorted_data], dtype=np.float32
        )
        return embeddings

    def _embed(self, texts: list) -> "np.ndarray":
        if self.embedding_backend == "vllm":
            return self._embed_vllm(texts)
        return self._embed_fmapi(texts)

    # ---------------------------------------------------------------------- #
    # predict
    # ---------------------------------------------------------------------- #

    def predict(self, context, model_input, params=None):
        """
        Perform binary classification on text input.

        Parameters
        ----------
        model_input : pandas.DataFrame | list[str] | str
            • DataFrame with a column named ``"text"``
            • A plain Python list of strings
            • A single string

        Returns
        -------
        list[int]
            Binary class labels (0 or 1) for each input.
        """
        import torch
        import pandas as pd

        # ------------------------------------------------------------------ #
        # Normalise input to list[str]
        # ------------------------------------------------------------------ #
        if isinstance(model_input, pd.DataFrame):
            if "text" not in model_input.columns:
                raise ValueError(
                    "DataFrame input must contain a column named 'text'."
                )
            texts = model_input["text"].tolist()
        elif isinstance(model_input, list):
            texts = [str(t) for t in model_input]
        elif isinstance(model_input, str):
            texts = [model_input]
        else:
            raise TypeError(
                f"Unsupported input type: {type(model_input)}. "
                "Expected str, list[str], or pandas.DataFrame."
            )

        if not texts:
            return []

        # ------------------------------------------------------------------ #
        # Embed
        # ------------------------------------------------------------------ #
        embeddings_np = self._embed(texts)                   # (N, D)
        embeddings_t = torch.tensor(embeddings_np, device=self.device)

        # ------------------------------------------------------------------ #
        # Classify
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            logits = self.head(embeddings_t)                 # (N, 2)
            predictions = logits.argmax(dim=-1).cpu().tolist()

        return predictions

mlflow.models.set_model(Qwen3ClassifierModel())
