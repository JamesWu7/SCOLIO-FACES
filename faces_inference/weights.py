import os
from pathlib import Path

from huggingface_hub import hf_hub_download

from .config import PROJECT_ROOT


HF_MODEL_REPO_ID_ENV = "SCOLIO_FACES_MODEL_REPO_ID"
HF_MODEL_TOKEN_ENV = "SCOLIO_FACES_MODEL_TOKEN"


def resolve_weight_path(local_path: Path) -> Path:
    local_path = Path(local_path)
    if local_path.exists():
        return local_path

    repo_id = os.environ.get(HF_MODEL_REPO_ID_ENV)
    token = os.environ.get(HF_MODEL_TOKEN_ENV) or os.environ.get("HF_TOKEN")
    if not repo_id:
        raise FileNotFoundError(
            f"Missing model file: {local_path}. Set {HF_MODEL_REPO_ID_ENV} to a private Hugging Face model repo "
            "or include the file locally."
        )

    try:
        filename = str(local_path.relative_to(PROJECT_ROOT))
    except ValueError:
        filename = local_path.name

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        token=token,
    )
    return Path(downloaded)
