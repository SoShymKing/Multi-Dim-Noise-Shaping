import hashlib
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _image_key(image_path: str, *, keep_suffix: bool) -> str:
    resolved = Path(image_path).expanduser().resolve(strict=False)
    digest = hashlib.sha1(resolved.as_posix().encode("utf-8")).hexdigest()[:12]
    stem = resolved.stem.replace(" ", "_")
    if keep_suffix:
        return f"{stem}__{digest}{resolved.suffix}"
    return f"{stem}__{digest}"


def generated_dir() -> Path:
    path = _repo_root() / "artifacts" / "generated"
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_path(filename: str) -> str:
    return str(generated_dir() / Path(filename).name)


def mul_array_path(image_path: str) -> str:
    return artifact_path(f"mul_array_{_image_key(image_path, keep_suffix=True)}.npy")


def clean_bp_mul_array_path(image_path: str) -> str:
    return artifact_path(f"mul_array_clean_bp_{_image_key(image_path, keep_suffix=True)}.npy")


def default_output_bmp_path() -> str:
    return artifact_path("output.bmp")


def bp_raw_output_path() -> str:
    return artifact_path("bp_raw_output.npy")


def preview_output_path(prefix: str, image_path: str) -> str:
    return artifact_path(f"{prefix}_{_image_key(image_path, keep_suffix=False)}.bmp")


def lpf_output_path(filename: str) -> str:
    return artifact_path(filename)
