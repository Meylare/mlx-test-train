import hashlib
import os
from typing import Tuple


def parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    if not gs_uri or not gs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gs_uri}")
    path = gs_uri[5:]
    bucket, _, blob = path.partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid GCS URI: {gs_uri}")
    return bucket, blob


def _cache_path(gs_uri: str, cache_dir: str) -> str:
    bucket, blob = parse_gs_uri(gs_uri)
    name = os.path.basename(blob) or "video.bin"
    digest = hashlib.sha256(gs_uri.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, f"{bucket}_{digest}_{name}")


def download_gcs_uri(gs_uri: str, cache_dir: str, use_cache: bool = True) -> str:
    if not gs_uri:
        raise ValueError("gcs_uri is required")

    if not gs_uri.startswith("gs://"):
        if os.path.exists(gs_uri):
            return gs_uri
        raise ValueError(f"Not a gs:// URI and file not found: {gs_uri}")

    os.makedirs(cache_dir, exist_ok=True)
    dest = _cache_path(gs_uri, cache_dir)
    if use_cache and os.path.exists(dest):
        return dest

    bucket_name, blob_name = parse_gs_uri(gs_uri)
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(dest)
    return dest
