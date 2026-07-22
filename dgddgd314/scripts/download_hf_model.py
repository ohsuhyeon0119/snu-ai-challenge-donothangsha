import argparse
import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def default_name(repo_id):
    return repo_id.rstrip("/").split("/")[-1]


def add_directory_to_tar(tar, directory):
    root_name = directory.name
    for path in directory.rglob("*"):
        if path.is_file():
            tar.add(path, arcname=str(Path(root_name) / path.relative_to(directory)))


def replace_directory(src, dst):
    if dst.exists():
        shutil.rmtree(dst)
    ignore = shutil.ignore_patterns(".cache")
    shutil.copytree(src, dst, symlinks=False, ignore=ignore)


def main():
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model snapshot locally and optionally package it for upload."
    )
    parser.add_argument("--repo-id", default="google/paligemma2-10b-pt-224")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--cache-dir", default="local_models/.hf_cache")
    parser.add_argument("--archive", default=None)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--skip-if-exists", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    model_name = default_name(args.repo_id)
    out_dir = Path(args.out_dir or Path("local_models") / model_name)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_if_exists and (out_dir / "config.json").exists():
        print(f"Using existing model at: {out_dir.resolve()}")
        return

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        token=args.token,
        cache_dir=str(cache_dir),
    )
    replace_directory(Path(snapshot_path), out_dir)

    manifest = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "snapshot_path": str(Path(snapshot_path).resolve()),
        "exported_path": str(out_dir.resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = out_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Downloaded model to: {out_dir.resolve()}")

    if args.no_archive:
        return

    archive_path = Path(args.archive or Path("local_models") / f"{out_dir.name}.tar")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w") as tar:
        add_directory_to_tar(tar, out_dir)

    print(f"Wrote archive: {archive_path.resolve()}")


if __name__ == "__main__":
    main()
