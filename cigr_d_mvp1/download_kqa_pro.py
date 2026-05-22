from __future__ import annotations

import argparse
import subprocess
import urllib.request
from pathlib import Path


URLS = {
    "kb.json": "https://huggingface.co/datasets/drt/kqa_pro/resolve/main/kb.json",
    "train.json": "https://huggingface.co/datasets/drt/kqa_pro/resolve/main/train.json",
    "val.json": "https://huggingface.co/datasets/drt/kqa_pro/resolve/main/val.json",
    "test.json": "https://huggingface.co/datasets/drt/kqa_pro/resolve/main/test.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download KQA Pro JSON files from Hugging Face.")
    parser.add_argument("--output", default="data/kqa_pro")
    parser.add_argument(
        "--files",
        default="kb.json,val.json",
        help="Comma-separated subset. Default is enough for MVP1 smoke runs.",
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for filename in args.files.split(","):
        filename = filename.strip()
        if filename not in URLS:
            raise SystemExit(f"unknown file {filename}; valid files: {', '.join(URLS)}")
        destination = output / filename
        if destination.exists():
            print(f"skip existing {destination}")
            continue
        print(f"downloading {filename} -> {destination}")
        download(URLS[filename], destination)


def download(url: str, destination: Path) -> None:
    try:
        urllib.request.urlretrieve(url, destination)
    except Exception as exc:
        print(f"urllib download failed ({type(exc).__name__}: {exc}); trying curl")
        completed = subprocess.run(
            ["curl", "-L", "--fail", "--retry", "3", "-o", str(destination), url],
            check=False,
        )
        if completed.returncode != 0:
            destination.unlink(missing_ok=True)
            raise SystemExit(
                "download failed. Try downloading from the Hugging Face page manually "
                "and put the files under data/kqa_pro/."
            )


if __name__ == "__main__":
    main()
