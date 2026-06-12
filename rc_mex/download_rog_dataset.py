"""Download a RoG-preprocessed KGQA dataset (WebQSP / CWQ) from HuggingFace.

These are the community-standard per-question Freebase subgraphs released
with RoG (ICLR 2024) and reused by GCR / SubgraphRAG, so numbers are
directly comparable to published tables. Each row: id, question, answer
(list of names), q_entity (topic entities), a_entity, graph (list of
[head, relation, tail] name-level triples).

Run on the Mac (needs VPN + pyarrow), then scp the JSONL to the server:
  python3 -m rc_mex.download_rog_dataset --dataset rmanluo/RoG-webqsp --split test --output data/webqsp/test.jsonl
  python3 -m rc_mex.download_rog_dataset --dataset rmanluo/RoG-cwq --split test --output data/cwq/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import urllib.request


def _ssl_context():
    try:
        import certifi
        import ssl

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return None


def fetch_parquet_urls(dataset: str, split: str) -> list[str]:
    api = f"https://huggingface.co/api/datasets/{dataset}/parquet/default/{split}"
    with urllib.request.urlopen(api, timeout=60, context=_ssl_context()) as response:
        urls = json.load(response)
    if not isinstance(urls, list) or not urls:
        raise RuntimeError(f"No parquet shards listed at {api}: {urls!r}")
    return urls


def download_file(url: str, destination: str) -> None:
    try:
        with urllib.request.urlopen(url, timeout=600, context=_ssl_context()) as response, open(destination, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
    except Exception as exc:  # same fallback pattern as cigr_d_mvp1.download_kqa_pro
        print(f"  urllib download failed ({type(exc).__name__}: {exc}); trying curl")
        subprocess.run(["curl", "-L", "--fail", "--retry", "3", "-o", destination, url], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="rmanluo/RoG-webqsp")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import pyarrow.parquet as pq  # local-only dependency; the server consumes JSONL

    urls = fetch_parquet_urls(args.dataset, args.split)
    print(f"{args.dataset} [{args.split}]: {len(urls)} parquet shard(s)")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    rows = 0
    with open(args.output, "w") as out:
        for url in urls:
            print(f"  downloading {url}")
            with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
                download_file(url, tmp.name)
                table = pq.read_table(tmp.name)
                for batch in table.to_batches():
                    for record in batch.to_pylist():
                        graph = [list(triple) for triple in record.get("graph") or []]
                        out.write(
                            json.dumps(
                                {
                                    "id": record.get("id", ""),
                                    "question": record.get("question", ""),
                                    "answer": list(record.get("answer") or []),
                                    "q_entity": list(record.get("q_entity") or []),
                                    "a_entity": list(record.get("a_entity") or []),
                                    "graph": graph,
                                }
                            )
                            + "\n"
                        )
                        rows += 1
            print(f"  ... {rows} rows written")
    print(f"Wrote {rows} rows to {args.output}")


if __name__ == "__main__":
    main()
