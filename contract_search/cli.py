from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ceiling import cwq_ceiling, webqsp_ceiling
from .baseline import run_baseline
from .audit import run_failure_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="contract_search")
    commands = parser.add_subparsers(dest="command", required=True)
    ceiling = commands.add_parser(
        "webqsp-ceiling", help="execute official WebQSP programs on a released subgraph"
    )
    ceiling.add_argument("--official", type=Path, required=True)
    ceiling.add_argument("--substrate", type=Path, required=True)
    ceiling.add_argument("--output", type=Path, required=True)
    cwq = commands.add_parser(
        "cwq-ceiling", help="execute official CWQ programs on a released subgraph"
    )
    cwq.add_argument("--official", type=Path, required=True)
    cwq.add_argument("--substrate", type=Path, required=True)
    cwq.add_argument("--output", type=Path, required=True)
    baseline = commands.add_parser(
        "baseline", help="run matched-budget executable relation search"
    )
    baseline.add_argument("--substrate", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--scorer-model")
    baseline.add_argument("--device", default="auto")
    baseline.add_argument("--limit", type=int)
    audit = commands.add_parser(
        "phase0-audit", help="classify baseline failures against reference programs"
    )
    audit.add_argument("--predictions", type=Path, required=True)
    audit.add_argument("--ceiling", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "webqsp-ceiling":
        metrics = webqsp_ceiling(args.official, args.substrate, args.output)
        print(json.dumps(metrics, indent=2))
    elif args.command == "cwq-ceiling":
        metrics = cwq_ceiling(args.official, args.substrate, args.output)
        print(json.dumps(metrics, indent=2))
    elif args.command == "baseline":
        metrics = run_baseline(
            args.substrate,
            args.output,
            args.scorer_model,
            args.device,
            args.limit,
        )
        print(json.dumps(metrics, indent=2))
    elif args.command == "phase0-audit":
        metrics = run_failure_audit(
            args.predictions,
            args.ceiling,
            args.output,
        )
        print(json.dumps(metrics, indent=2))
