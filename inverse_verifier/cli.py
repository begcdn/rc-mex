from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import prepare_dataset
from .evaluate import evaluate, evaluate_gold_generation
from .model import train_model
from .selector import run_verifier_pipeline
from .retrieval import SRTK_SCORER, run_retrieval_probe


DEFAULT_DATA_DIR = Path("runs/inverse_verifier/data")
DEFAULT_TRAIN_DIR = Path("runs/inverse_verifier/joint")
DEFAULT_EVAL_DIR = Path("runs/inverse_verifier/evaluation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inverse_verifier",
        description="Prepare, train, and evaluate the inverse path-to-question verifier.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="build leakage-resistant path/question splits")
    prepare.add_argument("--kqa-train", type=Path, default=Path("data/kqa_pro/train.json"))
    prepare.add_argument("--kqa-val", type=Path, default=Path("data/kqa_pro/val.json"))
    prepare.add_argument("--kqa-kb", type=Path, default=Path("data/kqa_pro/kb.json"))
    prepare.add_argument(
        "--webqsp-test",
        type=Path,
        default=Path("data/webqsp_official/WebQSP/data/WebQSP.test.json"),
    )
    prepare.add_argument("--webqsp-graphs", type=Path, default=Path("data/webqsp/test.jsonl"))
    prepare.add_argument(
        "--webqsp-train",
        type=Path,
        default=Path("data/webqsp_official/WebQSP/data/WebQSP.train.json"),
    )
    prepare.add_argument("--webqsp-train-graphs", type=Path, default=Path("data/webqsp/train.jsonl"))
    prepare.add_argument("--output", type=Path, default=DEFAULT_DATA_DIR)

    train = subparsers.add_parser("train", help="fine-tune the inverse generator")
    train.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    train.add_argument("--output", type=Path, default=DEFAULT_TRAIN_DIR)
    train.add_argument("--base-model", default="google/flan-t5-small")
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--rank-weight", type=float, default=1.0)
    train.add_argument("--regime", choices=("kqa_only", "multi_kg"), default="kqa_only")
    train.add_argument(
        "--objective",
        choices=("type_aware_generator", "inverse", "direct", "joint", "ranker"),
        default="type_aware_generator",
    )
    train.add_argument("--device", default="auto")
    train.add_argument("--limit", type=int)

    evaluate_parser = subparsers.add_parser("evaluate", help="measure generation and path ranking")
    evaluate_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    evaluate_parser.add_argument("--output", type=Path, default=DEFAULT_EVAL_DIR)
    evaluate_parser.add_argument("--trained-model", default=str(DEFAULT_TRAIN_DIR / "model"))
    evaluate_parser.add_argument("--direct-model")
    evaluate_parser.add_argument("--joint-model")
    evaluate_parser.add_argument("--ranker-model")
    evaluate_parser.add_argument("--base-model", default="google/flan-t5-small")
    evaluate_parser.add_argument(
        "--splits",
        default="dev,test_unseen_relation,test_unseen_composition,test_kqa_val,test_cross_kg_webqsp",
    )
    evaluate_parser.add_argument("--batch-size", type=int, default=16)
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.add_argument("--limit-per-split", type=int)
    evaluate_parser.add_argument("--generation-examples", type=int, default=64)
    evaluate_parser.add_argument("--skip-pretrained", action="store_true")

    generate = subparsers.add_parser(
        "generate", help="test question generation from held-out gold paths only"
    )
    generate.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    generate.add_argument(
        "--model", default="runs/inverse_verifier/type_aware_generator_multi_kg/model"
    )
    generate.add_argument(
        "--output", type=Path, default=Path("runs/inverse_verifier/gold_generation")
    )
    generate.add_argument(
        "--splits",
        default="dev,test_unseen_relation,test_unseen_composition,test_kqa_val,test_cross_kg_webqsp,test_executable_webqsp",
    )
    generate.add_argument("--batch-size", type=int, default=16)
    generate.add_argument("--limit-per-split", type=int)
    generate.add_argument("--device", default="auto")

    verify = subparsers.add_parser(
        "verify", help="propose executable paths and verify them by generated-question meaning"
    )
    verify.add_argument(
        "--questions",
        type=Path,
        default=Path("data/webqsp_official/WebQSP/data/WebQSP.test.json"),
    )
    verify.add_argument("--graphs", type=Path, default=Path("data/webqsp/test.jsonl"))
    verify.add_argument(
        "--model", default="runs/inverse_verifier/type_aware_generator_multi_kg/model"
    )
    verify.add_argument("--retriever-model", default=SRTK_SCORER)
    verify.add_argument(
        "--output", type=Path, default=Path("runs/inverse_verifier/path_verifier")
    )
    verify.add_argument("--limit", type=int, default=25)
    verify.add_argument("--device", default="auto")

    retrieve = subparsers.add_parser(
        "retrieve", help="retrieve a high-recall path set and union subgraph with SRTK"
    )
    retrieve.add_argument(
        "--questions",
        type=Path,
        default=Path("data/webqsp_official/WebQSP/data/WebQSP.test.json"),
    )
    retrieve.add_argument("--graphs", type=Path, default=Path("data/webqsp/test.jsonl"))
    retrieve.add_argument("--retriever-model", default=SRTK_SCORER)
    retrieve.add_argument(
        "--output", type=Path, default=Path("runs/inverse_verifier/path_retrieval")
    )
    retrieve.add_argument("--limit", type=int, default=25)
    retrieve.add_argument("--device", default="auto")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        print("[1/3] Extracting KQA Pro relation chains", flush=True)
        print("[2/3] Extracting official WebQSP inferential chains", flush=True)
        manifest = prepare_dataset(
            args.kqa_train,
            args.kqa_val,
            args.webqsp_test,
            args.output,
            args.webqsp_graphs,
            args.kqa_kb,
            args.webqsp_train,
            args.webqsp_train_graphs,
        )
        print("[3/3] Writing split manifest", flush=True)
        print(json.dumps(manifest["counts"], indent=2))
        print(f"Prepared data in {args.output}")
    elif args.command == "train":
        train_name = "train_multi_kg.jsonl" if args.regime == "multi_kg" else "train.jsonl"
        dev_name = "dev_multi_kg.jsonl" if args.regime == "multi_kg" else "dev.jsonl"
        run = train_model(
            args.data / train_name,
            args.data / dev_name,
            args.output,
            args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            rank_weight=args.rank_weight,
            device_name=args.device,
            limit=args.limit,
            regime=args.regime,
            objective=args.objective,
        )
        if args.objective == "type_aware_generator":
            print(
                "Best development type compatibility accuracy: "
                f"{run['best_dev_pair_accuracy']:.3f}"
            )
        else:
            print(f"Best development pair accuracy: {run['best_dev_pair_accuracy']:.3f}")
        print(f"Saved model to {args.output / 'model'}")
    elif args.command == "evaluate":
        metrics = evaluate(
            args.data,
            args.output,
            args.trained_model,
            args.base_model,
            [name.strip() for name in args.splits.split(",") if name.strip()],
            device=args.device,
            batch_size=args.batch_size,
            limit_per_split=args.limit_per_split,
            generation_examples=args.generation_examples,
            include_pretrained=not args.skip_pretrained,
            direct_model=args.direct_model,
            joint_model=args.joint_model,
            ranker_model=args.ranker_model,
        )
        print(f"Wrote evaluation to {args.output} in {metrics['elapsed_seconds']:.1f}s")
    elif args.command == "generate":
        metrics = evaluate_gold_generation(
            args.data,
            args.output,
            args.model,
            [name.strip() for name in args.splits.split(",") if name.strip()],
            device=args.device,
            batch_size=args.batch_size,
            limit_per_split=args.limit_per_split,
        )
        print(f"Wrote gold-path generation evaluation to {args.output}")
        print(f"Completed in {metrics['elapsed_seconds']:.1f}s")
    elif args.command == "verify":
        metrics = run_verifier_pipeline(
            args.questions,
            args.graphs,
            args.model,
            args.retriever_model,
            args.output,
            args.limit,
            args.device,
        )
        print(
            f"Path recall@100={metrics['proposal_recall']['recall_at_100']:.3f} "
            f"selected_gold={metrics['selected_gold_path_accuracy']:.3f} "
            f"answer_f1={metrics['answer_f1']:.3f}"
        )
        print(f"Wrote verifier evaluation to {args.output}")
    elif args.command == "retrieve":
        from .selector import supported_questions

        metrics = run_retrieval_probe(
            args.questions,
            args.graphs,
            args.retriever_model,
            args.output,
            args.limit,
            args.device,
            supported_questions(args.questions),
        )
        print(
            f"Path recall@100={metrics['path_recall']['recall_at_100']:.3f} "
            f"answer_coverage={metrics['answer_coverage']:.3f}"
        )
        print(f"Wrote retrieval evaluation to {args.output}")


if __name__ == "__main__":
    main()
