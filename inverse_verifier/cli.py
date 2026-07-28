from __future__ import annotations

import argparse
import json
from pathlib import Path

from .comparator_corpus import build_comparator_corpus
from .rescore import SCORER_KINDS, rescore_run, rescore_with_comparator
from .comparator import (
    COMPARATOR_INPUT_MODES,
    evaluate_comparator,
    materialize_comparator_data,
    train_comparator,
)
from .data import prepare_dataset
from .dataset_builder import build_naturalized_dataset
from .evaluate import evaluate, evaluate_gold_generation
from .generalization import evaluate_faithful_generalization
from .model import train_model
from .openai_naturalize import run_openai_naturalization
from .selector import run_verifier_pipeline
from .semantic_benchmark import (
    adjudicate_semantic_benchmark,
    build_semantic_benchmark,
)
from .selection_experiment import (
    audit_view_runs,
    build_fixed_supervision_study,
    export_answer_equivalent_path_audit,
)
from .retrieval import SRTK_SCORER, run_retrieval_probe
from .synthetic import evaluate_faithful_generation, naturalize_corpus, synthesize_corpus
from .training_data import (
    merge_executable_direction_pairs,
    prepare_executable_direction_pairs,
    repair_faithful_corpus,
)


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

    synthesize = subparsers.add_parser(
        "synthesize", help="sample balanced executable paths with faithful question targets"
    )
    synthesize.add_argument("--kqa-kb", type=Path, default=Path("data/kqa_pro/kb.json"))
    synthesize.add_argument(
        "--webqsp-graphs", type=Path, default=Path("data/webqsp/train.jsonl")
    )
    synthesize.add_argument(
        "--output", type=Path, default=Path("runs/inverse_verifier/faithful_data")
    )
    synthesize.add_argument("--paths", type=int, default=30_000)
    synthesize.add_argument("--seed", type=int, default=17)

    naturalize = subparsers.add_parser(
        "naturalize", help="rewrite controlled faithful questions with a local Ollama model"
    )
    naturalize.add_argument(
        "--data", type=Path, default=Path("runs/inverse_verifier/faithful_data")
    )
    naturalize.add_argument(
        "--output", type=Path, default=Path("runs/inverse_verifier/faithful_data_natural")
    )
    naturalize.add_argument("--model", default="qwen3:8b")
    naturalize.add_argument(
        "--ollama-host",
        action="append",
        help=(
            "Ollama endpoint; repeat once per GPU-backed server. "
            "Defaults to http://127.0.0.1:11434."
        ),
    )

    openai_naturalize = subparsers.add_parser(
        "naturalize-openai",
        help="naturalize a bounded faithful corpus with the resumable OpenAI Batch API",
    )
    openai_naturalize.add_argument(
        "--data", type=Path, default=Path("runs/inverse_verifier/faithful_data")
    )
    openai_naturalize.add_argument(
        "--output", type=Path, default=Path("runs/inverse_verifier/faithful_data_openai")
    )
    openai_naturalize.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    openai_naturalize.add_argument("--max-paths", type=int, default=3_000)
    openai_naturalize.add_argument("--max-negatives", type=int, default=3)
    openai_naturalize.add_argument("--max-budget-usd", type=float, default=8.0)
    openai_naturalize.add_argument("--dry-run", action="store_true")

    build_dataset = subparsers.add_parser(
        "build-naturalized-dataset",
        help="build a validated natural corpus using grounded relation semantics",
    )
    build_dataset.add_argument(
        "--data", type=Path, default=Path("runs/inverse_verifier/faithful_data_30k_probe")
    )
    build_dataset.add_argument(
        "--output", type=Path, default=Path("runs/inverse_verifier/naturalized_dataset")
    )
    build_dataset.add_argument("--kqa-kb", type=Path, default=Path("data/kqa_pro/kb.json"))
    build_dataset.add_argument("--webqsp-graphs", type=Path, default=Path("data/webqsp/train.jsonl"))
    build_dataset.add_argument("--max-paths", type=int, default=3_000)
    build_dataset.add_argument("--max-negatives", type=int, default=3)

    faithfulness = subparsers.add_parser(
        "faithfulness", help="compare gold and executable-negative generated questions"
    )
    faithfulness.add_argument(
        "--data",
        type=Path,
        default=Path("runs/inverse_verifier/faithful_data_natural/dev_faithful.jsonl"),
    )
    faithfulness.add_argument("--model", required=True)
    faithfulness.add_argument(
        "--output", type=Path, default=Path("runs/inverse_verifier/faithfulness_eval")
    )
    faithfulness.add_argument("--semantic-model", default="BAAI/bge-small-en-v1.5")
    faithfulness.add_argument("--limit", type=int, default=500)
    faithfulness.add_argument("--batch-size", type=int, default=16)
    faithfulness.add_argument("--device", default="auto")

    generalize = subparsers.add_parser(
        "generalize", help="evaluate faithful generation on training-relative held-out slices"
    )
    generalize.add_argument(
        "--train-data",
        type=Path,
        default=Path("runs/inverse_verifier/naturalized_dataset_3000_v8/train_faithful.jsonl"),
    )
    generalize.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    generalize.add_argument("--model", required=True)
    generalize.add_argument("--semantic-model", default="BAAI/bge-small-en-v1.5")
    generalize.add_argument(
        "--splits", default="test_kqa_val,test_executable_webqsp"
    )
    generalize.add_argument(
        "--output", type=Path, default=Path("runs/inverse_verifier/generalization_eval")
    )
    generalize.add_argument("--batch-size", type=int, default=32)
    generalize.add_argument("--device", default="auto")

    corpus = subparsers.add_parser(
        "comparator-corpus",
        help="turn a verify run into listwise comparator training data",
    )
    corpus.add_argument("--predictions", type=Path, required=True)
    corpus.add_argument("--output", type=Path, required=True)
    corpus.add_argument("--hard-negatives", type=int, default=12)
    corpus.add_argument("--random-negatives", type=int, default=4)
    corpus.add_argument("--dev-fraction", type=float, default=0.1)
    corpus.add_argument("--seed", type=int, default=17)

    supervision_study = subparsers.add_parser(
        "build-supervision-study",
        help="build fixed candidate pools for annotated/denotation label ablations",
    )
    supervision_study.add_argument("--predictions", type=Path, required=True)
    supervision_study.add_argument("--output", type=Path, required=True)
    supervision_study.add_argument("--hard-negatives", type=int, default=12)
    supervision_study.add_argument("--random-negatives", type=int, default=4)
    supervision_study.add_argument("--dev-fraction", type=float, default=0.1)
    supervision_study.add_argument("--seed", type=int, default=17)

    audit_views = subparsers.add_parser(
        "audit-views",
        help="compare cross-view disagreement with same-view model disagreement",
    )
    audit_views.add_argument("--predictions", type=Path, required=True)
    audit_views.add_argument(
        "--generated-runs", type=Path, nargs="+", required=True
    )
    audit_views.add_argument("--path-runs", type=Path, nargs="+", required=True)
    audit_views.add_argument(
        "--path-labels",
        type=Path,
        help="optional independently labeled candidates.jsonl from export-path-audit",
    )
    audit_views.add_argument("--output", type=Path, required=True)

    export_path_audit = subparsers.add_parser(
        "export-path-audit",
        help="export raw non-annotated paths that exactly reach the gold answer",
    )
    export_path_audit.add_argument("--predictions", type=Path, required=True)
    export_path_audit.add_argument("--graphs", type=Path)
    export_path_audit.add_argument("--output", type=Path, required=True)

    rescore = subparsers.add_parser(
        "rescore",
        help="re-rank a finished run with a different comparator, no graph pass",
    )
    rescore.add_argument("--predictions", type=Path, required=True)
    rescore.add_argument(
        "--model", help="off-the-shelf model id; mutually exclusive with --comparator"
    )
    rescore.add_argument("--kind", choices=SCORER_KINDS, default="cross_encoder")
    rescore.add_argument("--output", type=Path)
    rescore.add_argument("--batch-size", type=int, default=64)
    rescore.add_argument("--device", default="auto")
    rescore.add_argument("--no-endpoint-filter", action="store_true")
    rescore.add_argument(
        "--comparator",
        help="trained comparator checkpoint; uses its stored input mode instead of --kind",
    )
    rescore.add_argument(
        "--graphs", type=Path, help="graphs file, to rebuild paths for path-using modes"
    )

    prepare_comparator = subparsers.add_parser(
        "prepare-comparator",
        help="materialize generated questions and hard-negative candidate sets",
    )
    prepare_comparator.add_argument("--data", type=Path, required=True)
    prepare_comparator.add_argument("--generator", required=True)
    prepare_comparator.add_argument("--output", type=Path, required=True)
    prepare_comparator.add_argument("--batch-size", type=int, default=8)
    prepare_comparator.add_argument("--device", default="auto")
    prepare_comparator.add_argument("--limit", type=int)

    train_comparator_parser = subparsers.add_parser(
        "train-comparator",
        help="train a listwise cross-encoder over path candidate sets",
    )
    train_comparator_parser.add_argument("--data", type=Path, required=True)
    train_comparator_parser.add_argument("--output", type=Path, required=True)
    train_comparator_parser.add_argument(
        "--base-model", default="microsoft/deberta-v3-base"
    )
    train_comparator_parser.add_argument(
        "--input-mode",
        choices=COMPARATOR_INPUT_MODES,
        default="question_generated_path",
    )
    train_comparator_parser.add_argument("--epochs", type=int, default=4)
    train_comparator_parser.add_argument("--batch-size", type=int, default=4)
    train_comparator_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_comparator_parser.add_argument("--device", default="auto")
    train_comparator_parser.add_argument("--limit", type=int)
    train_comparator_parser.add_argument("--seed", type=int, default=17)

    evaluate_comparator_parser = subparsers.add_parser(
        "evaluate-comparator",
        help="evaluate a cross-encoder and optional cosine baseline",
    )
    evaluate_comparator_parser.add_argument("--data", type=Path, required=True)
    evaluate_comparator_parser.add_argument("--model", required=True)
    evaluate_comparator_parser.add_argument("--output", type=Path, required=True)
    evaluate_comparator_parser.add_argument("--split", default="dev")
    evaluate_comparator_parser.add_argument("--semantic-model")
    evaluate_comparator_parser.add_argument("--batch-size", type=int, default=8)
    evaluate_comparator_parser.add_argument("--device", default="auto")
    evaluate_comparator_parser.add_argument("--limit", type=int)

    semantic_benchmark = subparsers.add_parser(
        "build-semantic-benchmark",
        help="label generated questions by text-level semantic equivalence",
    )
    semantic_benchmark.add_argument("--data", type=Path, required=True)
    semantic_benchmark.add_argument("--output", type=Path, required=True)
    semantic_benchmark.add_argument("--model", default="gpt-4o-2024-11-20")
    semantic_benchmark.add_argument("--workers", type=int, default=3)
    semantic_benchmark.add_argument("--limit", type=int)

    semantic_adjudication = subparsers.add_parser(
        "adjudicate-semantic-benchmark",
        help="independently audit disputed semantic-equivalence labels",
    )
    semantic_adjudication.add_argument("--data", type=Path, required=True)
    semantic_adjudication.add_argument("--output", type=Path, required=True)
    semantic_adjudication.add_argument("--model", default="gpt-4o-2024-11-20")
    semantic_adjudication.add_argument("--workers", type=int, default=3)
    semantic_adjudication.add_argument("--agreement-sample", type=int, default=30)
    semantic_adjudication.add_argument("--seed", type=int, default=17)

    repair_data = subparsers.add_parser(
        "repair-training-data",
        help="add direction contrasts and remove malformed generated questions",
    )
    repair_data.add_argument("--data", type=Path, required=True)
    repair_data.add_argument("--glossary", type=Path, required=True)
    repair_data.add_argument("--output", type=Path, required=True)

    prepare_direction = subparsers.add_parser(
        "prepare-direction-pairs",
        help="select paired executable forward/backward one-hop training paths",
    )
    prepare_direction.add_argument("--data", type=Path, required=True)
    prepare_direction.add_argument("--base", type=Path, required=True)
    prepare_direction.add_argument("--glossary", type=Path, required=True)
    prepare_direction.add_argument("--output", type=Path, required=True)

    merge_direction = subparsers.add_parser(
        "merge-direction-pairs",
        help="validate and merge naturalized executable direction pairs",
    )
    merge_direction.add_argument("--base", type=Path, required=True)
    merge_direction.add_argument("--pairs", type=Path, required=True)
    merge_direction.add_argument("--output", type=Path, required=True)

    train = subparsers.add_parser("train", help="fine-tune the inverse generator")
    train.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    train.add_argument("--output", type=Path, default=DEFAULT_TRAIN_DIR)
    train.add_argument("--base-model", default="google/flan-t5-small")
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--rank-weight", type=float, default=1.0)
    train.add_argument(
        "--relation-glossary",
        type=Path,
        help=(
            "grounded KG relation glossary used to render canonical facts; "
            "it is embedded in the trained checkpoint"
        ),
    )
    train.add_argument(
        "--regime",
        choices=("kqa_only", "multi_kg", "faithful_synthetic"),
        default="kqa_only",
    )
    train.add_argument(
        "--objective",
        choices=(
            "causal_inverse",
            "faithful_inverse",
            "type_aware_generator",
            "inverse",
            "direct",
            "joint",
            "ranker",
        ),
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
        "--model", default="runs/inverse_verifier/joint_ranker_multi_kg_b4/model"
    )
    verify.add_argument("--retriever-model", default=SRTK_SCORER)
    verify.add_argument(
        "--output", type=Path, default=Path("runs/inverse_verifier/path_verifier")
    )
    verify.add_argument("--limit", type=int, default=25)
    verify.add_argument("--device", default="auto")
    verify.add_argument(
        "--comparison-mode",
        choices=("cosine", "cross_encoder"),
        default="cosine",
    )
    verify.add_argument(
        "--comparator-model",
        help="trained comparator checkpoint; required for cross_encoder mode",
    )

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
    elif args.command == "synthesize":
        print("[1/3] Loading executable KQA Pro and WebQSP training graphs", flush=True)
        print("[2/3] Sampling balanced one-, two-, and three-hop paths", flush=True)
        manifest = synthesize_corpus(
            args.kqa_kb,
            args.webqsp_graphs,
            args.output,
            total_paths=args.paths,
            seed=args.seed,
        )
        print("[3/3] Writing faithful train/dev corpus", flush=True)
        print(json.dumps(manifest, indent=2))
        print(f"Prepared faithful data in {args.output}")
    elif args.command == "naturalize":
        print("Naturalizing faithful questions with complete-hop checks", flush=True)
        manifest = naturalize_corpus(
            args.data,
            args.output,
            model=args.model,
            host=args.ollama_host or ["http://127.0.0.1:11434"],
        )
        print(json.dumps(manifest, indent=2))
        print(f"Prepared natural faithful data in {args.output}")
    elif args.command == "naturalize-openai":
        manifest = run_openai_naturalization(
            args.data,
            args.output,
            model=args.model,
            max_paths=args.max_paths,
            max_negatives=args.max_negatives,
            max_budget_usd=args.max_budget_usd,
            dry_run=args.dry_run,
        )
        print(json.dumps(manifest, indent=2))
        print(f"Prepared OpenAI-naturalized data in {args.output}")
    elif args.command == "build-naturalized-dataset":
        manifest = build_naturalized_dataset(
            args.data,
            args.output,
            args.kqa_kb,
            args.webqsp_graphs,
            max_paths=args.max_paths,
            max_negatives=args.max_negatives,
        )
        print(json.dumps(manifest, indent=2))
        print(f"Prepared validated natural dataset in {args.output}")
    elif args.command == "faithfulness":
        metrics = evaluate_faithful_generation(
            args.data,
            args.model,
            args.output,
            semantic_model=args.semantic_model,
            limit=args.limit,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(json.dumps(metrics, indent=2))
        print(f"Wrote faithfulness evaluation to {args.output}")
    elif args.command == "generalize":
        metrics = evaluate_faithful_generalization(
            args.train_data,
            args.data,
            args.model,
            args.semantic_model,
            args.output,
            [name.strip() for name in args.splits.split(",") if name.strip()],
            batch_size=args.batch_size,
            device=args.device,
        )
        print(json.dumps(metrics["training_relative_coverage"], indent=2))
        print(f"Wrote faithful generalization evaluation to {args.output}")
    elif args.command == "comparator-corpus":
        manifest = build_comparator_corpus(
            args.predictions,
            args.output,
            seed=args.seed,
            dev_fraction=args.dev_fraction,
            hard=args.hard_negatives,
            random_count=args.random_negatives,
        )
        print(json.dumps(manifest, indent=2))
        print(f"Wrote comparator corpus to {args.output}")
    elif args.command == "build-supervision-study":
        manifest = build_fixed_supervision_study(
            args.predictions,
            args.output,
            seed=args.seed,
            dev_fraction=args.dev_fraction,
            hard_negatives=args.hard_negatives,
            random_negatives=args.random_negatives,
        )
        print(json.dumps(manifest, indent=2))
        print(f"Wrote fixed supervision study to {args.output}")
    elif args.command == "audit-views":
        metrics = audit_view_runs(
            args.predictions,
            args.generated_runs,
            args.path_runs,
            args.output,
            path_labels=args.path_labels,
        )
        comparison = metrics["comparison"]
        print(
            "Cross-view excess oracle gain: "
            f"{comparison['cross_view_excess_oracle_gain']}"
        )
        print(f"Wrote representation audit to {args.output}")
    elif args.command == "export-path-audit":
        manifest = export_answer_equivalent_path_audit(
            args.predictions,
            args.output,
            graphs=args.graphs,
        )
        print(json.dumps(manifest, indent=2))
        print(f"Wrote raw path audit to {args.output}")
    elif args.command == "rescore":
        if bool(args.model) == bool(args.comparator):
            raise SystemExit("rescore needs exactly one of --model or --comparator")
        if args.comparator:
            result = rescore_with_comparator(
                args.predictions,
                args.comparator,
                args.output,
                graphs=args.graphs,
                device=args.device,
                batch_size=args.batch_size,
                endpoint_filter=not args.no_endpoint_filter,
            )
            print(json.dumps(result, indent=2))
            return
        result = rescore_run(
            args.predictions,
            args.model,
            args.kind,
            args.output,
            device=args.device,
            batch_size=args.batch_size,
            endpoint_filter=not args.no_endpoint_filter,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "prepare-comparator":
        manifest = materialize_comparator_data(
            args.data,
            args.generator,
            args.output,
            batch_size=args.batch_size,
            device_name=args.device,
            limit=args.limit,
        )
        print(json.dumps(manifest["splits"], indent=2))
        print(f"Wrote comparator candidate sets to {args.output}")
    elif args.command == "train-comparator":
        run = train_comparator(
            args.data,
            args.output,
            args.base_model,
            args.input_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device_name=args.device,
            limit=args.limit,
            seed=args.seed,
        )
        print(
            f"Best development R@1={run['best_dev_recall_at_1']:.3f} "
            f"MRR={run['best_dev_mrr']:.3f}"
        )
        print(f"Saved comparator to {args.output / 'model'}")
    elif args.command == "evaluate-comparator":
        metrics = evaluate_comparator(
            args.data,
            args.model,
            args.output,
            semantic_model=args.semantic_model,
            split=args.split,
            batch_size=args.batch_size,
            device_name=args.device,
            limit=args.limit,
        )
        print(json.dumps(metrics, indent=2))
        print(f"Wrote comparator evaluation to {args.output}")
    elif args.command == "build-semantic-benchmark":
        manifest = build_semantic_benchmark(
            args.data,
            args.output,
            model=args.model,
            workers=args.workers,
            limit=args.limit,
        )
        print(json.dumps(manifest, indent=2))
        print(f"Wrote semantic comparator benchmark to {args.output}")
    elif args.command == "adjudicate-semantic-benchmark":
        manifest = adjudicate_semantic_benchmark(
            args.data,
            args.output,
            model=args.model,
            workers=args.workers,
            agreement_sample=args.agreement_sample,
            seed=args.seed,
        )
        print(json.dumps(manifest, indent=2))
        print(f"Wrote adjudicated semantic benchmark to {args.output}")
    elif args.command == "repair-training-data":
        manifest = repair_faithful_corpus(args.data, args.output, args.glossary)
        print(json.dumps(manifest["counts"], indent=2))
        print(f"Wrote repaired faithful corpus to {args.output}")
    elif args.command == "prepare-direction-pairs":
        manifest = prepare_executable_direction_pairs(
            args.data, args.base, args.glossary, args.output
        )
        print(json.dumps(manifest, indent=2))
    elif args.command == "merge-direction-pairs":
        manifest = merge_executable_direction_pairs(args.base, args.pairs, args.output)
        print(json.dumps(manifest, indent=2))
    elif args.command == "train":
        names = {
            "kqa_only": ("train.jsonl", "dev.jsonl"),
            "multi_kg": ("train_multi_kg.jsonl", "dev_multi_kg.jsonl"),
            "faithful_synthetic": ("train_faithful.jsonl", "dev_faithful.jsonl"),
        }
        train_name, dev_name = names[args.regime]
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
            relation_glossary_path=args.relation_glossary,
        )
        if args.objective == "causal_inverse":
            print(f"Best development token NLL: {run['best_dev_token_nll']:.3f}")
        elif args.objective == "type_aware_generator":
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
            args.comparison_mode,
            args.comparator_model,
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
