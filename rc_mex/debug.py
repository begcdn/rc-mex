from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from cigr_d_mvp1.io_utils import write_json, write_jsonl

from .metrics import classification_metrics


DEFAULT_METADATA_RELATION_PATTERNS = [
    r"wikidata property",
    r"external id",
    r"\bid\b",
    r"url",
    r"identifier",
    r"metadata",
    r"category",
    r"imported from",
    r"described by source",
    r"reference",
    r"source",
]


def parse_metadata_patterns(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_METADATA_RELATION_PATTERNS)
    return [item.strip() for item in value.split(",") if item.strip()]


def is_metadata_relation(relation_id: str, patterns: list[str] | None = None) -> bool:
    patterns = patterns or DEFAULT_METADATA_RELATION_PATTERNS
    return any(re.search(pattern, relation_id, flags=re.I) for pattern in patterns)


def graph_debug_stats(graph: Any) -> dict[str, int]:
    relation_ids: set[str] = set()
    triples = 0
    for entity_id in graph.entities:
        for relation in graph.iter_relations(entity_id):
            triples += 1
            predicate = str(relation.get("predicate", ""))
            if predicate:
                relation_ids.add(predicate)
    return {
        "entities": len(graph.entities),
        "concepts": len(graph.concepts),
        "relations": len(relation_ids),
        "triples": triples,
    }


def prediction_result(row: dict[str, Any]) -> str:
    expected = bool(row.get("expected_label"))
    predicted = bool(row.get("predicted_satisfies"))
    direction_correct = bool(row.get("predicted_direction_correct", predicted))
    category = row.get("category")
    if expected and predicted and direction_correct:
        return "TRUE POSITIVE"
    if not expected and not predicted:
        return "TRUE NEGATIVE"
    if expected and not predicted:
        return "FALSE NEGATIVE"
    if expected and predicted and not direction_correct:
        return "DIRECTION ERROR"
    if category == "swapped_direction" and predicted:
        return "DIRECTION ERROR"
    return "FALSE POSITIVE"


def prediction_correct(row: dict[str, Any]) -> bool:
    return prediction_result(row) in {"TRUE POSITIVE", "TRUE NEGATIVE"}


def short_diagnosis(row: dict[str, Any]) -> str:
    result = prediction_result(row)
    category = row.get("category")
    if result == "FALSE NEGATIVE":
        return "Rejected a held-out positive; the card may be too narrow or the primitive may be incoherent."
    if result == "FALSE POSITIVE" and category == "hard_negative":
        return "Accepted a confusable hard negative; the card description may be too broad."
    if result == "FALSE POSITIVE" and category == "random_negative":
        return "Accepted an easy random negative; the predicate may be overly broad."
    if result == "DIRECTION ERROR":
        return "Accepted the wrong argument order; direction is not well grounded."
    return "Correct classification."


def pair_text(pair: dict[str, Any]) -> str:
    return f"{pair.get('head', '')} \u2192 {pair.get('tail', '')}"


def type_text(pair: dict[str, Any]) -> str:
    head_types = ", ".join(pair.get("head_types", []) or ["unknown"])
    tail_types = ", ".join(pair.get("tail_types", []) or ["unknown"])
    return f"{head_types} \u2192 {tail_types}"


def prediction_to_debug(row: dict[str, Any]) -> dict[str, Any]:
    pair = row.get("pair", {}) or {}
    return {
        "pair": pair_text(pair),
        "types": type_text(pair),
        "head": pair.get("head", ""),
        "tail": pair.get("tail", ""),
        "head_types": pair.get("head_types", []),
        "tail_types": pair.get("tail_types", []),
        "expected_label": bool(row.get("expected_label")),
        "predicted_label": bool(row.get("predicted_satisfies")),
        "category": row.get("category", ""),
        "confidence": float(row.get("confidence", 0.0) or 0.0),
        "result": prediction_result(row),
        "diagnosis": short_diagnosis(row),
    }


def compute_primitive_metrics(
    cards: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    metadata_patterns: list[str],
) -> list[dict[str, Any]]:
    card_by_key = {
        (card["primitive_id"], card["condition_id"], card["card_variant"]): card
        for card in cards
    }
    rows_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        rows_by_key[(row["primitive_id"], row["condition_id"], row["card_variant"])].append(row)

    metrics_rows: list[dict[str, Any]] = []
    for key in sorted(card_by_key):
        card = card_by_key[key]
        rows = rows_by_key.get(key, [])
        metrics = classification_metrics(rows)
        result_counts = count_results(rows)
        diagnosis = diagnose_card(card, metrics, result_counts, metadata_patterns)
        metrics_rows.append(
            {
                "primitive_id": card["primitive_id"],
                "relation_id": card["relation_id"],
                "direction": card["direction"],
                "condition_id": card["condition_id"],
                "card_variant": card["card_variant"],
                "f1": metrics["f1"],
                "positive_accuracy": metrics["positive_accuracy"],
                "hard_negative_rejection_accuracy": metrics["hard_negative_rejection_accuracy"],
                "random_negative_rejection_accuracy": metrics["random_negative_rejection_accuracy"],
                "swapped_direction_rejection_accuracy": metrics["swapped_direction_rejection_accuracy"],
                "direction_accuracy": metrics["direction_accuracy"],
                "avg_validation_confidence": metrics["avg_validation_confidence"],
                "avg_card_confidence": float(card.get("confidence", 0.0) or 0.0),
                "n_validation_pairs": metrics["n_validation_pairs"],
                "opaque": bool(card.get("opaque_reason")),
                "opaque_reason": card.get("opaque_reason", ""),
                "false_positive_count": result_counts["false_positive_count"],
                "false_negative_count": result_counts["false_negative_count"],
                "swapped_direction_failure_count": result_counts["swapped_direction_failure_count"],
                "high_confidence_wrong_count": result_counts["high_confidence_wrong_count"],
                "diagnosis": diagnosis,
            }
        )

    add_cross_condition_diagnoses(metrics_rows)
    return metrics_rows


def count_results(rows: list[dict[str, Any]]) -> dict[str, int]:
    out = {
        "false_positive_count": 0,
        "false_negative_count": 0,
        "swapped_direction_failure_count": 0,
        "high_confidence_wrong_count": 0,
    }
    for row in rows:
        result = prediction_result(row)
        confidence = float(row.get("confidence", 0.0) or 0.0)
        if result == "FALSE POSITIVE":
            out["false_positive_count"] += 1
        if result == "FALSE NEGATIVE":
            out["false_negative_count"] += 1
        if result == "DIRECTION ERROR" or (row.get("category") == "swapped_direction" and bool(row.get("predicted_satisfies"))):
            out["swapped_direction_failure_count"] += 1
        if result not in {"TRUE POSITIVE", "TRUE NEGATIVE"} and confidence >= 0.75:
            out["high_confidence_wrong_count"] += 1
    return out


def diagnose_card(
    card: dict[str, Any],
    metrics: dict[str, float],
    result_counts: dict[str, int],
    metadata_patterns: list[str],
) -> list[str]:
    labels: list[str] = []
    relation_id = str(card.get("relation_id", ""))
    description = str(card.get("description", ""))
    if is_metadata_relation(relation_id, metadata_patterns):
        labels.append("metadata_relation")
    if card.get("opaque_reason") or metrics.get("positive_accuracy", 0.0) < 0.5:
        labels.append("possibly_opaque")
    if metrics.get("hard_negative_rejection_accuracy", 0.0) < 0.55:
        labels.append("hard_negatives_not_helping")
    if metrics.get("swapped_direction_rejection_accuracy", 0.0) < 0.55:
        labels.append("direction_confusion")
    if (
        result_counts.get("false_positive_count", 0) > 0
        or metrics.get("hard_negative_rejection_accuracy", 0.0) < 0.55
        or metrics.get("random_negative_rejection_accuracy", 0.0) < 0.55
    ):
        labels.append("too_broad_description")
    if len(description.split()) <= 4 and metrics.get("f1", 0.0) < 0.65:
        labels.append("too_broad_description")
    return sorted(set(labels))


def add_cross_condition_diagnoses(metrics_rows: list[dict[str, Any]]) -> None:
    by_primitive_variant: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in metrics_rows:
        by_primitive_variant[(row["primitive_id"], row["card_variant"])][row["condition_id"]] = row
    for condition_rows in by_primitive_variant.values():
        b1 = condition_rows.get("B1")
        b2 = condition_rows.get("B2")
        b3 = condition_rows.get("B3")
        if b1 and b2 and b1["f1"] >= 0.5 and b1["f1"] - b2["f1"] >= 0.25:
            add_label(b1, "entity_name_dependency")
            add_label(b2, "entity_name_dependency")
        if b2 and b3 and b2["f1"] >= 0.5 and b2["f1"] - b3["f1"] >= 0.25:
            add_label(b2, "type_only_failure")
            add_label(b3, "type_only_failure")


def add_label(row: dict[str, Any], label: str) -> None:
    if label not in row["diagnosis"]:
        row["diagnosis"].append(label)
        row["diagnosis"].sort()


def build_examples_summary(
    cards: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    primitive_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics_by_key = {
        (row["primitive_id"], row["condition_id"], row["card_variant"]): row
        for row in primitive_metrics
    }
    prediction_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        prediction_groups[(row["primitive_id"], row["condition_id"], row["card_variant"])].append(row)

    summary: dict[str, Any] = {}
    for card in cards:
        primitive_id = card["primitive_id"]
        key = (primitive_id, card["condition_id"], card["card_variant"])
        card_key = f"{card['condition_id']}/{card['card_variant']}"
        primitive = summary.setdefault(
            primitive_id,
            {
                "primitive_id": primitive_id,
                "relation_id": card["relation_id"],
                "direction": card["direction"],
                "metadata_relation": "metadata_relation" in metrics_by_key.get(key, {}).get("diagnosis", []),
                "cards": {},
            },
        )
        outcomes = [prediction_to_debug(row) for row in prediction_groups.get(key, [])]
        primitive["cards"][card_key] = {
            "condition_id": card["condition_id"],
            "card_variant": card["card_variant"],
            "description": card.get("description", ""),
            "argument_1_role": card.get("generated", {}).get("argument_1_role", ""),
            "argument_2_role": card.get("generated", {}).get("argument_2_role", ""),
            "domain": card.get("generated", {}).get("domain", ""),
            "range": card.get("generated", {}).get("range", ""),
            "domain_types": card.get("domain_types", []),
            "range_types": card.get("range_types", []),
            "confidence": float(card.get("confidence", 0.0) or 0.0),
            "opaque": bool(card.get("opaque_reason")),
            "opaque_reason": card.get("opaque_reason", ""),
            "examples": {
                "positive_train": card.get("positive_examples_train", []),
                "positive_heldout": card.get("positive_examples_heldout", []),
                "hard_negative_train": card.get("hard_negative_examples_train", []),
                "hard_negative_heldout": card.get("hard_negative_examples_heldout", []),
                "random_negative_heldout": card.get("random_negative_examples_heldout", []),
                "swapped_direction_heldout": card.get("swapped_direction_examples_heldout", []),
                "all_validation_outcomes": outcomes,
                "false_positives": [row for row in outcomes if row["result"] == "FALSE POSITIVE"],
                "false_negatives": [row for row in outcomes if row["result"] == "FALSE NEGATIVE"],
                "hard_negative_failures": [
                    row for row in outcomes if row["category"] == "hard_negative" and row["predicted_label"]
                ],
                "swapped_direction_failures": [
                    row for row in outcomes if row["result"] == "DIRECTION ERROR" or (row["category"] == "swapped_direction" and row["predicted_label"])
                ],
                "correct_examples": [
                    row for row in outcomes if row["result"] in {"TRUE POSITIVE", "TRUE NEGATIVE"}
                ],
            },
            "metrics": metrics_by_key.get(key, {}),
            "diagnosis": metrics_by_key.get(key, {}).get("diagnosis", []),
        }
    return summary


def select_wins_losses(summary: dict[str, Any], limit: int = 10) -> dict[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for primitive in summary.values():
        for card_key, card in primitive["cards"].items():
            for outcome in card["examples"]["all_validation_outcomes"]:
                row = {
                    **outcome,
                    "primitive_id": primitive["primitive_id"],
                    "relation_id": primitive["relation_id"],
                    "direction": primitive["direction"],
                    "card_key": card_key,
                    "description": card["description"],
                }
                rows.append(row)
    correct = [row for row in rows if row["result"] in {"TRUE POSITIVE", "TRUE NEGATIVE"}]
    wrong = [row for row in rows if row["result"] not in {"TRUE POSITIVE", "TRUE NEGATIVE"}]
    return {
        "clearest_wins": sorted(correct, key=lambda row: -row["confidence"])[:limit],
        "clearest_failures": sorted(wrong, key=lambda row: -row["confidence"])[:limit],
        "hard_negative_failures": sorted(
            [row for row in wrong if row["category"] == "hard_negative"],
            key=lambda row: -row["confidence"],
        )[:limit],
        "swapped_direction_failures": sorted(
            [row for row in wrong if row["category"] == "swapped_direction" or row["result"] == "DIRECTION ERROR"],
            key=lambda row: -row["confidence"],
        )[:limit],
        "high_confidence_wrong_predictions": sorted(
            [row for row in wrong if row["confidence"] >= 0.75],
            key=lambda row: -row["confidence"],
        )[:limit],
    }


def write_debug_artifacts(
    output_dir: Path,
    cards: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    metrics: dict[str, Any],
    metadata_patterns: list[str],
) -> dict[str, Any]:
    primitive_metrics = compute_primitive_metrics(cards, predictions, metadata_patterns)
    examples_summary = build_examples_summary(cards, predictions, primitive_metrics)
    wins_losses = select_wins_losses(examples_summary)
    write_json(output_dir / "examples_summary.json", examples_summary)
    write_jsonl(output_dir / "primitive_metrics.jsonl", primitive_metrics)
    write_debug_examples_md(output_dir / "debug_examples.md", examples_summary, wins_losses)
    write_debug_report_html(output_dir / "debug_report.html", examples_summary, primitive_metrics, metrics, wins_losses)
    return {
        "examples_summary": examples_summary,
        "primitive_metrics": primitive_metrics,
        "wins_losses": wins_losses,
    }


def write_debug_examples_md(path: Path, summary: dict[str, Any], wins_losses: dict[str, list[dict[str, Any]]]) -> None:
    lines = [
        "# RC-MEX MVP1 Debug Examples",
        "",
        "This report is optimized for inspecting whether relation cards are coherent and where validation fails.",
        "",
        "## Wins And Losses",
        "",
    ]
    for title, rows in wins_losses.items():
        lines.extend([f"### {title.replace('_', ' ').title()}", ""])
        lines.extend(outcome_table(rows))
        lines.append("")

    for primitive_id, primitive in sorted(summary.items()):
        lines.extend(
            [
                f"## Primitive {primitive_id} — relation=\"{escape_md(primitive['relation_id'])}\" — direction={escape_md(primitive['direction'])}",
                "",
            ]
        )
        for card_key, card in sorted(primitive["cards"].items()):
            lines.extend(
                [
                    f"### Card {escape_md(card_key)}",
                    "",
                    "#### Generated Card",
                    "",
                    f"Description: {escape_md(card['description'])}",
                    f"Argument 1 role: {escape_md(card['argument_1_role'])}",
                    f"Argument 2 role: {escape_md(card['argument_2_role'])}",
                    f"Domain/range: {escape_md(card['domain'])} → {escape_md(card['range'])}",
                    f"Confidence: {card['confidence']:.3f}",
                    f"Opaque: {str(card['opaque']).lower()}",
                ]
            )
            if card["opaque_reason"]:
                lines.append(f"Opaque reason: {escape_md(card['opaque_reason'])}")
            if card["diagnosis"]:
                lines.extend(["", "#### Diagnosis", "", ", ".join(f"`{label}`" for label in card["diagnosis"])])
            lines.append("")
            for section, key in [
                ("Positive Train Examples", "positive_train"),
                ("Positive Heldout Examples", "positive_heldout"),
                ("Hard Negative Examples", "hard_negative_heldout"),
                ("Random Negative Examples", "random_negative_heldout"),
                ("Swapped-Direction Examples", "swapped_direction_heldout"),
            ]:
                lines.extend([f"#### {section}", ""])
                lines.extend(example_table(card["examples"].get(key, [])))
                lines.append("")
            for section, key in [
                ("False Positives", "false_positives"),
                ("False Negatives", "false_negatives"),
                ("Hard-Negative Failures", "hard_negative_failures"),
                ("Swapped-Direction Failures", "swapped_direction_failures"),
                ("Correct Examples", "correct_examples"),
            ]:
                lines.extend([f"#### {section}", ""])
                lines.extend(outcome_table(card["examples"].get(key, [])))
                lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def example_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["_No examples._"]
    out = ["| Pair | Types |", "|---|---|"]
    for row in rows:
        out.append(f"| {escape_md(pair_text(row))} | {escape_md(type_text(row))} |")
    return out


def outcome_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["_No examples._"]
    out = [
        "| Primitive | Card | Pair | Types | Expected | Predicted | Category | Confidence | Result | Diagnosis |",
        "|---|---|---|---|---:|---:|---|---:|---|---|",
    ]
    for row in rows:
        out.append(
            "| {primitive} | {card} | {pair} | {types} | {expected} | {predicted} | {category} | {confidence:.3f} | {result} | {diagnosis} |".format(
                primitive=escape_md(str(row.get("primitive_id", ""))),
                card=escape_md(str(row.get("card_key", ""))),
                pair=escape_md(str(row.get("pair", ""))),
                types=escape_md(str(row.get("types", ""))),
                expected=str(bool(row.get("expected_label"))).lower(),
                predicted=str(bool(row.get("predicted_label"))).lower(),
                category=escape_md(str(row.get("category", ""))),
                confidence=float(row.get("confidence", 0.0) or 0.0),
                result=escape_md(str(row.get("result", ""))),
                diagnosis=escape_md(str(row.get("diagnosis", ""))),
            )
        )
    return out


def escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def write_debug_report_html(
    path: Path,
    summary: dict[str, Any],
    primitive_metrics: list[dict[str, Any]],
    metrics: dict[str, Any],
    wins_losses: dict[str, list[dict[str, Any]]],
) -> None:
    dashboard = build_dashboard(summary, primitive_metrics, metrics)
    data = {
        "dashboard": dashboard,
        "metrics": metrics,
        "primitiveMetrics": primitive_metrics,
        "summary": summary,
        "winsLosses": wins_losses,
    }
    json_blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    path.write_text(HTML_TEMPLATE.replace("__REPORT_DATA__", json_blob), encoding="utf-8")


def build_dashboard(
    summary: dict[str, Any],
    primitive_metrics: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    metric_items = sorted(metrics.items(), key=lambda item: float(item[1].get("f1", 0.0)))
    worst = metric_items[0][0] if metric_items else ""
    best = metric_items[-1][0] if metric_items else ""
    return {
        "n_primitives": len(summary),
        "n_cards": len(primitive_metrics),
        "n_validation_pairs": int(sum(float(row.get("n_validation_pairs", 0.0)) for row in primitive_metrics)),
        "best_condition_by_f1": best,
        "worst_condition_by_f1": worst,
        "opaque_rate": average([1.0 if row.get("opaque") else 0.0 for row in primitive_metrics]),
        "hard_negative_rejection": average([float(row.get("hard_negative_rejection_accuracy", 0.0)) for row in primitive_metrics]),
        "swapped_direction_rejection": average([float(row.get("swapped_direction_rejection_accuracy", 0.0)) for row in primitive_metrics]),
    }


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RC-MEX MVP1 Debug Report</title>
  <style>
    :root { --bg:#f7f8fb; --panel:#fff; --ink:#172033; --muted:#647083; --line:#dce2ea; --green:#d8f5df; --yellow:#fff1c7; --red:#ffd8d8; --blue:#e0edff; }
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:var(--bg); }
    header { padding:24px 32px; background:#162033; color:#fff; }
    header h1 { margin:0 0 6px; font-size:28px; }
    main { padding:24px 32px 60px; max-width:1500px; margin:0 auto; }
    h2 { margin-top:32px; border-bottom:1px solid var(--line); padding-bottom:8px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }
    .stat, details, .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; box-shadow:0 1px 2px rgba(0,0,0,.04); }
    .stat .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
    .stat .value { font-size:24px; font-weight:700; margin-top:4px; }
    table { width:100%; border-collapse:collapse; background:var(--panel); }
    th, td { border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; font-size:13px; }
    th { cursor:pointer; position:sticky; top:0; background:#edf1f7; z-index:1; }
    .strong { background:var(--green); }
    .medium { background:var(--yellow); }
    .weak { background:var(--red); }
    .controls { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin:14px 0; }
    input, select, label.filter { padding:8px; border:1px solid var(--line); border-radius:6px; background:#fff; font-size:13px; }
    label.filter { display:flex; gap:8px; align-items:center; }
    summary { cursor:pointer; font-weight:700; }
    .badge { display:inline-block; padding:3px 7px; border-radius:999px; font-size:11px; font-weight:700; margin:2px; }
    .TRUE-POSITIVE, .TRUE-NEGATIVE { background:var(--green); }
    .FALSE-POSITIVE, .FALSE-NEGATIVE, .DIRECTION-ERROR { background:var(--red); }
    .OPAQUE { background:#d9dce3; }
    .diag { background:var(--blue); color:#17335f; }
    .muted { color:var(--muted); }
    .cards { display:grid; gap:12px; }
    .examples { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:12px; }
    pre { white-space:pre-wrap; }
  </style>
</head>
<body>
<header>
  <h1>RC-MEX MVP1 Debug Report</h1>
  <div>Relation-card coherence, validation failures, and schema-grounding diagnostics.</div>
</header>
<main>
  <section>
    <h2>Summary Dashboard</h2>
    <div id="dashboard" class="grid"></div>
  </section>

  <section>
    <h2>Metrics Table</h2>
    <table id="metricsTable"></table>
  </section>

  <section>
    <h2>Primitive Browser</h2>
    <div class="controls">
      <input id="search" placeholder="Search primitive, relation, description">
      <select id="conditionFilter"><option value="">Any condition</option></select>
      <select id="variantFilter"><option value="">Any variant</option></select>
      <select id="directionFilter"><option value="">Any direction</option></select>
      <select id="opaqueFilter"><option value="">Opaque true/false</option><option value="true">Opaque</option><option value="false">Not opaque</option></select>
      <label class="filter"><input id="lowF1Filter" type="checkbox"> low F1</label>
      <label class="filter"><input id="highFPFilter" type="checkbox"> high false positives</label>
      <label class="filter"><input id="highFNFilter" type="checkbox"> high false negatives</label>
      <label class="filter"><input id="metadataFilter" type="checkbox"> metadata-looking relation</label>
    </div>
    <div id="primitiveCards" class="cards"></div>
  </section>

  <section>
    <h2>Most Important Failures</h2>
    <div id="failures" class="cards"></div>
  </section>
</main>

<script>
const DATA = __REPORT_DATA__;

function fmt(v) { return typeof v === 'number' ? v.toFixed(3) : String(v ?? ''); }
function cls(v) { return v >= 0.75 ? 'strong' : (v >= 0.45 ? 'medium' : 'weak'); }
function esc(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function badge(label) { return `<span class="badge ${esc(label).replaceAll(' ', '-')}">${esc(label)}</span>`; }
function diag(labels) { return (labels || []).map(x => `<span class="badge diag">${esc(x)}</span>`).join(' '); }

function renderDashboard() {
  const d = DATA.dashboard;
  const items = [
    ['Number of primitives', d.n_primitives],
    ['Number of cards', d.n_cards],
    ['Number of validation pairs', d.n_validation_pairs],
    ['Best condition by F1', d.best_condition_by_f1],
    ['Worst condition by F1', d.worst_condition_by_f1],
    ['Opaque rate', fmt(d.opaque_rate)],
    ['Hard-negative rejection', fmt(d.hard_negative_rejection)],
    ['Swapped-direction rejection', fmt(d.swapped_direction_rejection)],
  ];
  document.getElementById('dashboard').innerHTML = items.map(([label, value]) =>
    `<div class="stat"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div></div>`
  ).join('');
}

let metricSort = {idx: 2, asc: false};
function renderMetricsTable() {
  const headers = ['condition','variant','F1','positive accuracy','hard-negative rejection','random-negative rejection','swapped-direction rejection','direction accuracy','opaque rate','avg confidence'];
  let rows = Object.entries(DATA.metrics).map(([key, m]) => {
    const [condition, variant] = key.split('/');
    return [condition, variant, m.f1, m.positive_accuracy, m.hard_negative_rejection_accuracy, m.random_negative_rejection_accuracy, m.swapped_direction_rejection_accuracy, m.direction_accuracy, m.opaque_rate, m.avg_card_confidence];
  });
  rows.sort((a,b) => {
    const av = a[metricSort.idx], bv = b[metricSort.idx];
    const cmp = typeof av === 'number' ? av - bv : String(av).localeCompare(String(bv));
    return metricSort.asc ? cmp : -cmp;
  });
  document.getElementById('metricsTable').innerHTML =
    `<thead><tr>${headers.map((h,i)=>`<th onclick="metricSort={idx:${i},asc:metricSort.idx===${i}?!metricSort.asc:true};renderMetricsTable()">${esc(h)}</th>`).join('')}</tr></thead>` +
    `<tbody>${rows.map(row => `<tr>${row.map((v,i)=>`<td class="${i>=2 ? cls(Number(v)||0) : ''}">${esc(fmt(v))}</td>`).join('')}</tr>`).join('')}</tbody>`;
}

function unique(values) { return [...new Set(values.filter(Boolean))].sort(); }
function setupFilters() {
  const metrics = DATA.primitiveMetrics;
  fillSelect('conditionFilter', unique(metrics.map(x => x.condition_id)));
  fillSelect('variantFilter', unique(metrics.map(x => x.card_variant)));
  fillSelect('directionFilter', unique(metrics.map(x => x.direction)));
  for (const id of ['search','conditionFilter','variantFilter','directionFilter','opaqueFilter','lowF1Filter','highFPFilter','highFNFilter','metadataFilter']) {
    document.getElementById(id).addEventListener('input', renderPrimitiveCards);
    document.getElementById(id).addEventListener('change', renderPrimitiveCards);
  }
}
function fillSelect(id, values) {
  const el = document.getElementById(id);
  values.forEach(v => el.insertAdjacentHTML('beforeend', `<option value="${esc(v)}">${esc(v)}</option>`));
}

function renderPrimitiveCards() {
  const query = document.getElementById('search').value.toLowerCase();
  const condition = document.getElementById('conditionFilter').value;
  const variant = document.getElementById('variantFilter').value;
  const direction = document.getElementById('directionFilter').value;
  const opaque = document.getElementById('opaqueFilter').value;
  const lowF1 = document.getElementById('lowF1Filter').checked;
  const highFP = document.getElementById('highFPFilter').checked;
  const highFN = document.getElementById('highFNFilter').checked;
  const metadata = document.getElementById('metadataFilter').checked;
  const cards = [];
  for (const primitive of Object.values(DATA.summary)) {
    for (const [cardKey, card] of Object.entries(primitive.cards)) {
      const m = card.metrics || {};
      const text = `${primitive.primitive_id} ${primitive.relation_id} ${primitive.direction} ${card.description}`.toLowerCase();
      if (query && !text.includes(query)) continue;
      if (condition && card.condition_id !== condition) continue;
      if (variant && card.card_variant !== variant) continue;
      if (direction && primitive.direction !== direction) continue;
      if (opaque && String(card.opaque) !== opaque) continue;
      if (lowF1 && !(m.f1 < 0.5)) continue;
      if (highFP && !(m.false_positive_count > 0)) continue;
      if (highFN && !(m.false_negative_count > 0)) continue;
      if (metadata && !(card.diagnosis || []).includes('metadata_relation')) continue;
      cards.push({primitive, cardKey, card, m});
    }
  }
  document.getElementById('primitiveCards').innerHTML = cards.map(renderCard).join('') || '<div class="panel">No cards match the current filters.</div>';
}

function renderCard({primitive, cardKey, card, m}) {
  const outcomes = card.examples.all_validation_outcomes || [];
  const show = list => (list || []).slice(0, 8).map(x => `<li>${badge(x.result)} ${esc(x.pair)} <span class="muted">${esc(x.types)}</span> conf=${fmt(x.confidence)}</li>`).join('') || '<li class="muted">None</li>';
  const examples = name => (card.examples[name] || []).slice(0, 6).map(x => `<li>${esc(x.head)} → ${esc(x.tail)} <span class="muted">${esc((x.head_types||[]).join(', '))} → ${esc((x.tail_types||[]).join(', '))}</span></li>`).join('') || '<li class="muted">None</li>';
  return `<details>
    <summary>${esc(primitive.primitive_id)} | ${esc(primitive.relation_id)} | ${esc(primitive.direction)} | ${esc(cardKey)} ${card.opaque ? badge('OPAQUE') : ''}</summary>
    <p><b>Description:</b> ${esc(card.description)}</p>
    <p><b>Roles:</b> ${esc(card.argument_1_role)} → ${esc(card.argument_2_role)} | <b>Confidence:</b> ${fmt(card.confidence)}</p>
    <p>${diag(card.diagnosis)}</p>
    <table><tr><th>F1</th><th>Positive acc.</th><th>Hard-neg reject</th><th>Random-neg reject</th><th>Swapped reject</th><th>FP</th><th>FN</th></tr>
      <tr><td class="${cls(m.f1||0)}">${fmt(m.f1||0)}</td><td class="${cls(m.positive_accuracy||0)}">${fmt(m.positive_accuracy||0)}</td><td class="${cls(m.hard_negative_rejection_accuracy||0)}">${fmt(m.hard_negative_rejection_accuracy||0)}</td><td class="${cls(m.random_negative_rejection_accuracy||0)}">${fmt(m.random_negative_rejection_accuracy||0)}</td><td class="${cls(m.swapped_direction_rejection_accuracy||0)}">${fmt(m.swapped_direction_rejection_accuracy||0)}</td><td>${esc(m.false_positive_count||0)}</td><td>${esc(m.false_negative_count||0)}</td></tr>
    </table>
    <div class="examples">
      <div><h3>Positive examples</h3><ul>${examples('positive_heldout')}</ul></div>
      <div><h3>Hard negatives</h3><ul>${examples('hard_negative_heldout')}</ul></div>
      <div><h3>Random negatives</h3><ul>${examples('random_negative_heldout')}</ul></div>
      <div><h3>Swapped examples</h3><ul>${examples('swapped_direction_heldout')}</ul></div>
      <div><h3>False positives</h3><ul>${show(card.examples.false_positives)}</ul></div>
      <div><h3>False negatives</h3><ul>${show(card.examples.false_negatives)}</ul></div>
    </div>
  </details>`;
}

function renderFailures() {
  const groups = [
    ['False negatives on positives', DATA.winsLosses.clearest_failures.filter(x => x.result === 'FALSE NEGATIVE')],
    ['False positives on hard negatives', DATA.winsLosses.hard_negative_failures],
    ['Swapped-direction failures', DATA.winsLosses.swapped_direction_failures],
    ['High-confidence wrong predictions', DATA.winsLosses.high_confidence_wrong_predictions],
    ['Incoherent/metadata primitives', DATA.primitiveMetrics.filter(x => (x.diagnosis||[]).includes('metadata_relation') || (x.diagnosis||[]).includes('possibly_opaque'))],
  ];
  document.getElementById('failures').innerHTML = groups.map(([title, rows]) => `<div class="panel"><h3>${esc(title)}</h3>${renderFailureRows(rows)}</div>`).join('');
}
function renderFailureRows(rows) {
  if (!rows || !rows.length) return '<p class="muted">None.</p>';
  return `<ul>${rows.slice(0,10).map(row => {
    if (row.pair) return `<li>${badge(row.result)} <b>${esc(row.primitive_id)}</b> ${esc(row.card_key)}: ${esc(row.pair)} <span class="muted">${esc(row.description)}</span> conf=${fmt(row.confidence)}</li>`;
    return `<li><b>${esc(row.primitive_id)}</b> ${esc(row.condition_id)}/${esc(row.card_variant)} ${diag(row.diagnosis)} F1=${fmt(row.f1)}</li>`;
  }).join('')}</ul>`;
}

renderDashboard();
renderMetricsTable();
setupFilters();
renderPrimitiveCards();
renderFailures();
</script>
</body>
</html>
"""
