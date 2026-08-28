#!/usr/bin/env python3
"""Build a portable traceability manifest for the ICLR paper artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "paper" / "iclr2027" / "artifact_manifest.json"
OUT_MD = ROOT / "paper" / "iclr2027" / "artifact_manifest.md"


def entry(label, paper_object, claim, results, analyses, protocols, configs=()):
    return {
        "label": label,
        "paper_object": paper_object,
        "claim": claim,
        "result_files": list(results),
        "analysis_or_rendering": list(analyses),
        "protocols": list(protocols),
        "configs_or_manifests": list(configs),
    }


OBJECTS = [
    entry(
        "fig:chain",
        "Figure 1: BBH surrogate chain",
        "DSMC improves measured geometry and targeting loss while same-item and held-out exact match decline relative to Random.",
        (
            "experiments/less_aligned/results_summary/bbh_forensic_geometry.json",
            "experiments/less_aligned/results_summary/bbh_forensic_query_loss.json",
            "experiments/less_aligned/results_summary/bbh_forensic_query_cot.json",
            "experiments/less_aligned/results_summary/llama32_results.json",
            "experiments/less_aligned/results_summary/llama32_diagnostics.json",
        ),
        ("paper/iclr2027/scripts/make_surrogate_chain.py",),
        (
            "experiments/less_aligned/prereg_bbh_external.md",
            "experiments/less_aligned/prereg_second_model.md",
        ),
        (
            "experiments/less_aligned/bbh_external_launch_manifest.json",
            "experiments/less_aligned/bbh_eval_pin_manifest.json",
        ),
    ),
    entry(
        "tab:attribution",
        "Table 1: Llama-2 MMLU representation by selector attribution",
        "The second-moment representation and MMD-style selector contribute separately in the controlled 2x2 comparison.",
        (
            "experiments/less_aligned/results_summary/attribution_2x2_results.csv",
            "experiments/less_aligned/results_summary/attribution_2x2_summary.md",
        ),
        ("experiments/less_aligned/run_attribution_2x2.sh",),
        ("experiments/less_aligned/target_draw_protocol.md",),
    ),
    entry(
        "tab:mmlu_crossstack",
        "Table 2: cross-stack MMLU comparison",
        "The Llama-2 method-level advantage over round-robin selectors does not transfer to Llama-3.2.",
        (
            "experiments/less_aligned/results_summary/full5draw_5pct_aggregate.csv",
            "experiments/less_aligned/results_summary/full5draw_5pct_results.md",
            "experiments/less_aligned/results_summary/llama32_mmlu5pct_results.json",
        ),
        ("scripts/analyse_llama32_mmlu5pct.py",),
        (
            "experiments/less_aligned/target_draw_protocol.md",
            "experiments/less_aligned/prereg_llama32_mmlu5pct.md",
        ),
        (
            "experiments/less_aligned/full5draw_launch_manifest.json",
            "experiments/less_aligned/llama32_mmlu5pct_run_state.json",
        ),
    ),
    entry(
        "tab:bbh",
        "Table 3: held-out BBH exact match",
        "DSMC underperforms Random in every query/selection draw on both model stacks.",
        (
            "experiments/less_aligned/results_summary/bbh_external_results.md",
            "experiments/less_aligned/results_summary/llama32_results.json",
        ),
        (
            "scripts/run_bbh_full.py",
            "scripts/analyse_llama32_results.py",
        ),
        (
            "experiments/less_aligned/prereg_bbh_external.md",
            "experiments/less_aligned/prereg_second_model.md",
        ),
        (
            "experiments/less_aligned/bbh_full_run_state.json",
            "experiments/less_aligned/llama32_full_run_state.json",
        ),
    ),
    entry(
        "tab:mmlu_main",
        "Appendix: Llama-2 MMLU budgets",
        "At 1%, DSMC and Random both exceed the no-SFT mean while DSMC is geometrically closer and lower utility.",
        (
            "experiments/less_aligned/results_summary/full5draw_5pct_aggregate.csv",
            "experiments/less_aligned/results_summary/full1pct_aggregate.csv",
            "experiments/less_aligned/results_summary/full1pct_budget_interaction_results.md",
            "experiments/less_aligned/results_summary/base_model_reference.json",
            "experiments/less_aligned/results_summary/forensic_mechanism_analysis.md",
        ),
        ("scripts/forensic_robustness.py",),
        (
            "experiments/less_aligned/target_draw_protocol.md",
            "experiments/less_aligned/prereg_1pct_equalstep.md",
        ),
        (
            "experiments/less_aligned/full5draw_launch_manifest.json",
            "experiments/less_aligned/full1pct_launch_manifest.json",
            "experiments/less_aligned/targetdraw_10draw_master_manifest.json",
        ),
    ),
    entry(
        "tab:signed_d1",
        "Appendix: signed first-moment BBH discrepancy",
        "First-RR is closer than Random under signed D1 in every draw on both stacks but has lower downstream utility.",
        ("experiments/less_aligned/results_summary/bbh_signed_d1_taskwise.json",),
        ("scripts/analyse_bbh_signed_and_taskwise.py",),
        (
            "experiments/less_aligned/prereg_bbh_external.md",
            "experiments/less_aligned/prereg_second_model.md",
        ),
    ),
    entry(
        "tab:taskwise",
        "Appendix: task-level BBH heterogeneity",
        "The Llama-2 gap is broad across subtasks; the smaller Llama-3.2 gap is more heterogeneous.",
        ("experiments/less_aligned/results_summary/bbh_signed_d1_taskwise.json",),
        ("scripts/analyse_bbh_signed_and_taskwise.py",),
        (
            "experiments/less_aligned/prereg_bbh_external.md",
            "experiments/less_aligned/prereg_second_model.md",
        ),
    ),
    entry(
        "tab:proxy_geometry",
        "Appendix: query-disjoint proxy geometry",
        "Frozen DSMC remains closer than Random under D2 in every draw on both stacks on a proxy set disjoint from all selection queries.",
        (
            "experiments/less_aligned/results_summary/bbh_proxy_geometry.json",
            "data/bbh_external/bbh_proxy_test_manifest.json",
        ),
        (
            "scripts/build_bbh_proxy_test.py",
            "scripts/analyse_bbh_proxy_geometry.py",
        ),
        ("experiments/less_aligned/prereg_bbh_proxy_geometry.md",),
        (
            "experiments/less_aligned/configs/draws/select_proxy_l2.yaml",
            "experiments/less_aligned/configs/draws/select_proxy_l32.yaml",
            "src/dataflex/configs/components_proxy_geometry.yaml",
            "data/dataset_info.json",
        ),
    ),
    entry(
        "tab:evalctx",
        "Appendix: evaluation-matched target-geometry sensitivity",
        "Frozen DSMC remains closer than Random under evaluation-matched D2 in every draw on both stacks.",
        ("experiments/less_aligned/results_summary/evalctx_d2_sensitivity.json",),
        ("scripts/forensic_evalctx_d2_sensitivity.py",),
        (
            "experiments/less_aligned/prereg_bbh_external.md",
            "experiments/less_aligned/prereg_second_model.md",
        ),
    ),
    entry(
        "tab:parser",
        "Appendix: BBH parser-validity audit",
        "The DSMC–Random utility gap persists conditional on parser-valid outputs and under conservative recovery.",
        ("experiments/less_aligned/results_summary/bbh_parser_validity.json",),
        ("scripts/analyse_bbh_parser_validity.py",),
        ("experiments/less_aligned/prereg_bbh_parser_audit.md",),
        (
            "experiments/less_aligned/bbh_eval_pin_manifest.json",
            "experiments/less_aligned/bbh_full_run_state.json",
            "experiments/less_aligned/llama32_full_run_state.json",
        ),
    ),
    entry(
        "tab:cost",
        "Appendix: compute and selection overhead",
        "Reports measured training minutes and coarse stage-level accounting without treating them as microbenchmarks.",
        (
            "experiments/less_aligned/bbh_full_run_state.json",
            "experiments/less_aligned/llama32_full_run_state.json",
            "experiments/less_aligned/llama32_mmlu5pct_run_state.json",
        ),
        (),
        (
            "experiments/less_aligned/prereg_bbh_external.md",
            "experiments/less_aligned/prereg_second_model.md",
            "experiments/less_aligned/prereg_llama32_mmlu5pct.md",
        ),
    ),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def enrich_file(path_str: str) -> dict[str, object]:
    path = ROOT / path_str
    if not path.is_file():
        raise FileNotFoundError(path_str)
    return {
        "path": path_str,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def build() -> dict[str, object]:
    enriched = []
    for obj in OBJECTS:
        row = dict(obj)
        for key in (
            "result_files",
            "analysis_or_rendering",
            "protocols",
            "configs_or_manifests",
        ):
            row[key] = [enrich_file(path) for path in row[key]]
        enriched.append(row)
    return {
        "schema_version": 1,
        "paper_source": enrich_file("paper/iclr2027/main.tex"),
        "scope": (
            "Portable mapping from each major paper figure/table to committed "
            "result artifacts, analysis code, frozen protocols, and run metadata."
        ),
        "notes": [
            "Paths are repository-relative; external model, gradient, adapter, and generation caches are not included.",
            "Post-hoc analyses are labelled as such in the paper and their protocol files.",
            "Hashes identify the exact repository files used when this manifest was generated.",
        ],
        "objects": enriched,
    }


def render_markdown(manifest: dict[str, object]) -> str:
    lines = [
        "# Paper artifact manifest",
        "",
        manifest["scope"],
        "",
        "| Paper object | Main result artifact(s) | Analysis / rendering | Protocol |",
        "|---|---|---|---|",
    ]
    for obj in manifest["objects"]:
        paths = lambda key: "<br>".join(f"`{x['path']}`" for x in obj[key]) or "—"
        lines.append(
            f"| `{obj['label']}` — {obj['paper_object']} "
            f"| {paths('result_files')} "
            f"| {paths('analysis_or_rendering')} "
            f"| {paths('protocols')} |"
        )
    lines += [
        "",
        "The JSON companion records SHA-256 and byte size for every listed file.",
        "All paths are repository-relative.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    manifest = build()
    OUT_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    OUT_MD.write_text(render_markdown(manifest))
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
