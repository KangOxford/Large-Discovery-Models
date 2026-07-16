"""Tests for the ``bo-*-ldm`` method names in plot_search_results.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List

import plot_search_results


# ---------------------------------------------------------------------------
# Color / label dict
# ---------------------------------------------------------------------------


def test_method_colors_includes_ldm_variants() -> None:
    assert "bo-tanimoto-ldm" in plot_search_results.METHOD_COLORS
    assert "bo-strkernel-ldm" in plot_search_results.METHOD_COLORS


def test_method_labels_includes_ldm_variants() -> None:
    assert "bo-tanimoto-ldm" in plot_search_results.METHOD_LABELS
    assert "bo-strkernel-ldm" in plot_search_results.METHOD_LABELS
    # Labels should mention LLM / LDM.
    assert "LDM" in plot_search_results.METHOD_LABELS["bo-tanimoto-ldm"]
    assert "LDM" in plot_search_results.METHOD_LABELS["bo-strkernel-ldm"]


# ---------------------------------------------------------------------------
# End-to-end: plot a fake run with the new methods
# ---------------------------------------------------------------------------


def _write_fake_json(path: Path, method: str, n_eval: int = 5) -> None:
    payload = {
        "config": {
            "method": method,
            "seed": 0,
            "n_objectives": 1,
            "minimize": True,
        },
        "history": [
            {"index": i, "smiles": f"CC{'C' * i}", "score": -float(i + 1)}
            for i in range(n_eval)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_plot_summary_with_ldm_method(tmp_path: Path) -> None:
    """The plotter should not error when given a ``bo-*-ldm`` method."""
    _write_fake_json(tmp_path / "bo-tanimoto-ldm_seed=0.json",
                     "bo-tanimoto-ldm")
    _write_fake_json(tmp_path / "random_seed=0.json", "random")

    results, meta = plot_search_results.load_inputs(tmp_path, methods_filter=None)
    assert "bo-tanimoto-ldm" in results
    assert "random" in results

    summary = plot_search_results.aggregate(
        results, meta, num_evaluations=5,
    )
    assert "bo-tanimoto-ldm" in summary
    assert "random" in summary

    # Render to a file. We don't check the contents (matplotlib-free
    # in the test env) — just verify no exception.
    out_path = tmp_path / "summary.png"
    plot_search_results.plot_summary(summary, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_methods_filter_with_ldm(tmp_path: Path) -> None:
    """``--methods bo-tanimoto-ldm`` should filter correctly."""
    _write_fake_json(tmp_path / "bo-tanimoto-ldm_seed=0.json",
                     "bo-tanimoto-ldm")
    _write_fake_json(tmp_path / "random_seed=0.json", "random")

    results, _ = plot_search_results.load_inputs(
        tmp_path, methods_filter={"bo-tanimoto-ldm"},
    )
    assert "bo-tanimoto-ldm" in results
    assert "random" not in results
