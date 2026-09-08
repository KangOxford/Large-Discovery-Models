"""Real-evaluation plumbing: materialisation, parsing, classification, cache.

The trainer itself needs a GPU and 300s, so these tests substitute a fake
``run_command`` that prints trainer-shaped output. Everything except the
training step is exercised for real, including patching the actual
``scripts/train.py`` in this checkout.
"""

from __future__ import annotations

import json
import os
import re

import pytest

from tasks.nanogpt.core import rl_eval as E, rl_knobs as K

REAL_METRIC_BLOCK = """\
step 00123 (100.0%) | loss: 3.1 | mfu: 39.5%
---
val_bpb:          0.995752
training_seconds: 300.1
total_seconds:    328.4
peak_vram_mb:     45210.5
mfu_percent:      39.52
total_tokens_M:   1234.5
num_steps:        2355
num_params_M:     33.6
depth:            8
"""


def _fake_command(script: str) -> list[str]:
    return ["python3", "-c", script]


# --- materialisation ------------------------------------------------------


def test_patch_rewrites_the_real_train_py():
    scripts = os.path.join(os.path.dirname(os.path.dirname(K.__file__)), "scripts")
    with open(os.path.join(scripts, "train.py")) as handle:
        source = handle.read()
    config = K.validate({**K.DEFAULTS, "DEPTH": 12, "MATRIX_LR": 0.0731})
    patched = E.patch_train_source(
        source, {name: config[name] for name in K.KNOB_ORDER}
    )
    assert re.search(r"^DEPTH = 12\b", patched, re.M)
    assert re.search(r"^MATRIX_LR = 0\.0731\b", patched, re.M)
    # The neighbouring tuple on line 444 is not a knob and must survive.
    assert re.search(r"^ADAM_BETAS = \(0\.8, 0\.95\)", patched, re.M)


def test_patch_refuses_to_silently_skip_a_missing_knob():
    # A moved template must fail loudly: substituting nothing would measure a
    # different config from the one the reward is attributed to.
    with pytest.raises(KeyError, match="NOT_A_KNOB"):
        E.patch_train_source("DEPTH = 8\n", {"DEPTH": 9, "NOT_A_KNOB": 1})


def test_materialise_run_writes_train_and_prepare(tmp_path):
    config = K.validate(K.DEFAULTS)
    E.materialise_run(config, str(tmp_path), time_budget=150)
    train = (tmp_path / "train.py").read_text()
    prepare = (tmp_path / "prepare.py").read_text()
    assert re.search(r"^TIME_BUDGET = 150\b", prepare, re.M)
    # prepare.py must be a real copy, not a symlink, or the budget cannot vary.
    assert not (tmp_path / "prepare.py").is_symlink()
    # ...but the dataset stays shared via the env var, not duplicated.
    assert "AUTORESEARCH_CACHE_DIR" in prepare
    assert "from prepare import" in train


# --- parsing and classification -------------------------------------------


def test_parse_metrics_reads_the_trainer_block():
    metrics = E.parse_metrics(REAL_METRIC_BLOCK)
    assert metrics["val_bpb"] == pytest.approx(0.995752)
    assert metrics["num_steps"] == 2355
    assert metrics["peak_vram_mb"] == pytest.approx(45210.5)
    assert set(metrics) == set(E.METRIC_KEYS)


def test_parse_metrics_ignores_the_progress_line_mfu():
    # The \r progress line also contains "mfu: 39.5%"; only the final
    # "mfu_percent:" line is a result.
    metrics = E.parse_metrics(REAL_METRIC_BLOCK)
    assert metrics["mfu_percent"] == pytest.approx(39.52)


@pytest.mark.parametrize(
    ("log", "expected"),
    [
        ("Traceback\n  assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0\nAssertionError",
         "rejected_by_trainer"),
        ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 576.00 MiB",
         "oom"),
        ("step 00003 | loss: nan\nFAIL\n", "loss_diverged"),
        ("Traceback\n  assert x.ndim == 4\nAssertionError", "rejected_by_trainer"),
        ("ImportError: no module named torch", "crashed"),
    ],
)
def test_classify_failure(log, expected):
    kind, reason = E.classify_failure(log, 1)
    assert kind == expected
    assert reason


def test_oom_wins_over_generic_error_text():
    # An OOM traceback also contains "Error"; ordering must not misclassify it.
    log = "RuntimeError\ntorch.OutOfMemoryError: CUDA out of memory\n"
    assert E.classify_failure(log, 1)[0] == "oom"


# --- cache -----------------------------------------------------------------


def test_cache_round_trips_and_dedups(tmp_path):
    cache = E.ResultCache(str(tmp_path / "c.jsonl"))
    outcome = E.RunOutcome(
        canonical_key="k1", config={"DEPTH": 8}, time_budget=300.0,
        ok=True, metrics={"val_bpb": 0.99},
    )
    assert cache.get("k1") is None
    cache.put(outcome)
    hit = cache.get("k1")
    assert hit is not None and hit.cached and hit.val_bpb == pytest.approx(0.99)
    assert [o.canonical_key for o in cache.successes()] == ["k1"]


def test_cache_remembers_final_failures_but_retries_timeouts(tmp_path):
    cache = E.ResultCache(str(tmp_path / "c.jsonl"))
    cache.put(E.RunOutcome(canonical_key="oom", config={}, time_budget=300.0,
                           ok=False, failure_kind="oom"))
    cache.put(E.RunOutcome(canonical_key="slow", config={}, time_budget=300.0,
                           ok=False, failure_kind="timed_out"))
    # An OOM will OOM again; paying 20s of GPU to rediscover that is waste.
    assert cache.get("oom") is not None
    # A timeout can be a transient node problem, so it is not final.
    assert cache.get("slow") is None


# --- GPU pool --------------------------------------------------------------


def test_gpu_pool_is_exclusive_and_releases(tmp_path):
    pool = E.GpuPool(["0", "1"], str(tmp_path / "locks"))
    with pool.claim() as first:
        with pool.claim() as second:
            assert {first, second} == {"0", "1"}
            with pytest.raises(TimeoutError):
                with pool.claim(timeout=0.0, poll=0.0):
                    pass
    # both released
    with pool.claim() as again:
        assert again in {"0", "1"}


def test_gpu_pool_defaults_to_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOGPT_EVAL_GPUS", "4, 5,6")
    assert E.GpuPool.from_env(str(tmp_path)).devices == ["4", "5", "6"]
    monkeypatch.delenv("NANOGPT_EVAL_GPUS")
    assert E.GpuPool.from_env(str(tmp_path)).devices == ["0"]


# --- evaluator integration (fake trainer) ---------------------------------


def _evaluator(tmp_path, script, **kwargs):
    return E.RealNanogptEvaluator(
        output_dir=str(tmp_path),
        run_command=_fake_command(script),
        eval_gpus="0",
        **kwargs,
    )


def test_evaluate_config_success_and_cache_hit(tmp_path):
    script = f"print({REAL_METRIC_BLOCK!r})"
    runner = _evaluator(tmp_path, script)
    first = runner.evaluate_config(K.DEFAULTS)
    assert first.ok and not first.cached
    assert first.val_bpb == pytest.approx(0.995752)
    assert first.gpu == "0"
    # The run directory keeps what was actually executed.
    assert os.path.exists(os.path.join(first.run_dir, "train.py"))
    assert json.load(open(os.path.join(first.run_dir, "config.json")))["config"]

    second = runner.evaluate_config(K.DEFAULTS)
    assert second.cached and second.val_bpb == first.val_bpb
    assert runner.measured_history() and runner.measured_history()[0][1] == pytest.approx(
        0.995752
    )


def test_illegal_config_is_rejected_without_launching_the_trainer(tmp_path):
    # DEVICE_BATCH_SIZE=96 must never reach a GPU.
    runner = _evaluator(tmp_path, "raise SystemExit('should not run')")
    outcome = runner.evaluate_config({**K.DEFAULTS, "DEVICE_BATCH_SIZE": 96})
    assert not outcome.ok
    assert outcome.failure_kind == "rejected_by_trainer"
    assert outcome.run_dir == ""


def test_oom_is_classified_and_cached(tmp_path):
    script = (
        "import sys; sys.stdout.write('torch.OutOfMemoryError: CUDA out of memory\\n');"
        " sys.exit(1)"
    )
    runner = _evaluator(tmp_path, script)
    outcome = runner.evaluate_config(K.DEFAULTS)
    assert not outcome.ok and outcome.failure_kind == "oom"
    assert runner.evaluate_config(K.DEFAULTS).cached


def test_zero_exit_without_metrics_is_a_failure(tmp_path):
    # Silently returning "no score" would otherwise look like a successful
    # evaluation with a missing objective.
    runner = _evaluator(tmp_path, "print('trained fine, forgot to report')")
    outcome = runner.evaluate_config(K.DEFAULTS)
    assert not outcome.ok and outcome.failure_kind == "no_metrics"


def test_timeout_is_reported_and_not_cached(tmp_path):
    runner = _evaluator(
        tmp_path, "import time; time.sleep(30)", time_budget=0.1, timeout_slack=0.4
    )
    outcome = runner.evaluate_config(K.DEFAULTS)
    assert not outcome.ok and outcome.failure_kind == "timed_out"
    assert runner.cache.get(outcome.canonical_key) is None


def test_run_dir_is_stable_across_processes(tmp_path):
    # str.__hash__ is salted per process; a run must be findable from its key.
    script = f"print({REAL_METRIC_BLOCK!r})"
    first = _evaluator(tmp_path / "a", script).evaluate_config(K.DEFAULTS)
    second = _evaluator(tmp_path / "b", script).evaluate_config(K.DEFAULTS)
    assert os.path.basename(first.run_dir) == os.path.basename(second.run_dir)
