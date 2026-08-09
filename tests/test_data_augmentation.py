from __future__ import annotations

import json

from ldm_tts.data import (
    ExpertJustificationPipeline,
    JustificationRequest,
    make_complete_design_ir,
)


class SequenceExpert:
    def __init__(self, *responses, identity=None):
        self.responses = list(responses)
        self.requests: list[JustificationRequest] = []
        if identity is not None:
            self.cache_identity = identity

    def justify(self, request: JustificationRequest) -> str:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _ir(*, reasoning=None, reasoning_available=True, design="CCN"):
    return make_complete_design_ir(
        task_id="small_molecule",
        domain="molecule",
        task_description="Propose molecules for a two-objective search.",
        objectives=[
            {
                "name": "vina",
                "direction": "minimize",
                "description": "Docking score; lower is better.",
            }
        ],
        design_space_description="Single-component organic SMILES.",
        observations=[
            {"design": "CCO", "results": {"vina": -2.4}, "roles": ["recent"]}
        ],
        candidates=[{"design": design, "rationale": "nearby candidate"}],
        request_description="Generate one valid, novel SMILES without scores.",
        reasoning_available=reasoning_available,
        reasoning=reasoning,
    )


def _write_array(path, rows):
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_ir_augmentation_populates_action_reasoning_and_renders_sft(tmp_path):
    source = tmp_path / "ir.json"
    output = tmp_path / "ir_augmented.jsonl"
    sft_output = tmp_path / "sft.jsonl"
    _write_array(source, [_ir()])
    expert = SequenceExpert("<think>I would favor the nearby amine.</think>")

    report = ExpertJustificationPipeline(expert, workers=1).run(
        source, output, sft_output_path=sft_output
    )

    augmented = _read_jsonl(output)[0]
    rendered = _read_jsonl(sft_output)[0]
    assert augmented["action"]["reasoning"] == "I would favor the nearby amine."
    assert augmented["collection"]["augmentation"] == {
        "kind": "expert_justification",
        "expert": "SequenceExpert",
    }
    assert json.loads(rendered["output"])["reasoning"] == (
        "I would favor the nearby amine."
    )
    assert report.generated == 1
    assert report.failed == 0
    assert "private" not in expert.requests[0].instruction


def test_alpaca_augmentation_handles_structured_and_plain_outputs(tmp_path):
    source = tmp_path / "alpaca.json"
    output = tmp_path / "alpaca_augmented.jsonl"
    _write_array(
        source,
        [
            {"instruction": "Compute 2 + 2.", "input": "", "output": "4"},
            {
                "instruction": "Choose an action.",
                "input": "state",
                "output": json.dumps(
                    {"type": "propose", "reasoning": None, "payload": {"x": 1}}
                ),
            },
        ],
    )
    expert = SequenceExpert("I add the two terms.", "I use the visible state.")

    report = ExpertJustificationPipeline(expert, workers=1).run(source, output)

    rows = _read_jsonl(output)
    assert rows[0]["output"] == "<think>\nI add the two terms.\n</think>\n\n4"
    assert json.loads(rows[1]["output"])["reasoning"] == "I use the visible state."
    assert "## Additional input\nstate" in expert.requests[1].instruction
    assert report.generated == 2


def test_existing_and_reasoning_unavailable_records_are_skipped(tmp_path):
    source = tmp_path / "ir.json"
    output = tmp_path / "ir_augmented.jsonl"
    _write_array(
        source,
        [
            _ir(reasoning="Already justified."),
            _ir(reasoning_available=False, design="CCC"),
        ],
    )
    expert = SequenceExpert()

    report = ExpertJustificationPipeline(expert).run(source, output)

    assert report.skipped_existing == 1
    assert report.skipped_unavailable == 1
    assert report.generated == 0
    assert expert.requests == []
    assert _read_jsonl(output)[1]["action"]["reasoning"] is None


def test_checkpoint_resumes_successes_and_retries_only_failed_rows(tmp_path):
    source = tmp_path / "ir.json"
    first_output = tmp_path / "first.jsonl"
    final_output = tmp_path / "final.jsonl"
    checkpoint = tmp_path / "checkpoint.jsonl"
    _write_array(source, [_ir(design="CCN"), _ir(design="CCC")])
    first_expert = SequenceExpert("First rationale.", RuntimeError("temporary"))

    first_report = ExpertJustificationPipeline(
        first_expert, workers=1, max_retries=1
    ).run(source, first_output, checkpoint_path=checkpoint)

    second_expert = SequenceExpert("Second rationale.")
    second_report = ExpertJustificationPipeline(
        second_expert, workers=1, max_retries=1
    ).run(source, final_output, checkpoint_path=checkpoint)

    rows = _read_jsonl(final_output)
    assert first_report.generated == 1
    assert first_report.failed_indices == (1,)
    assert second_report.resumed == 1
    assert second_report.generated == 1
    assert len(second_expert.requests) == 1
    assert rows[0]["action"]["reasoning"] == "First rationale."
    assert rows[1]["action"]["reasoning"] == "Second rationale."


def test_model_errors_are_retried_before_the_record_is_failed(tmp_path):
    source = tmp_path / "alpaca.json"
    output = tmp_path / "alpaca_augmented.jsonl"
    _write_array(source, [{"instruction": "Question", "input": "", "output": "A"}])
    expert = SequenceExpert(RuntimeError("rate limited"), "Recovered rationale.")

    report = ExpertJustificationPipeline(
        expert, workers=1, max_retries=2, retry_backoff_seconds=0
    ).run(source, output)

    assert len(expert.requests) == 2
    assert report.generated == 1
    assert report.failed == 0


def test_checkpoint_is_scoped_to_the_expert_configuration(tmp_path):
    source = tmp_path / "alpaca.json"
    checkpoint = tmp_path / "checkpoint.jsonl"
    _write_array(source, [{"instruction": "Question", "input": "", "output": "A"}])

    ExpertJustificationPipeline(
        SequenceExpert("Model A rationale.", identity="model-a"), workers=1
    ).run(source, tmp_path / "a.jsonl", checkpoint_path=checkpoint)
    expert_b = SequenceExpert("Model B rationale.", identity="model-b")
    report = ExpertJustificationPipeline(expert_b, workers=1).run(
        source, tmp_path / "b.jsonl", checkpoint_path=checkpoint
    )

    assert report.resumed == 0
    assert report.generated == 1
    assert len(expert_b.requests) == 1
    assert "Model B rationale." in _read_jsonl(tmp_path / "b.jsonl")[0]["output"]
