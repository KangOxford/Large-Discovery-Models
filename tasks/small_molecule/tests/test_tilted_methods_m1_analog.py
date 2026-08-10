import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tasks.small_molecule.core.ldm_tilted_case2.config import TiltedLDMCase2Config
from tasks.small_molecule.core.ldm_tilted_case2.methods.llm_seed_analog import LLMSeedAnalogReservoirBuilder
from tasks.small_molecule.core.llm_advisor.client import MockLLMClient
from tasks.small_molecule.core.rng import RNG


def test_llm_seed_analog_uses_llm_seeds_and_analog_generator():
    client = MockLLMClient(
        scripted_responses=[
            json.dumps({
                "seeds": [
                    {"smiles": "CCO", "budget": 3, "intent": "seed one"},
                    {"smiles": "CCN", "budget": 3, "intent": "seed two"},
                ]
            })
        ]
    )
    seen_seed_batches = []

    def analog_fn(seeds):
        seen_seed_batches.append(list(seeds))
        return [seed + "C" for seed in seeds] + [seed + "O" for seed in seeds]

    cfg = TiltedLDMCase2Config(
        "m1_llm_seed_analog_oversample_sir",
        m1_analog_n_llm_seeds=2,
        m1_analog_k_total=6,
    )

    result = LLMSeedAnalogReservoirBuilder().build([], cfg, client, analog_fn, RNG(0))

    assert len(client.call_log) == 1
    assert seen_seed_batches == [["CCO"], ["CCN"]]
    assert {candidate.canonical_smiles for candidate in result.candidates} == {
        "CCOC",
        "CCOO",
        "CCNC",
        "CCNO",
    }
    assert result.parsed_llm_json["seed_plan"]["seeds"][0]["budget"] == 3
    assert all(source.source_type == "reasyn" for source in result.sources)


def test_llm_seed_analog_batches_real_generator_when_targets_available():
    client = MockLLMClient(
        scripted_responses=[
            json.dumps({
                "seeds": [
                    {"smiles": "CCO", "budget": 3, "intent": "seed one"},
                    {"smiles": "CCN", "budget": 3, "intent": "seed two"},
                ]
            })
        ]
    )
    seen_batches = []

    def analog_fn(seeds):
        raise AssertionError("real target-aware generator path should be used")

    def generate_with_targets(seeds):
        seen_batches.append(list(seeds))
        return [
            {"target": "CCO", "smiles": "CCOC"},
            {"target": "CCO", "smiles": "CCOO"},
            {"target": "CCN", "smiles": "CCNC"},
            {"target": "CCN", "smiles": "CCNO"},
        ]

    analog_fn.generate_with_targets = generate_with_targets
    cfg = TiltedLDMCase2Config(
        "m1_llm_seed_analog_oversample_sir",
        m1_analog_n_llm_seeds=2,
        m1_analog_k_total=6,
    )

    result = LLMSeedAnalogReservoirBuilder().build([], cfg, client, analog_fn, RNG(0))

    assert seen_batches == [["CCO", "CCN"]]
    sources_by_id = {source.source_id: source for source in result.sources}
    assert sources_by_id["m1_analog_seed_0"].generated_count == 2
    assert sources_by_id["m1_analog_seed_1"].generated_count == 2
    assert {candidate.canonical_smiles for candidate in result.candidates} == {
        "CCOC",
        "CCOO",
        "CCNC",
        "CCNO",
    }


def test_llm_seed_analog_refills_missing_seed_count():
    client = MockLLMClient(
        scripted_responses=[
            json.dumps({"seeds": [{"smiles": "CCO", "budget": 3, "intent": "first"}]}),
            json.dumps({"seeds": [{"smiles": "CCN", "budget": 3, "intent": "second"}]}),
        ]
    )

    def analog_fn(seeds):
        return [seed + "C" for seed in seeds]

    cfg = TiltedLDMCase2Config(
        "m1_llm_seed_analog_oversample_sir",
        m1_analog_n_llm_seeds=2,
        m1_analog_k_total=6,
        llm_max_retries=1,
    )

    result = LLMSeedAnalogReservoirBuilder().build([], cfg, client, analog_fn, RNG(0))

    assert len(client.call_log) == 2
    assert "Need 1 additional seed" in client.call_log[1]["user"]
    assert {source.seed_smiles for source in result.sources} == {"CCO", "CCN"}


def test_llm_seed_analog_refills_when_valid_pool_is_below_target():
    client = MockLLMClient(
        scripted_responses=[
            json.dumps({"seeds": [{"smiles": "CCO", "budget": 2, "intent": "first"}]}),
            json.dumps({"seeds": [{"smiles": "CCN", "budget": 2, "intent": "refill"}]}),
        ]
    )
    calls = []

    def analog_fn(seeds):
        calls.append(list(seeds))
        seed = seeds[0]
        if seed == "CCO":
            return ["CCOC"]
        return ["CCNC", "CCNO", "CCNN"]

    cfg = TiltedLDMCase2Config(
        "m1_llm_seed_analog_oversample_sir",
        m1_analog_n_llm_seeds=1,
        m1_analog_k_total=4,
        max_candidates_per_round=3,
        llm_max_retries=1,
    )

    result = LLMSeedAnalogReservoirBuilder().build([("CCCC", (-1.0, 5.1))], cfg, client, analog_fn, RNG(0))

    assert len(client.call_log) == 2
    assert calls == [["CCO"], ["CCN"]]
    assert result.parsed_llm_json["analog_refill_rounds"] == 1
    assert len(result.candidates) >= 3
