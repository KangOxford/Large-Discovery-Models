"""tests/core/ldm/llm/test_response_parser.py"""
from __future__ import annotations

import pytest

from tasks.antibody.core.ldm.llm.response_parser import ParsedUpdate, parse_response


class TestParseResponse:
    def test_empty_string(self):
        with pytest.raises(ValueError, match="empty"):
            parse_response("")

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            parse_response("not json")

    def test_not_an_object(self):
        with pytest.raises(ValueError, match="must be a JSON object"):
            parse_response("[1, 2, 3]")

    def test_empty_object_is_noop(self):
        result = parse_response("{}")
        assert isinstance(result, ParsedUpdate)
        assert result.is_noop

    def test_update_trust_region_only(self):
        result = parse_response(
            '{"update_trust_region": "LocalSearch(\'ARYYGSYWYFD\', radius=2, restart=1, steps=100)"}'
        )
        assert result.update_trust_region is not None
        assert result.update_bias is None
        assert not result.is_noop

    def test_update_bias_only(self):
        result = parse_response('{"update_bias": "MaxCysteine(1)"}')
        assert result.update_trust_region is None
        assert result.update_bias == "MaxCysteine(1)"

    def test_both_fields(self):
        result = parse_response(
            '{"update_trust_region": "NeighborSampling(\'ARDYGNYWYFD\', mut_pr=0.5)", '
            '"update_bias": "MaxCysteine(1)"}'
        )
        assert result.update_trust_region is not None
        assert result.update_bias is not None

    def test_rationale_parsed(self):
        result = parse_response(
            '{"rationale": "tightening", '
            '"update_trust_region": "LocalSearch(\'ARYYGSYWYFD\', radius=1, restart=2, steps=50)"}'
        )
        assert result.rationale == "tightening"

    def test_rationale_only_is_noop(self):
        result = parse_response('{"rationale": "keep"}')
        assert result.is_noop

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="unknown"):
            parse_response('{"action": "pass"}')

    def test_json_fence(self):
        result = parse_response(
            "```json\n"
            '{"update_trust_region": "LatinHyperCubeSampling(num=1000)"}\n'
            "```"
        )
        assert result.update_trust_region is not None
