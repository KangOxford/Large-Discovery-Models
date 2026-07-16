"""tests/bo/ldm/llm/test_client.py"""
from __future__ import annotations

import pytest

from bo.ldm.llm.client import LLMClient


class MockClient(LLMClient):
    """Concrete mock for testing."""
    def __init__(self, response: str = "") -> None:
        self.response = response
        self.calls: list[dict] = []

    def call(self, prompt, temperature, timeout_s):
        self.calls.append({"prompt": prompt, "temperature": temperature, "timeout_s": timeout_s})
        return self.response


class TestLLMClientAbstract:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            LLMClient()

    def test_mock_call(self):
        client = MockClient("hello")
        out = client.call("test prompt", 0.5, 10)
        assert out == "hello"
        assert len(client.calls) == 1
        assert client.calls[0]["temperature"] == 0.5

    def test_mock_multiple_calls(self):
        client = MockClient("response")
        for i in range(3):
            client.call(f"prompt {i}", 0.1, 5)
        assert len(client.calls) == 3