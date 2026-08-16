"""tests/core/ldm/dsl/test_exceptions.py"""
from __future__ import annotations

import pytest

from tasks.antibody.core.ldm.dsl.exceptions import (
    DSLValidationError,
    DSLSyntaxError,
    NestingTooDeep,
    SamplingTimeout,
)


class TestExceptions:
    def test_syntax_error_is_exception(self):
        assert issubclass(DSLSyntaxError, Exception)

    def test_validation_error_is_exception(self):
        assert issubclass(DSLValidationError, Exception)

    def test_sampling_timeout_is_validation_error(self):
        assert issubclass(SamplingTimeout, DSLValidationError)

    def test_nesting_too_deep_is_validation_error(self):
        assert issubclass(NestingTooDeep, DSLValidationError)

    def test_sampling_timeout_message(self):
        e = SamplingTimeout("timed out")
        assert "timed out" in str(e)

    def test_can_raise_and_catch(self):
        with pytest.raises(DSLValidationError):
            raise SamplingTimeout("test")
