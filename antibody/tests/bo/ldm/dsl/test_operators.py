"""tests/bo/ldm/dsl/test_operators.py"""
from bo.ldm.dsl.search_space import LocalSearch, Or


def test_or_operator():
    a = LocalSearch("ARYYGSYWYFD", restart=1, steps=10)
    b = LocalSearch("VRGYYSDWYMD", restart=1, steps=10)
    combined = a | b
    assert isinstance(combined, Or)
    assert combined.children == [a, b]


def test_chained_or():
    a = LocalSearch("ARYYGSYWYFD", restart=1, steps=1)
    b = LocalSearch("VRGYYSDWYMD", restart=1, steps=1)
    c = LocalSearch("QQQQQQQQQQQ", restart=1, steps=1)
    combined = a | b | c
    assert isinstance(combined, Or)


def test_operator_with_non_atom():
    a = LocalSearch("ARYYGSYWYFD")
    assert a.__or__(42) is NotImplemented
