"""Tests for allktmpl.alammar_instruction_data_template."""

from morie.fn.allktmpl import alammar_instruction_data_template


def test_allktmpl_basic():
    out = alammar_instruction_data_template(
        [{"instruction": "add", "input": "2 2", "output": "4"}])
    s, e = out["output_spans"][0]
    assert out["texts"][0][s:e] == "4"


def test_allktmpl_edge():
    import pytest
    with pytest.raises(ValueError, match="missing"):
        alammar_instruction_data_template([{"instruction": "x"}])
