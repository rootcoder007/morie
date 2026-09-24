"""Tests for kmnxtg.kamath_nextgpt_any2any."""

import pytest

from morie.fn.kmnxtg import kamath_nextgpt_any2any


def test_kmnxtg_basic():
    """Test basic multi-modal encode / LLM / decode pipeline."""
    encoders = {
        "text": lambda s: [len(s)],
        "image": lambda a: [float(sum(a))],
    }
    decoders = {
        "audio": lambda h: ("wav", list(h)),
        "caption": lambda h: ("cap", list(h)),
    }

    def llm(feats):
        return [feats["image"][0] + feats["text"][0]]

    inputs_by_modality = {"text": "abc", "image": [1, 2, 3]}
    result = kamath_nextgpt_any2any(
        inputs_by_modality, encoders, llm, decoders)
    assert isinstance(result, dict)
    assert "outputs" in result
    assert "features" in result
    assert "llm_state" in result
    assert "input_modalities" in result
    assert "output_modalities" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    assert result["input_modalities"] == ["image", "text"]
    assert set(result["outputs"].keys()) == {"audio", "caption"}
    assert result["estimate"] == 2
    assert result["n"] == 4
    assert result["method"] == "NExT-GPT any-to-any encode / LLM / decode pipeline"


def test_kmnxtg_edge():
    """A missing encoder for an input modality must raise ValueError."""
    encoders = {"text": lambda s: [len(s)]}
    decoders = {"audio": lambda h: ("wav", list(h))}

    def llm(feats):
        return list(feats["text"])

    inputs_by_modality = {"text": "hi", "image": [1, 2]}
    with pytest.raises(ValueError):
        kamath_nextgpt_any2any(inputs_by_modality, encoders, llm, decoders)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmnxtg as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
