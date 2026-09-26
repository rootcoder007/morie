"""smigr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.smigr import smiles_grammar_parse


def test_smigr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        smiles_grammar_parse(smiles=None)
