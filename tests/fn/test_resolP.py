"""resolP is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.resolP import resolution_proof


def test_resolP_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        resolution_proof(clauses=None)
