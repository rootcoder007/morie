"""rcbnf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rcbnf import rcbnf


def test_rcbnf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rcbnf()
