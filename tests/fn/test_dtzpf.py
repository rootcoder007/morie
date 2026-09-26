"""dtzpf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dtzpf import dtzpf


def test_dtzpf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dtzpf()
