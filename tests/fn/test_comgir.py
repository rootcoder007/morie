"""comgir is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.comgir import girvan_newman


def test_comgir_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        girvan_newman(G=None)
