"""soseq is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.soseq import soseq


def test_soseq_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        soseq()
