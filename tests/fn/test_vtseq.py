"""vtseq is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.vtseq import vtseq


def test_vtseq_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        vtseq()
