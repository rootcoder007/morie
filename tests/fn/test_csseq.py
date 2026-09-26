"""csseq is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csseq import csseq


def test_csseq_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csseq()
