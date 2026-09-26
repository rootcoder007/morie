"""rescor is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rescor import rescore_consensus


def test_rescor_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rescore_consensus(scores=None)
