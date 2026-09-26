"""pcfsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pcfsp import pcfsp


def test_pcfsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pcfsp()
