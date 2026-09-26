"""entfp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.entfp import entfp


def test_entfp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        entfp()
