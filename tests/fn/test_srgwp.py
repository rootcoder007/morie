"""srgwp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srgwp import srgwp


def test_srgwp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srgwp()
