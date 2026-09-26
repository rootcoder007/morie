"""srgwl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srgwl import srgwl


def test_srgwl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srgwl()
