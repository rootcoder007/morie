"""srgwb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srgwb import srgwb


def test_srgwb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srgwb()
