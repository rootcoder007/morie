"""srgwk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srgwk import srgwk


def test_srgwk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srgwk()
