"""srgwe is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srgwe import srgwe


def test_srgwe_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srgwe()
