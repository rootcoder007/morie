"""srgam is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srgam import srgam


def test_srgam_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srgam()
