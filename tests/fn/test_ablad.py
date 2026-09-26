"""ablad is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ablad import ablad


def test_ablad_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ablad()
