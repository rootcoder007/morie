"""adpsmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.adpsmp import adpsmp


def test_adpsmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        adpsmp()
