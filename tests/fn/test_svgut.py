"""svgut is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svgut import svgut


def test_svgut_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svgut(ideal=None, pos=None)
