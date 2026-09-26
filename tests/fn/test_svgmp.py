"""svgmp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svgmp import svgmp


def test_svgmp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svgmp()
