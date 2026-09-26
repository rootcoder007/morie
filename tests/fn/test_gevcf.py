"""gevcf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gevcf import gevcf


def test_gevcf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gevcf()
