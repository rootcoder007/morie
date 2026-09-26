"""aitpie is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.aitpie import compositional_pielou


def test_aitpie_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        compositional_pielou(x=None)
