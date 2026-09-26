"""chlann is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.chlann import chlann


def test_chlann_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        chlann()
