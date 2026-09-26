"""stmet is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stmet import stmet


def test_stmet_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stmet()
