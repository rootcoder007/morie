"""enelr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.enelr import enelr


def test_enelr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        enelr()
