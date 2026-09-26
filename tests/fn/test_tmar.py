"""tmar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tmar import tmar


def test_tmar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tmar()
