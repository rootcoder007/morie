"""nmwnc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nmwnc import wnominate_class


def test_nmwnc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wnominate_class(data=None)
