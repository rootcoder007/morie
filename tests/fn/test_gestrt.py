"""gestrt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gestrt import gestrt


def test_gestrt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gestrt()
