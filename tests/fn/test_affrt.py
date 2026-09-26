"""affrt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.affrt import affrt


def test_affrt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        affrt()
