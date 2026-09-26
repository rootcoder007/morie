"""foreld is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.foreld import foreld


def test_foreld_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        foreld()
