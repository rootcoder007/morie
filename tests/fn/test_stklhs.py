"""stklhs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stklhs import stklhs


def test_stklhs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stklhs()
