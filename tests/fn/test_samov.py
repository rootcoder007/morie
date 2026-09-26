"""samov is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.samov import samov


def test_samov_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        samov()
