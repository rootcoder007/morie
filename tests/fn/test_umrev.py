"""umrev is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.umrev import umrev


def test_umrev_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        umrev()
