"""sawad is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sawad import sawad


def test_sawad_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sawad()
