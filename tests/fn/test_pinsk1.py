"""pinsk1 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pinsk1 import pinsker_inequality


def test_pinsk1_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pinsker_inequality(p=None, q=None)
