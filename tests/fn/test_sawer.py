"""sawer is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sawer import sawer


def test_sawer_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sawer()
