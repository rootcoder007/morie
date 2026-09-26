"""hyhyd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyhyd import hyhyd


def test_hyhyd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyhyd()
