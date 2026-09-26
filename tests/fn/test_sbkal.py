"""sbkal is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sbkal import sbkal


def test_sbkal_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sbkal()
