"""hyaqu is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyaqu import hyaqu


def test_hyaqu_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyaqu()
