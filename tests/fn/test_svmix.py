"""svmix is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svmix import svmix


def test_svmix_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svmix()
