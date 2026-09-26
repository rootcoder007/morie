"""ptnni is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptnni import nn_index


def test_ptnni_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nn_index(data=None)
