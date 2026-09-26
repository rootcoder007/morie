"""zsnni is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.zsnni import natural_neighbor


def test_zsnni_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        natural_neighbor(data=None)
