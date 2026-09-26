"""plgni is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.plgni import plgni


def test_plgni_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        plgni()
