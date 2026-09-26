"""hyefl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyefl import hyefl


def test_hyefl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyefl()
