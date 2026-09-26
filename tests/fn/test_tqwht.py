"""tqwht is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tqwht import turboquant_walsh_hadamard_transform


def test_tqwht_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        turboquant_walsh_hadamard_transform(x=None)
