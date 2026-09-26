"""mstnk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mstnk import trustworthiness


def test_mstnk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trustworthiness(data=None)
