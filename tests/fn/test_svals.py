"""svals is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svals import svals


def test_svals_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svals()
