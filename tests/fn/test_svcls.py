"""svcls is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svcls import coalition_size


def test_svcls_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        coalition_size(data=None)
