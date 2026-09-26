"""svcly is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svcly import coalition_yolk


def test_svcly_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        coalition_yolk(data=None)
