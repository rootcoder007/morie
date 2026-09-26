"""encfd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.encfd import encfd


def test_encfd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        encfd()
