"""msrot is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msrot import rotate_config


def test_msrot_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rotate_config(data=None)
