"""mdcwd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mdcwd import mdcwd


def test_mdcwd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mdcwd()
