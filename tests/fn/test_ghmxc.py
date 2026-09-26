"""ghmxc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghmxc import ghmxc


def test_ghmxc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghmxc()
