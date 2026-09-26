"""mdcyc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mdcyc import mdcyc


def test_mdcyc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mdcyc()
