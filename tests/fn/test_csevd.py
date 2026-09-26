"""csevd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csevd import csevd


def test_csevd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csevd()
