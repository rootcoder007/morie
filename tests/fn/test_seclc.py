"""seclc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.seclc import seclc


def test_seclc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        seclc()
