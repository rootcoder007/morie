"""srmgc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srmgc import srmgc


def test_srmgc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srmgc()
