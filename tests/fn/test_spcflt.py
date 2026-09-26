"""spcflt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.spcflt import spcflt


def test_spcflt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        spcflt()
