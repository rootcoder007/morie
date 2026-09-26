"""srper is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srper import srper


def test_srper_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srper()
