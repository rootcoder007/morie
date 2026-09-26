"""dttvs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dttvs import dttvs


def test_dttvs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dttvs()
