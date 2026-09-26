"""umcon is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.umcon import umcon


def test_umcon_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        umcon()
