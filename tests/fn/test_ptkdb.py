"""ptkdb is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ptkdb import kde_bandwidth


def test_ptkdb_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        kde_bandwidth(data=None)
