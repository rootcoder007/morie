"""svagn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svagn import agenda_1d


def test_svagn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agenda_1d(data=None)
