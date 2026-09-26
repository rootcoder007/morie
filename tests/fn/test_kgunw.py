"""kgunw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgunw import uk_weights


def test_kgunw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        uk_weights(data=None)
