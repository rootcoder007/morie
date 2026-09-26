"""svcpw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svcpw import copeland_winner


def test_svcpw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        copeland_winner(data=None)
