"""trasg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trasg import trasg


def test_trasg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trasg()
