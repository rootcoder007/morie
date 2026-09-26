"""segrw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.segrw import segrw


def test_segrw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        segrw()
