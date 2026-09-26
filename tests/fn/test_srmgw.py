"""srmgw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.srmgw import srmgw


def test_srmgw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        srmgw()
