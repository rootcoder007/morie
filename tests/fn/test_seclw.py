"""seclw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.seclw import seclw


def test_seclw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        seclw()
