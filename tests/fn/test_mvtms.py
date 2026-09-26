"""mvtms is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mvtms import mvtms


def test_mvtms_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mvtms()
