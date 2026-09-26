"""hyprm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.hyprm import hyprm


def test_hyprm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        hyprm()
