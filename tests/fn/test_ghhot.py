"""ghhot is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghhot import ghhot


def test_ghhot_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghhot()
