"""eldiv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.eldiv import eldiv


def test_eldiv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        eldiv()
