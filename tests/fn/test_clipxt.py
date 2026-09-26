"""clipxt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clipxt import clip_text_encoder


def test_clipxt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clip_text_encoder(text=None)
