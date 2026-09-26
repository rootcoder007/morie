"""cliptx is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cliptx import clip_image_text


def test_cliptx_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clip_image_text(images=None, texts=None, tau=None)
