"""clipbn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clipbn import clip_image_text_align


def test_clipbn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clip_image_text_align(images=None, texts=None, tau=None)
