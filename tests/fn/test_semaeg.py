"""semaeg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.semaeg import sam_image_encoder


def test_semaeg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sam_image_encoder(image=None)
