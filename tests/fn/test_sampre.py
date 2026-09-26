"""sampre is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sampre import sam_prompt_encoder


def test_sampre_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sam_prompt_encoder(prompts=None)
