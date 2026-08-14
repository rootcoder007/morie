# morie.fn -- function file (rootcoder007/morie)
"""Local differential privacy via randomized response -- alias of rrand.

The generated stub described the local model of differential privacy
(each user randomizes before the collector sees anything), citing
Kasiviswanathan et al. (2011).  The canonical local-DP mechanism -- and
the one that paper builds on -- is Warner randomized response, which
already ships as ``rrand.randomized_response`` with the flip
probability 1/(1 + e^epsilon) that achieves epsilon-local-DP.  This
module aliases it rather than adding a second implementation.

References
----------
Kasiviswanathan, S. P., Lee, H. K., Nissim, K., Raskhodnikova, S., &
    Smith, A. (2011). What can we learn privately? *SIAM Journal on
    Computing*, 40(3), 793-826. (Local model, section 1; randomized
    response as the basic local protocol.)
Warner, S. L. (1965). Randomized response: a survey technique for
    eliminating evasive answer bias. *JASA*, 60(309), 63-69.
Dwork, C., & Roth, A. (2014). *FnT-TCS*, 9(3-4), section 3.2.
    Local source: /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/dwork-roth-2014-algorithmic-foundations-differential-privacy.pdf
"""

from .rrand import randomized_response

__all__ = ["locdp", "local_dp"]

#: Primary name: alias of :func:`morie.fn.rrand.randomized_response`.
locdp = randomized_response

#: Legacy stub name, kept for compatibility.
local_dp = randomized_response


def cheatsheet():
    return "locdp: local DP randomized response (alias of rrand.randomized_response)."

# public names resolved by fn/_lazy_map.json
localdp = randomized_response
