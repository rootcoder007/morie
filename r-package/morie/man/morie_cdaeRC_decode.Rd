% Generated from the module source by ledger/gen_rd.R -- roxygen2 hangs
% on this package. Usage comes from args(), so it tracks the formals.
\name{morie_cdaeRC_decode}
\alias{morie_cdaeRC_decode}
\title{Collaborative denoising auto-encoder for top-N recommendation}
\usage{
morie_cdaeRC_decode(z, Wp, bp, items = NULL, activation = "sigmoid")
}
\description{
Collaborative denoising auto-encoder for top-N recommendation. "Collaborative Denoising Auto-Encoders for Top-N Recommender Systems", *Proceedings of the Ninth ACM International Conference on Web Search and Data Mining (WSDM '16)*, 153-162, pair-wise objective framework, the four loss functions of Table 1 and the warning that log and hinge losses need y = -1 for negatives, and the need to augment positives with sampled negatives; Sec. 2.3 for the auto-encoder, the tied weights and the mask-out/drop-out corruption scaled by 1/(1-q) so it stays unbiased; Sec. 3 (eqs. (9)-(13)) for the CDAE with the user-specific node V_u, and Algorithm 1 for the SGD with negative sampling. Vincent, P., Larochelle, H., Bengio, Y. & features with denoising autoencoders", *ICML 2008*, 1096-1103, Rendle, S., Freudenthaler, C., Gantner, Z. & Schmidt-Thieme, L. Feedback", *UAI 2009*, 452-461, arXiv:1205.2618, for the pair-wise objective in Table 1.
}
\references{
Sources: Wu, Y., DuBois, C., Zheng, A. X. & Ester, M. (2016)
doi:10.1145/2835776.2835837, for Sec. 2's point-wise and
Manzagol, P.-A. (2008) "Extracting and composing robust
doi:10.1145/1390156.1390294, for the denoising auto-encoder.
(2009) "BPR: Bayesian Personalized Ranking from Implicit
}
\keyword{internal}
