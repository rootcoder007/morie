% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.bilm_forward}
\alias{.bilm_forward}
\title{.bilm_forward}
\usage{
.bilm_forward(X, layers)
}
\arguments{
\item{X}{See Usage.}
\item{layers}{See Usage.}
}
\description{
Part of the elmo_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{reps}, as built in the body.
}
\references{
Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K. &
Zettlemoyer, L. (2018) "Deep contextualized word representations",
Proceedings of NAACL-HLT 2018, 2227-2237, arXiv:1802.05365. Sec. 3,
eq. (1), and the layer-weighting scheme.
Hochreiter, S. & Schmidhuber, J. (1997) "Long Short-Term Memory",
Neural Computation 9(8), 1735-1780. The recurrent cell the biLM is
built from.
Ba, J. L., Kiros, J. R. & Hinton, G. E. (2016) "Layer Normalization",
arXiv:1607.06450. The normalisation the paper applies per layer
before weighting.
}
\keyword{internal}
