% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.elmo_mix}
\alias{.elmo_mix}
\title{.elmo_mix}
\usage{
.elmo_mix(reps, raw_weights, gamma = 1, position = NULL)
}
\arguments{
\item{reps}{A vector; its length is taken and its elements indexed.}
\item{raw_weights}{A vector; its length is taken.}
\item{gamma}{Numeric; combined arithmetically in the body. Defaults to \code{1}.}
\item{position}{Defaults to \code{NULL}.}
}
\description{
A step of the elmo_native implementation. Called by \code{.elmo_representation}. See the file header for the source the module follows.
}
\value{
One of two values, depending on the branch taken.
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
