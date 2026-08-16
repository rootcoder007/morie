% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.elmo_representation}
\alias{.elmo_representation}
\title{.elmo_representation}
\usage{
.elmo_representation(X, layers, raw_weights = NULL, gamma = 1)
}
\arguments{
\item{X}{Passed to \code{.bilm_forward}.}
\item{layers}{Passed to \code{.bilm_forward}.}
\item{raw_weights}{Defaults to \code{NULL}.}
\item{gamma}{Passed to \code{.elmo_mix}. Defaults to \code{1}.}
}
\description{
A step of the elmo_native implementation. Called by \code{morie_elmo}. See the file header for the source the module follows.
}
\value{
A list with \code{estimate}, \code{elmo}, \code{layers}, \code{weights}, \code{gamma}, \code{n_layers}, \code{L}, \code{d}, \code{top_layer}, \code{method}.
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
