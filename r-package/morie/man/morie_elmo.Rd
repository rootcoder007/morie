% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_elmo}
\alias{morie_elmo}
\title{morie_elmo}
\usage{
morie_elmo(X, layers, raw_weights = NULL, gamma = 1)
}
\arguments{
\item{X}{Passed to \code{.elmo_representation}.}
\item{layers}{Passed to \code{.elmo_representation}.}
\item{raw_weights}{Passed to \code{.elmo_representation}.}
\item{gamma}{Passed to \code{.elmo_representation}. Defaults to \code{1}.}
}
\description{
A step of the elmo_native implementation. No other function in the package calls it. See the file header for the source the module follows.
}
\value{
The value of \code{.elmo_representation}.
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
\examples{
set.seed(2)
d <- 2
X <- matrix(rnorm(6, 0, 0.5), 3, d)
mkw <- function() matrix(rnorm(4 * d * d, 0, 0.3), 4 * d, d)
layer <- list(mkw(), mkw(), rep(0, 4 * d), mkw(), mkw(), rep(0, 4 * d))
r <- morie_elmo(X, list(layer))
str(r, max.level = 1)
}
