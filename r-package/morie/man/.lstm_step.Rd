% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.lstm_step}
\alias{.lstm_step}
\title{.lstm_step}
\usage{
.lstm_step(x, h, c, Wx, Wh, b)
}
\arguments{
\item{x}{Coerced to numeric by the body, with \code{as.numeric}.}
\item{h}{A vector; its length is taken.}
\item{c}{A vector; its length is taken.}
\item{Wx}{A matrix; passed to \code{as.matrix}.}
\item{Wh}{A matrix; passed to \code{as.matrix}.}
\item{b}{Coerced to numeric by the body, with \code{as.numeric}.}
}
\description{
A step of the elmo_native implementation. Called by \code{.bilm_forward}. See the file header for the source the module follows.
}
\value{
A list with \code{h}, \code{c}.
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
.lstm_step(x = c(1, 2, 3, 4, 5, 6, 7, 8), h = c(1, 2, 3, 4, 5, 6, 7, 8), c = c(1, 2, 3, 4, 5, 6, 7, 8), Wx = c(1, 2, 3, 4, 5, 6, 7, 8), Wh = c(1, 2, 3, 4, 5, 6, 7, 8), b = 5L)
}
