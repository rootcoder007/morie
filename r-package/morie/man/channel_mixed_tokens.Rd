% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{channel_mixed_tokens}
\alias{channel_mixed_tokens}
\title{The alternative: one token per time position, all channels}
\usage{
channel_mixed_tokens(X, patch_len, stride = NULL)
}
\arguments{
\item{X}{A matrix; passed to \code{as.matrix}.}
\item{patch_len}{See Usage.}
\item{stride}{Defaults to \code{NULL}.}
}
\description{
blended. Provided for contrast. Summing across channels at the first projection is what makes such a model permutation-invariant and so unable to say which channel a signal came from.
}
\value{
A list with \code{tokens}, \code{n_patches}, \code{n_tokens_total}, \code{design}, \code{note}.
}
\keyword{internal}
\examples{
channel_mixed_tokens(X = c(1, 2, 3, 4, 5, 6, 7, 8), patch_len = 5L)
}
