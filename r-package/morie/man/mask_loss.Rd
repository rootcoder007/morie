% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mask_loss}
\alias{mask_loss}
\title{mask_loss}
\usage{
mask_loss(logits, target, decoupled = TRUE)
}
\arguments{
\item{logits}{A matrix; passed to \code{as.matrix}.}
\item{target}{A matrix; passed to \code{as.matrix}.}
\item{decoupled}{A flag; the body branches on it. Defaults to \code{TRUE}.}
}
\description{
Part of the masrcn_native implementation; see the file header for the source it follows.
}
\value{
A list, whose contents depend on the branch taken; across the branches its names are \code{loss}, \code{kind}, \code{note}, \code{caveat}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
mask_loss(V, V)
}
