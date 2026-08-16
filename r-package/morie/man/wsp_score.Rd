% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{wsp_score}
\alias{wsp_score}
\title{wsp_score}
\usage{
wsp_score(alignment, scoring, s_op = 2.4, weights = NULL)
}
\arguments{
\item{alignment}{See Usage.}
\item{scoring}{A list; the body reads \code{$matrix} from it.}
\item{s_op}{Numeric; combined arithmetically in the body. Defaults to \code{2.4}.}
\item{weights}{Defaults to \code{NULL}.}
}
\description{
Part of the mafft_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{total}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
D <- data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9))
wsp_score(V, D)
}
