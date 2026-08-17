% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{attack_dataset}
\alias{attack_dataset}
\title{attack_dataset}
\usage{
attack_dataset(model_predict, in_X, in_y, out_X, out_y)
}
\arguments{
\item{model_predict}{Numeric; the body checks with \code{is.numeric}.}
\item{in_X}{A vector; its length is taken.}
\item{in_y}{A vector; indexed elementwise.}
\item{out_X}{A vector; its length is taken.}
\item{out_y}{A vector; indexed elementwise.}
}
\description{
Part of the memb_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{rows}, \code{labels}, \code{classes}.
}
\keyword{internal}
\examples{
set.seed(1)
model_predict <- function(rows) lapply(rows, function(r)
  {p <- exp(r); p / sum(p)})
in_X <- lapply(1:5, function(i) rnorm(3))
out_X <- lapply(1:5, function(i) rnorm(3))
r <- attack_dataset(model_predict, in_X, as.list(rep(0, 5)),
                    out_X, as.list(rep(1, 5)))
str(r, max.level = 1)
}
