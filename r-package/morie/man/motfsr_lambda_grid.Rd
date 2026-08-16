% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{motfsr_lambda_grid}
\alias{motfsr_lambda_grid}
\title{motfsr_lambda_grid}
\usage{
motfsr_lambda_grid(n_starts_total, n_seqs, w, lambda0)
}
\arguments{
\item{n_starts_total}{Numeric; combined arithmetically in the body.}
\item{n_seqs}{Numeric; passed to \code{sqrt}.}
\item{w}{Numeric; combined arithmetically in the body.}
\item{lambda0}{Optional; may be \code{NULL}. Coerced to numeric by the body, with \code{as.numeric}.}
}
\description{
Part of the motfsr_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{out}, as built in the body.
}
\references{
  Bailey & Elkan (1994) ISMB-94, 28-36.
  Bailey & Elkan (1995) ISMB-95, 21-29.
Base R only.
}
\keyword{internal}
\examples{
motfsr_lambda_grid(n_starts_total = c(1, 2, 3, 4, 5, 6, 7, 8), n_seqs = c(1, 2, 3, 4, 5, 6, 7, 8), w = c(1, 2, 3, 4, 5, 6, 7, 8), lambda0 = c(1, 2, 3, 4, 5, 6, 7, 8))
}
