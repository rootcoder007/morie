% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{motfsr_run}
\alias{motfsr_run}
\title{motfsr_run}
\usage{
motfsr_run(sequences, w, alphabet = NULL, n_motifs = 1, beta = 0.01, lambda0 = NULL, max_iter = 1000, tol = 1e-06, normalize_overlaps = TRUE, starts = "subsequences", start_weight = 0.5, max_starts = 200, start_scoring = "one_step", erase_by = "letter", loss = NULL)
}
\arguments{
\item{sequences}{See Usage.}
\item{w}{See Usage.}
\item{alphabet}{Defaults to \code{NULL}.}
\item{n_motifs}{Defaults to \code{1}.}
\item{beta}{Defaults to \code{0.01}.}
\item{lambda0}{Defaults to \code{NULL}.}
\item{max_iter}{Defaults to \code{1000}.}
\item{tol}{Defaults to \code{1e-06}.}
\item{normalize_overlaps}{Defaults to \code{TRUE}.}
\item{starts}{Defaults to \code{"subsequences"}.}
\item{start_weight}{Defaults to \code{0.5}.}
\item{max_starts}{Defaults to \code{200}.}
\item{start_scoring}{Defaults to \code{"one_step"}.}
\item{erase_by}{Defaults to \code{"letter"}.}
\item{loss}{Defaults to \code{NULL}.}
}
\description{
Part of the motfsr_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{estimate}, \code{motifs}, \code{alphabet}, \code{w}, \code{n_subsequences}, \code{erasing}, \code{method}.
}
\references{
  Bailey & Elkan (1994) ISMB-94, 28-36.
  Bailey & Elkan (1995) ISMB-95, 21-29.
Base R only.
}
\keyword{internal}
\examples{
motfsr_run(sequences = data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9)), w = 5L)
}
