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
\item{sequences}{Passed to \code{motfsr_prepare}.}
\item{w}{A count; the body uses it as \code{seq_len(...)}.}
\item{alphabet}{Passed to \code{motfsr_prepare}.}
\item{n_motifs}{A count; the body uses it as \code{seq_len(...)}. Defaults to \code{1}.}
\item{beta}{Passed to \code{motfsr_mm_fit}. Defaults to \code{0.01}.}
\item{lambda0}{Passed to \code{motfsr_lambda_grid}.}
\item{max_iter}{Passed to \code{motfsr_mm_fit}. Defaults to \code{1000}.}
\item{tol}{Passed to \code{motfsr_mm_fit}. Defaults to \code{1e-06}.}
\item{normalize_overlaps}{Passed to \code{motfsr_mm_fit}. Defaults to \code{TRUE}.}
\item{starts}{One of \code{"subsequences"}, \code{"uniform"}. Defaults to \code{"subsequences"}.}
\item{start_weight}{Coerced to numeric by the body, with \code{as.numeric}. Defaults to \code{0.5}.}
\item{max_starts}{Coerced to integer by the body, with \code{as.integer}. Defaults to \code{200}.}
\item{start_scoring}{One of \code{"none"}, \code{"one_step"}. Defaults to \code{"one_step"}.}
\item{erase_by}{Passed to \code{motfsr_mm_fit}. Defaults to \code{"letter"}.}
\item{loss}{Passed to \code{motfsr_bayes_threshold}.}
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
