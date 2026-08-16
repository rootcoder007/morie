% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{iterative_refine}
\alias{iterative_refine}
\title{iterative_refine}
\usage{
iterative_refine(alignment, scoring, tree = NULL, s_op = 2.4, max_iterate = 16L, seq_type = "aa", use_fft = TRUE, ...)
}
\arguments{
\item{alignment}{Passed to \code{unlist}.}
\item{scoring}{Passed to \code{wsp_score}.}
\item{tree}{Optional; may be \code{NULL}. A vector; its length is taken and its elements indexed.}
\item{s_op}{Passed to \code{wsp_score}. Defaults to \code{2.4}.}
\item{max_iterate}{A count; the body uses it as \code{seq_len(...)}. Defaults to \code{16L}.}
\item{seq_type}{Passed to \code{find_homologous_segments}. Defaults to \code{"aa"}.}
\item{use_fft}{A flag; the body branches on it. Defaults to \code{TRUE}.}
\item{...}{Passed through.}
}
\description{
Part of the mafft_native implementation; see the file header for the source it follows.
}
\value{
A list with \code{aln}, \code{score}, \code{rounds}.
}
\keyword{internal}
\examples{
sc <- normalized_similarity_matrix(seq_type = "aa")
aln <- progressive_align(c("MKVLATLLPLAA", "MKVLATLLPLAG",
                           "MRVLATGLPLAA"), sc, use_fft = FALSE)
r <- iterative_refine(aln, sc, max_iterate = 2L)
str(r, max.level = 1)
}
