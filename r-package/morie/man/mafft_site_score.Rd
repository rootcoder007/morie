% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{mafft_site_score}
\alias{mafft_site_score}
\title{mafft_site_score}
\usage{
mafft_site_score(M, ga, gb, wa, wb, i, j) 
}
\arguments{
\item{M}{Passed to \code{mafft_lookup}.}
\item{ga}{A vector; its length is taken and its elements indexed.}
\item{gb}{A vector; its length is taken and its elements indexed.}
\item{wa}{A vector; indexed elementwise.}
\item{wb}{A vector; indexed elementwise.}
\item{i}{Character; passed to \code{substr}.}
\item{j}{Character; passed to \code{substr}.}
}
\description{
Part of the mafft_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{tot}, as built in the body.
}
\keyword{internal}
\examples{
M <- list(A_A = 1, C_C = 1, G_G = 1, T_T = 1)
ga <- c("ACGT")
gb <- c("ACGT")
s <- mafft_site_score(M, ga, gb, wa = 1, wb = 1, i = 1, j = 1)
is.numeric(s)
}
