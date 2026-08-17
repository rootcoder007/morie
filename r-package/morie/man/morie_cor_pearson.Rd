% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_cor_pearson}
\alias{morie_cor_pearson}
\title{Fast Pearson correlation}
\usage{
morie_cor_pearson(x, y)
}
\description{
Single-pass C++ kernel.  Equivalent to \code{cor(x, y)} when both vectors are equal-length and complete (no NA handling).
}
\value{
The value of \code{if}.
}
\keyword{internal}
\examples{
abs(morie_cor_pearson(1:10, 2 * (1:10) + 3) - 1) < 1e-9
}
