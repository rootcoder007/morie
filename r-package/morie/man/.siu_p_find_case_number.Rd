% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.siu_p_find_case_number}
\alias{.siu_p_find_case_number}
\title{SIU case numbers: 2 digits, optional hyphen, 3-4 letters, hyphen,}
\usage{
.siu_p_find_case_number(text)
}
\arguments{
\item{text}{Passed to \code{regmatches}.}
}
\description{
3 digits.
}
\value{
One of two values, depending on the branch taken.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.siu_p_find_case_number(V)
}
