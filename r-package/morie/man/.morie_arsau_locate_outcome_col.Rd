% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_arsau_locate_outcome_col}
\alias{.morie_arsau_locate_outcome_col}
\title{.morie_arsau_locate_outcome_col}
\usage{
.morie_arsau_locate_outcome_col(df, target)
}
\arguments{
\item{df}{See Usage.}
\item{target}{Character; passed to \code{tolower}.}
}
\description{
A step of the arsau_analyze implementation. Called by \code{morie_arsau_analyze_individual_records}. See the file header for the source the module follows.
}
\value{
The value of \code{[}.
}
\keyword{internal}
\examples{
.morie_arsau_locate_outcome_col(df = data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9)), target = 5L)
}
