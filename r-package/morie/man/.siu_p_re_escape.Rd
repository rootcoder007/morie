% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.siu_p_re_escape}
\alias{.siu_p_re_escape}
\title{.siu_p_re_escape}
\usage{
.siu_p_re_escape(s)
}
\arguments{
\item{s}{Character; passed to \code{gsub}.}
}
\description{
Part of the siu_parser implementation; see the file header for the source it follows.
}
\value{
The value of \code{gsub}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.siu_p_re_escape(V)
}
