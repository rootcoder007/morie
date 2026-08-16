% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.siu_p_detect_language}
\alias{.siu_p_detect_language}
\title{Crude ASCII fold for matching FR markers without accents}
\usage{
.siu_p_detect_language(text)
}
\arguments{
\item{text}{Character; passed to \code{grepl}.}
}
\description{
Part of the siu_parser implementation; see the file header for the source it follows.
}
\value{
A character value.
}
\keyword{internal}
\examples{
.siu_p_detect_language(text = 5L)
}
