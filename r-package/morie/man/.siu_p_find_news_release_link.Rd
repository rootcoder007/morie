% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.siu_p_find_news_release_link}
\alias{.siu_p_find_news_release_link}
\title{.siu_p_find_news_release_link}
\usage{
.siu_p_find_news_release_link(html, source_url)
}
\arguments{
\item{html}{Passed to \code{regmatches}.}
\item{source_url}{Optional; may be \code{NULL}. Character; passed to \code{sub}.}
}
\description{
Part of the siu_parser implementation; see the file header for the source it follows.
}
\value{
The value of \code{m}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
.siu_p_find_news_release_link(V, V)
}
