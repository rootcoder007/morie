% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_llm_extract_text}
\alias{.morie_llm_extract_text}
\title{.morie_llm_extract_text}
\usage{
.morie_llm_extract_text(data)
}
\arguments{
\item{data}{A list; the body reads \code{$choices} from it.}
}
\description{
Part of the llm implementation; see the file header for the source it follows.
}
\value{
One of two values, depending on the branch taken.
}
\keyword{internal}
\examples{
D <- data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9))
.morie_llm_extract_text(D)
}
