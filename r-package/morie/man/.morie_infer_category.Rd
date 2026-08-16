% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_infer_category}
\alias{.morie_infer_category}
\title{Infer a category for a function `fn_name` by searching the installed}
\usage{
.morie_infer_category(fn_name)
}
\arguments{
\item{fn_name}{Character; passed to \code{startsWith}.}
}
\description{
R/ directory for files whose names begin with a known prefix.  Falls back to scanning the function\\'s source attributes when available.
}
\value{
A character value.
}
\keyword{internal}
