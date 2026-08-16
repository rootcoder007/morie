% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_forensics_flatten_nibrs}
\alias{.morie_forensics_flatten_nibrs}
\title{Internal: flatten one nested NIBRS JSON record to a single row}
\usage{
.morie_forensics_flatten_nibrs(rec)
}
\arguments{
\item{rec}{See Usage.}
}
\description{
Nested dicts become dotted keys; scalar lists are ";"-joined; nested lists are JSON-serialised.
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
