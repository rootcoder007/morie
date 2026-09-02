% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{training_stage}
\alias{training_stage}
\title{training_stage}
\usage{
training_stage(stage) 
}
\arguments{
\item{stage}{Coerced to integer by the body, with \code{as.integer}.}
}
\description{
Part of the llavx_native implementation; see the file header for the source it follows.
}
\value{
A list, whose contents depend on the branch taken; across the branches its names are \code{stage}, \code{trainable}, \code{frozen}, \code{data}, \code{note}.
}
\keyword{internal}
\examples{
s <- training_stage(1)
s$stage == 1L
}
