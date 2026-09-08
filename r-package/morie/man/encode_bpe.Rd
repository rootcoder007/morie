% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{encode_bpe}
\alias{encode_bpe}
\title{Encode text with a trained BPE model}
\usage{
encode_bpe(text, model, add_prefix = TRUE) 
}
\description{
Encode text with a trained BPE model
}
\arguments{
\item{text}{Argument \code{text}; see Usage.}
\item{model}{Argument \code{model}; see Usage.}
\item{add_prefix}{Argument \code{add_prefix}; see Usage.}
}
\value{
The value of \code{out}, as built in the body.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
D <- data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9))
encode_bpe(V, D)
}
