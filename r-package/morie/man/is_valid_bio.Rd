% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{is_valid_bio}
\alias{is_valid_bio}
\title{is_valid_bio}
\usage{
is_valid_bio(path, labels = NULL) 
}
\arguments{
\item{path}{See Usage.}
\item{labels}{Accepted by the signature and not used anywhere in the body.}
}
\description{
A step of the benRea_native implementation. Called by \code{ner_decode}. See the file header for the source the module follows.
}
\value{
A logical value.
}
\keyword{internal}
\examples{
c(is_valid_bio(c("O", "B-PER", "I-PER", "O")),
  is_valid_bio(c("O", "I-PER", "O")))
}
