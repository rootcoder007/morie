% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{teacher_votes}
\alias{teacher_votes}
\title{The vote histogram n_j(x) for each record. teacher_predicts is a}
\usage{
teacher_votes(teacher_predicts, rows, n_classes = NULL) 
}
\arguments{
\item{teacher_predicts}{Coerced to list by the body, with \code{as.list}.}
\item{rows}{A vector; its length is taken.}
\item{n_classes}{Defaults to \code{NULL}.}
}
\description{
list of callables mapping rows to labels (or to probability vectors, in which case the argmax is the vote).
}
\value{
The value of \code{split}.
}
\keyword{internal}
