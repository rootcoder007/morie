% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{muzero_backup}
\alias{muzero_backup}
\title{muzero_backup}
\usage{
muzero_backup(path, value, gamma, mm) 
}
\arguments{
\item{path}{A vector; its length is taken and its elements indexed.}
\item{value}{See Usage.}
\item{gamma}{Numeric; combined arithmetically in the body.}
\item{mm}{A list; the body reads \code{$update} from it.}
}
\description{
Part of the muzero_native implementation; see the file header for the source it follows.
}
\value{
The value of \code{for}.
}
\keyword{internal}
\examples{
root <- muzero_Node()
root$expand("s", c(0.5, 0.5), list(0, 1))
child <- root$children[["0"]]
mm <- muzero_MinMax()
muzero_backup(list(root, child), value = 1.0, gamma = 0.99, mm)
c(root$visits, child$visits)
}
