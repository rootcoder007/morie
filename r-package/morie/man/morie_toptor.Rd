% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_toptor}
\alias{morie_toptor}
\title{Topological torsions for a molecule, or for a set of molecules}
\usage{
morie_toptor(elements, bonds, reference = NULL, common_types = NULL, activities = NULL, permutations = 40, seed = 0)
}
\arguments{
\item{elements}{Heavy-atom element symbols, either one molecule\'s   or a list of several.}
\item{bonds}{Matching bond list(s).}
\item{reference}{Optional \code{list(elements, bonds)} of a probe   molecule for the similarity probe.}
\item{common_types}{Optional element whitelist (see   \code{morie_topological_torsions}).}
\item{activities}{Optional numeric activity vector for the trend   vector.}
\item{permutations}{Number of permutations for the randomisation   test.}
\item{seed}{Seed for the shared generator.}
}
\description{
Single-molecule torsions with multiplicity counts, or a list of such dictionaries, with optional similarity ranking against a reference and an optional trend vector with randomisation test.
}
\value{
A list with \code{estimate}/\code{torsions},   \code{n_distinct}, \code{n_total}, and the optional   \code{similarity}, \code{ranking}, \code{trend}.
}
\keyword{internal}
\examples{
V <- c(1, 2, 3, 4, 5, 6, 7, 8)
D <- data.frame(x = c(1, 2, 3, 4), y = c(2, 4, 5, 9))
morie_toptor(V, D)
}
