% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{morie_topological_torsions}
\alias{morie_topological_torsions}
\title{Every topological torsion in a molecule, with multiplicities}
\usage{
morie_topological_torsions(elements, bonds, common_types = NULL)
}
\arguments{
\item{elements}{Character vector of heavy-atom element symbols.}
\item{bonds}{List of bonds; each entry is a length-2 or length-3   vector \code{c(i, j)} or \code{c(i, j, order)}.}
\item{common_types}{Optional character vector of element symbols   kept as themselves; everything else becomes \code{"Y"}.}
}
\description{
Returns a list keyed by the canonical 4-tuple of \code{(NPI, TYPE, NBR)} codes; values are the multiplicity of each.
}
\value{
A named list mapping canonical codes to integer counts.
}
\keyword{internal}
