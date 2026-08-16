% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.morie_mapq_synth_panel}
\alias{.morie_mapq_synth_panel}
\title{Deterministic synthetic MAPQII panel with a planted 4-factor}
\usage{
.morie_mapq_synth_panel(n = 400L, seed = 2026L)
}
\arguments{
\item{n}{Passed to \code{sample}. Defaults to \code{400L}.}
\item{seed}{Passed to \code{set.seed}. Defaults to \code{2026L}.}
}
\description{
structure, plus gender/age demographics and a Knowledge Scale (KS) driven by epistemic attitudes and a modest gender gap -- so the DML stage has a real signal to recover.
}
\value{
The value of \code{panel}, as built in the body.
}
\keyword{internal}
\examples{
.morie_mapq_synth_panel()
}
