% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.entheo_extract_pair}
\alias{.entheo_extract_pair}
\title{.entheo_extract_pair}
\usage{
.entheo_extract_pair(record_or_eeg, fmri)
}
\arguments{
\item{record_or_eeg}{A list; the body reads \code{$fmri} from it.}
\item{fmri}{Carried through into a list the body builds.}
}
\description{
A step of the entheo_analysis implementation. Called by \code{beautiful_loop_metric}, \code{san_score}. See the file header for the source the module follows.
}
\value{
A list with \code{e_dmt}, \code{f_dmt}, \code{e_pcb}, \code{f_pcb}.
}
\keyword{internal}
