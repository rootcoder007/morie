% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.siu_p_section_text}
\alias{.siu_p_section_text}
\title{.siu_p_section_text}
\usage{
.siu_p_section_text(text, header, end_markers = character())
}
\arguments{
\item{text}{A vector; its length is taken.}
\item{header}{Passed to \code{.siu_p_re_escape}.}
\item{end_markers}{Defaults to \code{character()}.}
}
\description{
Section slicer -- pulls text between `header` and the first end-marker. Pure-R equivalent of _section_text. ---------------------------------------------------------------------------
}
\value{
The value of \code{substr}.
}
\keyword{internal}
