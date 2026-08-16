% Generated from the package source by gen_rd_tree.R -- roxygen2
% hangs on this package. Usage comes from args(), so it tracks
% the formals.
\name{.siu_html_to_text}
\alias{.siu_html_to_text}
\title{Internal: R-side HTML-to-text helper. Strips tags + decodes the}
\usage{
.siu_html_to_text(h)
}
\arguments{
\item{h}{See Usage.}
}
\description{
most common entities so reports + news releases can be displayed as plain text. Mirrors the C++ html_to_text() but with the safer linear single-pass approach (no std::regex backtracking risk).
}
\value{
The value of \code{trimws}.
}
\keyword{internal}
