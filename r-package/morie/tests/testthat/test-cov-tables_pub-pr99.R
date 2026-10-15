# R/tables_pub.R, the PR #99 line the pr77 file leaves: the Type I table is stats::anova's.

test_that("a Type I ANOVA table is stats::anova's, sequential sums of squares", {
  set.seed(1)
  d <- data.frame(x = stats::rnorm(30), g = gl(3, 10))
  d$y <- d$x + as.numeric(d$g) + stats::rnorm(30)
  m <- stats::lm(y ~ x * g, data = d)
  ref <- stats::anova(m)
  out <- anova_table(m, typ = 1L)
  expect_identical(rownames(out), rownames(ref))
  expect_identical(out[["Sum Sq"]], formatC(ref[["Sum Sq"]], format = "f", digits = 3L))
  expect_identical(out[["Df"]], as.character(ref[["Df"]]))
  fv <- ref[["F value"]]
  expect_identical(out[["F value"]], ifelse(is.finite(fv), formatC(fv, format = "f", digits = 3L), ""))
  # sequential: the terms' sums of squares and the residual add to the total sum of squares
  expect_equal(sum(ref[["Sum Sq"]]), sum((d$y - mean(d$y))^2))
})
