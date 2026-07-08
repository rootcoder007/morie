/* SPDX-License-Identifier: AGPL-3.0-or-later
 *
 * test_c_api.c -- a pure-C consumer of libmorie_core.
 *
 * This is the proof that the C ABI actually works: it is compiled as C
 * (not C++), links the shared/static libmorie_core, and asserts the core's
 * numbers against hand-computed values. If the façade or the core breaks,
 * this fails to build or aborts. Run via `ctest` (see CMakeLists.txt).
 */
#include "morie_c_api.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static int close(double a, double b) { return fabs(a - b) < 1e-9; }

int main(void) {
    const double x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    const double y[5] = {2.0, 4.0, 6.0, 8.0, 10.0}; /* perfectly correlated */

    /* mean = 3 */
    assert(close(morie_mean(x, 5), 3.0));

    /* population variance (ddof=0) of 1..5 = 2.0; sample (ddof=1) = 2.5 */
    assert(close(morie_variance(x, 5, 0), 2.0));
    assert(close(morie_variance(x, 5, 1), 2.5));
    assert(close(morie_stddev(x, 5, 0), sqrt(2.0)));

    /* y = 2x -> Pearson r = 1 */
    assert(close(morie_cor_pearson(x, y, 5), 1.0));

    /* euclid dist between x and y = sqrt(sum (x-y)^2) */
    double sq = 0.0;
    for (int i = 0; i < 5; ++i) {
        double d = x[i] - y[i];
        sq += d * d;
    }
    assert(close(morie_euclid_dist(x, y, 5), sqrt(sq)));

    /* normal_pdf at the mean of N(0,1) = 1/sqrt(2pi) ~= 0.39894228 */
    double xin[1] = {0.0};
    double out[1];
    morie_normal_pdf(xin, 1, 0.0, 1.0, out);
    assert(close(out[0], 0.3989422804014327));

    /* logpdf consistency: log(pdf) == logpdf */
    double lout[1];
    morie_normal_logpdf(xin, 1, 0.0, 1.0, lout);
    assert(close(lout[0], log(out[0])));

    /* gamma CDF sanity: P(a,0)=0, monotone, in [0,1] */
    assert(close(morie_gamma_cdf_regularized(2.0, 0.0), 0.0));
    double g1 = morie_gamma_cdf_regularized(2.0, 1.0);
    double g2 = morie_gamma_cdf_regularized(2.0, 5.0);
    assert(g1 > 0.0 && g1 < 1.0 && g2 > g1 && g2 < 1.0);

    /* degenerate input -> NaN, not crash */
    assert(isnan(morie_mean(x, 0)));

    printf("OK  %s\n", morie_core_version());
    printf("PASS: all morie C ABI assertions held\n");
    return 0;
}
