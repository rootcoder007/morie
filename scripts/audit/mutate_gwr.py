"""Mutation test: each mutant must be caught by schab_gwr_verify.py."""

import shutil
import subprocess
import sys
import tempfile

SRC = "src/morie/fn/_schab_gwr.py"
VERIFY = "scripts/audit/schab_gwr_verify.py"

MUTANTS = [
    ("AICc denominator n-2-trS -> n-1-trS", "denom = n - 2.0 - tr_S", "denom = n - 1.0 - tr_S"),
    (
        "AICc log(sqrt(sigma2)) -> log(sigma2)",
        "return 2.0 * n * np.log(np.sqrt(sigma2)) + n * np.log(2.0 * np.pi) + n * (n + tr_S) / denom",
        "return 2.0 * n * np.log(sigma2) + n * np.log(2.0 * np.pi) + n * (n + tr_S) / denom",
    ),
    ("bisquare exponent 2 -> 1", "(1.0 - z * z) ** 2, 0.0)", "(1.0 - z * z) ** 1, 0.0)"),
    ("tricube inner power 3 -> 2", "(1.0 - z**3) ** 3, 0.0)", "(1.0 - z**2) ** 3, 0.0)"),
    ("CV stops leaving one out", "        w[i] = 0.0\n", "        w[i] = w[i]\n"),
    ("adaptive bandwidth off by one", "return float(d[k - 1] * eps)", "return float(d[k] * eps)"),
    (
        "MGWR drops the residual threading",
        "            temp_y = XB[:, [j]] + err",
        "            temp_y = XB[:, [j]] + 0.0",
    ),
]


def main():
    original = open(SRC).read()
    backup = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    backup.write(original)
    backup.close()
    caught = 0
    try:
        for name, old, new in MUTANTS:
            if original.count(old) != 1:
                print(f"  {name:<48} SKIP  anchor appears {original.count(old)}x")
                continue
            open(SRC, "w").write(original.replace(old, new))
            r = subprocess.run([sys.executable, VERIFY], capture_output=True, text=True)
            died = r.returncode != 0
            caught += died
            tag = "CAUGHT" if died else "SURVIVED"
            fails = [ln for ln in r.stdout.splitlines() if " FAIL " in ln]
            print(f"  {name:<48} {tag}  ({len(fails)} checks failed)")
            if not died:
                print("    !! mutant survived -- the checks do not pin this line")
    finally:
        open(SRC, "w").write(original)
        shutil.copy(backup.name, backup.name + ".kept")
    print(f"\n{caught}/{len(MUTANTS)} mutants caught")
    assert open(SRC).read() == original, "restore failed"
    print("source restored byte-for-byte")
    return 0 if caught == len(MUTANTS) else 1


if __name__ == "__main__":
    sys.exit(main())
