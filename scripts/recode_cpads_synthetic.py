# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recode src/morie/data/cpads_synthetic.csv to canonical CPADS PUMF codes.

The original synthetic fixture carried string codings for the demographic
columns (gender "female"/"male"/"other", age_group "15-24"..., province
"BC"/"Prairies"/...). The R module family — and the sibling fixture in
rmoriedata — use the CPADS PUMF numeric codes:

    gender          1=Female 2=Male 3=Non-binary
    age_group       1=16-19  2=20-22 3=23-25 4=26+
    province_region 1=Atlantic 2=Quebec 3=Ontario 4=Western

The string age brackets do not map one-to-one onto the youth brackets, so
"15-24" is split uniformly (seeded, reproducible) across codes 1-3 and the
older brackets collapse to 4. Everything else is deterministic. Run from
the repo root:

    python scripts/recode_cpads_synthetic.py
"""

import csv
import random
from pathlib import Path

CSV = Path(__file__).resolve().parents[1] / "src" / "morie" / "data" / "cpads_synthetic.csv"

GENDER = {"female": "1", "male": "2", "other": "3"}
PROVINCE = {"Atlantic": "1", "Quebec": "2", "Ontario": "3", "BC": "4", "Prairies": "4"}


def age_code(label: str, rng: random.Random) -> str:
    if label == "15-24":
        return str(rng.choice([1, 2, 3]))
    return "4"  # 25-44 / 45-64 / 65+


def main() -> None:
    rng = random.Random(91735246)  # the family's canonical fixture seed
    with CSV.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
        fieldnames = fh and list(rows[0].keys())
    for row in rows:
        row["gender"] = GENDER[row["gender"]]
        row["province_region"] = PROVINCE[row["province_region"]]
        row["age_group"] = age_code(row["age_group"], rng)
    with CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"recoded {len(rows)} rows -> {CSV}")


if __name__ == "__main__":
    main()
