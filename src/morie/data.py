import functools
import json
import logging
import os
import re
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen

from morie.fn import _frame_core as pd

from .cpads import (
    CPADS_REQUIRED_VARIABLES,
    canonicalize_cpads_frame,
    cpads_contract,
    has_raw_cpads_columns,
    infer_file_format,
    validate_cpads_frame,
)

_SAFE_TABLE_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _safe_table_name(name: str) -> str:
    """Validate a SQLite table name contains only safe characters."""
    if not _SAFE_TABLE_RE.match(name):
        raise ValueError(f"Unsafe table name: {name!r}")
    return name


logger = logging.getLogger(__name__)

DEFAULT_CKAN_API_BASE = "https://open.canada.ca/data/en/api/3/action/datastore_search"
DEFAULT_CACHE_DB = "morie.db"


def _user_cache_dir() -> Path:
    """Per-user cache directory for morie, portable across machines.

    Honours ``XDG_CACHE_HOME``; otherwise ``~/.cache/morie``. The
    SQLite cache and on-demand fetched datasets live here -- it is
    always user-writable and never depends on the install location.
    """
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base).expanduser() / "morie"


def _project_root() -> Path:
    """Best-effort repository root for a source checkout.

    Walks up from this file looking for a ``pyproject.toml`` marker.
    When morie runs from a source checkout this finds the repo root
    (used only to resolve the relative ``local_path`` of author-local
    datasets). For an installed package there is no such root, so the
    current working directory is returned as a neutral base.

    The cache never relies on this -- see :func:`_user_cache_dir` --
    so a wrong answer here cannot misplace user data.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


# ---------------------------------------------------------------------------
# CKAN dataset catalogue -- Canadian public health open data
# ---------------------------------------------------------------------------

CKAN_DATASETS: dict[str, dict[str, str]] = {
    "cpads": {
        "name": "CPADS 2021-2022 PUMF",
        "package_id": "736fa9b2-62e4-4e31-aea4-51869605b363",
        "resource_id": "d2639429-c304-45a6-90b3-770562f4d46d",
        "metadata_url": "https://open.canada.ca/data/api/action/package_show?id=736fa9b2-62e4-4e31-aea4-51869605b363",
    },
    "csads": {
        "name": "CSADS",
        "package_id": "1f15ca45-8bfd-4f9c-9ec6-2c0c440e69c2",
        "resource_id": "f6761337-47e9-455a-a3c4-ea8516aa634f",  # 2021-2022 CSTADS PUMF CSV
        "metadata_url": "https://open.canada.ca/data/api/action/package_show?id=1f15ca45-8bfd-4f9c-9ec6-2c0c440e69c2",
    },
    "csus": {
        "name": "CSUS",
        "package_id": "65e2d45e-efc6-4c29-9a9b-db59bc96aa0e",
        "resource_id": "c2c1795b-4501-49ba-9dd1-5b8360cc3b2e",  # 2023 CSUS PUMF CSV (verified via package_show 2026-04-18)
        "metadata_url": "https://open.canada.ca/data/api/action/package_show?id=65e2d45e-efc6-4c29-9a9b-db59bc96aa0e",
    },
}


# ---------------------------------------------------------------------------
# Full dataset catalog -- every file in data/datasets/
# ---------------------------------------------------------------------------

DATASET_CATALOG: dict[str, dict] = {
    # ── OpenCanada (oc) PUMF microdata ────────────────────────
    # rmoriedata (CRAN) slugs: the reviewed SIU corpus and manifest, read
    # straight from the CRAN source tarball (fetched once, no R needed).
    "siu": {
        "name": "SIU director's reports (reviewed corpus, rmoriedata)",
        "source": "rmoriedata",
        "survey": "siu",
        "year": "2005-2026",
        "format": "csv",
        "type": "oversight",
        "large_file": False,
        "local_path": "data/datasets/siu/siu_directors_reports.csv",
        "table_name": "siu",
        "ckan_resource_id": "",
        "rmoriedata": "siu_directors_reports",
    },
    "siumanifest": {
        "name": "SIU drid manifest (rmoriedata)",
        "source": "rmoriedata",
        "survey": "siu",
        "year": "2005-2026",
        "format": "csv",
        "type": "oversight",
        "large_file": False,
        "local_path": "data/datasets/siu/siu_drid_manifest.csv",
        "table_name": "siumanifest",
        "ckan_resource_id": "",
        "rmoriedata": "siu_drid_manifest",
    },
    "ocp21": {
        "name": "CPADS 2021-2022 PUMF",
        "source": "oc",
        "survey": "cpads",
        "year": "2021-2022",
        "format": "csv",
        "type": "pumf",
        "large_file": False,
        "local_path": "data/datasets/oc/CPADS/2021-2022/cpads-2021-2022-pumf2.csv",
        "table_name": "ocp21",
        "ckan_resource_id": "d2639429-c304-45a6-90b3-770562f4d46d",
    },
    # ── Statistics Canada direct-download PUMF ────────────────
    "cchs22": {
        "name": "CCHS 2022 PUMF (Canadian Community Health Survey)",
        "source": "statcan",
        "survey": "cchs",
        "year": "2022",
        "format": "fetcher",
        "type": "pumf",
        "large_file": True,
        "local_path": "data/datasets/statcan/CCHS/2022/cchs-2022-pumf.csv",
        "table_name": "cchs22",
        "ckan_resource_id": "",
        "fetcher": "morie.ingest.statcan:fetch_statcan_csv",
        "fetcher_args": {"url": "https://www150.statcan.gc.ca/n1/pub/82m0013x/2024001/2022_CSV.zip"},
    },
    "occ22": {
        "name": "CCS 2018-2022 PUMF",
        "source": "oc",
        "survey": "ccs",
        "year": "2018-2022",
        "format": "csv",
        "type": "pumf",
        "large_file": False,
        "local_path": "data/datasets/oc/CCS/2018-2022/ccs_pumf_2018to2022_final.csv",
        "table_name": "occ22",
        "ckan_resource_id": "262e6163-ba41-4562-bd2b-8996e738b1d4",
    },
    "occ23": {
        "name": "CCS 2023 PUMF",
        "source": "oc",
        "survey": "ccs",
        "year": "2023",
        "format": "csv",
        "type": "pumf",
        "large_file": False,
        "local_path": "data/datasets/oc/CCS/2023/ccs_2023_pumf.csv",
        "table_name": "occ23",
        "ckan_resource_id": "100c5845-664e-4c66-be15-3625ce236d8b",
    },
    "occ24": {
        "name": "CCS 2024 PUMF",
        "source": "oc",
        "survey": "ccs",
        "year": "2024",
        "format": "csv",
        "type": "pumf",
        "large_file": False,
        "local_path": "data/datasets/oc/CCS/2024/ccs_pumf_2024-002.csv",
        "table_name": "occ24",
        "ckan_resource_id": "420925be-399a-473b-8be3-26875a1c132a",
    },
    "ocs22mf": {
        "name": "CSADS 2021-2022 PUMF",
        "source": "oc",
        "survey": "csads",
        "year": "2021-2022",
        "format": "csv",
        "type": "pumf",
        "large_file": False,
        "local_path": "data/datasets/oc/CSADS/2021-2022/csads202122pumf.csv",
        "table_name": "ocs22mf",
        "ckan_resource_id": "f6761337-47e9-455a-a3c4-ea8516aa634f",
    },
    "ocs22bt": {
        "name": "CSADS 2021-2022 Bootstrap",
        "source": "oc",
        "survey": "csads",
        "year": "2021-2022",
        "format": "csv",
        "type": "bootstrap",
        "large_file": True,
        "local_path": "data/datasets/oc/CSADS/2021-2022/csads202122bootstrap.csv",
        "table_name": "ocs22bt",
        "ckan_resource_id": "ebdc36e1-910d-4685-81a3-6acfe44729bc",
    },
    "ocs24mf": {
        "name": "CSADS 2023-2024 PUMF",
        "source": "oc",
        "survey": "csads",
        "year": "2023-2024",
        "format": "csv",
        "type": "pumf",
        "large_file": False,
        "local_path": "data/datasets/oc/CSADS/2023-2024/csads202324pumf.csv",
        "table_name": "ocs24mf",
        "ckan_resource_id": "81a3adf0-61d0-4691-afba-588fa5f563da",
    },
    "ocs24bt": {
        "name": "CSADS 2023-2024 Bootstrap",
        "source": "oc",
        "survey": "csads",
        "year": "2023-2024",
        "format": "csv",
        "type": "bootstrap",
        "large_file": True,
        "local_path": "data/datasets/oc/CSADS/2023-2024/csads202324bootstrap.csv",
        "table_name": "ocs24bt",
        "ckan_resource_id": "58682536-1325-405a-83f0-7b1284b4f717",
    },
    "cu20mf": {
        "name": "CSUS 2019-2020 PUMF",
        "source": "oc",
        "survey": "csus",
        "year": "2019-2020",
        "format": "csv",
        "type": "pumf",
        "large_file": False,
        "local_path": "data/datasets/oc/CSUS/2019-2020/CADS201920pumf.csv",
        "zip_member": "CADS.csv",
        "table_name": "cu20mf",
        "ckan_resource_id": "0f4c0418-b9d1-4f89-a917-a660d20fd6d0",
    },
    "cu20bt": {
        "name": "CSUS 2019-2020 Bootstrap",
        "source": "oc",
        "survey": "csus",
        "year": "2019-2020",
        "format": "csv",
        "type": "bootstrap",
        "large_file": True,
        "local_path": "data/datasets/oc/CSUS/2019-2020/CADS201920bsw.csv",
        "zip_member": "CADS_bsw.csv",
        "table_name": "cu20bt",
        "ckan_resource_id": "0f4c0418-b9d1-4f89-a917-a660d20fd6d0",
    },
    "cu23mf": {
        "name": "CSUS 2023 PUMF",
        "source": "oc",
        "survey": "csus",
        "year": "2023",
        "format": "csv",
        "type": "pumf",
        "large_file": False,
        "local_path": "data/datasets/oc/CSUS/2023/csus2023_pumf_final.csv",
        "table_name": "cu23mf",
        "ckan_resource_id": "c2c1795b-4501-49ba-9dd1-5b8360cc3b2e",
    },
    "cu23bt": {
        "name": "CSUS 2023 Bootstrap",
        "source": "oc",
        "survey": "csus",
        "year": "2023",
        "format": "csv",
        "type": "bootstrap",
        "large_file": True,
        "local_path": "data/datasets/oc/CSUS/2023/csus2023_pumf_bwt.csv",
        "table_name": "cu23bt",
        "ckan_resource_id": "7d19d47a-5f42-4447-b735-aa4d677ad5ed",
    },
    # ── HealthInfobase (hib) aggregate ────────────────────────
    "hibp": {
        "name": "CPADS Aggregate",
        "source": "hib",
        "survey": "cpads",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CPADS/CPADS.csv",
        "hosted_key": "hib/cpads_cpads",
        "table_name": "hibp",
        "ckan_resource_id": "",
    },
    "hibsa": {
        "name": "CSADS Provinces",
        "source": "hib",
        "survey": "csads",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSADS/provinces.csv",
        "download_url": "https://health-infobase.canada.ca/src/data/csads/downloadable/CSADS-data.zip",
        "zip_member": "provinces.csv",
        "hosted_key": "hib/csads_provinces",
        "table_name": "hibsa",
        "ckan_resource_id": "",
    },
    "hibsb": {
        "name": "CSADS Trends",
        "source": "hib",
        "survey": "csads",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSADS/trends.csv",
        "download_url": "https://health-infobase.canada.ca/src/data/csads/downloadable/CSADS-data.zip",
        "zip_member": "trends.csv",
        "hosted_key": "hib/csads_trends",
        "table_name": "hibsb",
        "ckan_resource_id": "",
    },
    "hibua": {
        "name": "CSUS Alcohol",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Alcohol.csv",
        "download_url": "https://health-infobase.canada.ca/src/data/csus/CADS_data.zip",
        "zip_member": "Alcohol.csv",
        "hosted_key": "hib/csus_alcohol",
        "table_name": "hibua",
        "ckan_resource_id": "",
    },
    "hibub": {
        "name": "CSUS Cannabis",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Cannabis.csv",
        "download_url": "https://health-infobase.canada.ca/src/data/csus/CADS_data.zip",
        "zip_member": "Cannabis.csv",
        "hosted_key": "hib/csus_cannabis",
        "table_name": "hibub",
        "ckan_resource_id": "",
    },
    "hibuc": {
        "name": "CSUS Smoking & Vaping",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Cigarette smoking and vaping.csv",
        "download_url": "https://health-infobase.canada.ca/src/data/csus/CADS_data.zip",
        "zip_member": "Cigarette smoking and vaping.csv",
        "hosted_key": "hib/csus_cigarette_smoking_and_vaping",
        "table_name": "hibuc",
        "ckan_resource_id": "",
    },
    "hibud": {
        "name": "CSUS Illegal Substances",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Illegal substances.csv",
        "download_url": "https://health-infobase.canada.ca/src/data/csus/CADS_data.zip",
        "zip_member": "Illegal substances.csv",
        "hosted_key": "hib/csus_illegal_substances",
        "table_name": "hibud",
        "ckan_resource_id": "",
    },
    "hibue": {
        "name": "CSUS Opioids",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Opioids.csv",
        "download_url": "https://health-infobase.canada.ca/src/data/csus/CADS_data.zip",
        "zip_member": "Opioids.csv",
        "hosted_key": "hib/csus_opioids",
        "table_name": "hibue",
        "ckan_resource_id": "",
    },
    "hibuf": {
        "name": "CSUS OTC Products",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Over the counter products.csv",
        "download_url": "https://health-infobase.canada.ca/src/data/csus/CADS_data.zip",
        "zip_member": "Over the counter products.csv",
        "hosted_key": "hib/csus_over_the_counter_products",
        "table_name": "hibuf",
        "ckan_resource_id": "",
    },
    "hibug": {
        "name": "CSUS Polysubstance",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Polysubstance.csv",
        "download_url": "https://health-infobase.canada.ca/src/data/csus/CADS_data.zip",
        "zip_member": "Polysubstance.csv",
        "hosted_key": "hib/csus_polysubstance",
        "table_name": "hibug",
        "ckan_resource_id": "",
    },
    "hibuh": {
        "name": "CSUS Sedatives",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Sedatives.csv",
        "download_url": "https://health-infobase.canada.ca/src/data/csus/CADS_data.zip",
        "zip_member": "Sedatives.csv",
        "hosted_key": "hib/csus_sedatives",
        "table_name": "hibuh",
        "ckan_resource_id": "",
    },
    "hibui": {
        "name": "CSUS Stimulants",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Stimulants.csv",
        "download_url": "https://health-infobase.canada.ca/src/data/csus/CADS_data.zip",
        "zip_member": "Stimulants.csv",
        "hosted_key": "hib/csus_stimulants",
        "table_name": "hibui",
        "ckan_resource_id": "",
    },
    "hibuj": {
        "name": "CSUS Substance Use Harms",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Substance use harms.csv",
        "download_url": "https://health-infobase.canada.ca/src/data/csus/CADS_data.zip",
        "zip_member": "Substance use harms.csv",
        "hosted_key": "hib/csus_substance_use_harms",
        "table_name": "hibuj",
        "ckan_resource_id": "",
    },
    "hibuk": {
        "name": "CSUS Treatment",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Treatment.csv",
        "download_url": "https://health-infobase.canada.ca/src/data/csus/CADS_data.zip",
        "zip_member": "Treatment.csv",
        "hosted_key": "hib/csus_treatment",
        "table_name": "hibuk",
        "ckan_resource_id": "",
    },
    # ── CIHI indicator library ────────────────────────────────────────
    # ── CIHI (cihi) indicator library ───────────────────────────
    "cihidt": {
        "name": "CIHI All Indicators",
        "source": "cihi",
        "survey": "indicators",
        "year": "",
        "format": "xlsx",
        "type": "indicator",
        "large_file": False,
        "local_path": "data/datasets/cihi/indicator-library-all-indicator-data-en.xlsx",
        "fetcher": "morie.data:fetch_cihi_indicator_library",
        "table_name": "cihidt",
        "ckan_resource_id": "",
    },
    "cihi820a": {
        "name": "CIHI 820: Substance Use Harm",
        "source": "cihi",
        "survey": "indicators",
        "year": "",
        "format": "xlsx",
        "type": "indicator",
        "large_file": False,
        "local_path": "data/datasets/cihi/820/820-hospital-stays-for-harm-caused-by-substance-use-data-table-en.xlsx",
        "table_name": "cihi820a",
        "ckan_resource_id": "",
        "fetcher": "morie.ingest.cihi:fetch_cihi_xlsx",
        "fetcher_args": {
            "url": "https://www.cihi.ca/sites/default/files/document/data-file/820-hospital-stays-for-harm-caused-by-substance-use-data-table-en.xlsx"
        },
    },
    "cihi820b": {
        "name": "CIHI 820: Substance Use Breakdown 2024-2025",
        "source": "cihi",
        "survey": "indicators",
        "year": "2024-2025",
        "format": "xlsx",
        "type": "indicator",
        "large_file": False,
        "local_path": (
            "data/datasets/cihi/820/"
            "820-hospital-stays-harm-due-to-substance-use-breakdown-2024-2025-data-tables-en-additional.xlsx"
        ),
        "table_name": "cihi820b",
        "ckan_resource_id": "",
        "fetcher": "morie.ingest.cihi:fetch_cihi_xlsx",
        "fetcher_args": {
            "url": "https://www.cihi.ca/sites/default/files/document/hospital-stays-harm-due-to-substance-use-breakdown-2024-2025-data-tables-en.xlsx"
        },
    },
    "cihi849": {
        "name": "CIHI 849: Alcohol Use Harm",
        "source": "cihi",
        "survey": "indicators",
        "year": "",
        "format": "xlsx",
        "type": "indicator",
        "large_file": False,
        "local_path": "data/datasets/cihi/849/849-hospital-stays-for-harm-caused-by-alcohol-use-data-table-en.xlsx",
        "table_name": "cihi849",
        "ckan_resource_id": "",
        "fetcher": "morie.ingest.cihi:fetch_cihi_xlsx",
        "fetcher_args": {
            "url": "https://www.cihi.ca/sites/default/files/document/data-file/849-hospital-stays-for-harm-caused-by-alcohol-use-data-table-en.xlsx"
        },
    },
    "cihi885a": {
        "name": "CIHI 885: Youth Services",
        "source": "cihi",
        "survey": "indicators",
        "year": "",
        "format": "xlsx",
        "type": "indicator",
        "large_file": False,
        "local_path": (
            "data/datasets/cihi/885/"
            "885-youth-age-12-to-25-who-accessed-integrated-youth-services-for-mental-health"
            "-substance-use-and-well-being-support-data-table-en.xlsx"
        ),
        "table_name": "cihi885a",
        "ckan_resource_id": "",
        "fetcher": "morie.ingest.cihi:fetch_cihi_xlsx",
        "fetcher_args": {
            "url": "https://www.cihi.ca/sites/default/files/document/data-file/885-youth-age-12-to-25-who-accessed-integrated-youth-services-for-mental-health-substance-use-and-well-being-support-data-table-en.xlsx"
        },
    },
    "cihi885b": {
        "name": "CIHI 885: Youth Sites 2024-2025",
        "source": "cihi",
        "survey": "indicators",
        "year": "2024-2025",
        "format": "xlsx",
        "type": "indicator",
        "large_file": False,
        "local_path": (
            "data/datasets/cihi/885/885-number-integrated-youth-services-sites-2024-2025-data-tables-en-additonal.xlsx"
        ),
        "table_name": "cihi885b",
        "ckan_resource_id": "",
        "fetcher": "morie.ingest.cihi:fetch_cihi_xlsx",
        "fetcher_args": {
            "url": "https://www.cihi.ca/sites/default/files/document/integrated-youth-services-sites-2024-2025-data-tables-en.xlsx"
        },
    },
    # ── VSR Research Data ─────────────────────────────────────
    "mapq": {
        "name": "MAPQ: Modified Attitudes on Psychedelics Questionnaire",
        "source": "vsr",
        "survey": "mapq",
        "year": "2026",
        "format": "xlsx",
        "type": "psychometric",
        "large_file": False,
        "local_path": "data/datasets/vsr/TKARONTOMAPQ.xlsx",
        "table_name": "mapq",
        "ckan_resource_id": "",
        "sheets": {
            "MAPQII": "20-item Likert scale (EE/EA/UA/ER, 4 subscales)",
            "MAPQ + KS + KnAcqS": "MAPQ subscales + Knowledge Scale + demographics",
            "recode": "Original recoded APQ items",
            "KS test items": "Knowledge Scale drug identification items",
        },
    },
    "otis": {
        "name": "OTIS: Ontario Restrictive Confinement 2023-2025",
        "source": "vsr",
        "survey": "otis",
        "year": "2023-2025",
        "format": "rdata",
        "type": "correctional",
        "large_file": False,
        "local_path": "data/cache/correctional_stats_report_environment1b.RData",
        "hosted_file": "otis/correctional_stats_report_environment1b.RData",
        "table_name": "otis",
        "ckan_resource_id": "",
    },
    "otisexp": {
        "name": "OTIS Expanded (1.9M placement records)",
        "source": "vsr",
        "survey": "otis",
        "year": "2023-2025",
        "format": "rds",
        "type": "correctional",
        "large_file": True,
        "local_path": "data/cache/dt_expanded.rds",
        "hosted_file": "otis/dt_expanded.rds",
        "table_name": "otisexp",
        "ckan_resource_id": "",
    },
    "otisfin": {
        "name": "OTIS Complete Analysis Environment (239 objects)",
        "source": "vsr",
        "survey": "otis",
        "year": "2023-2025",
        "format": "rdata",
        "type": "correctional",
        "large_file": True,
        "local_path": "data/cache/finne_env.RData",
        "hosted_file": "otis/finne_env.RData",
        "table_name": "otisfin",
        "ckan_resource_id": "",
    },
    "otisloc": {
        "name": "OTIS: Ontario's adult provincial correctional institutions (locations)",
        "source": "otis",
        "survey": "otis",
        "year": "",
        "format": "csv",
        "type": "correctional",
        "large_file": False,
        "local_path": "data/datasets/otis/institutional_locations_en.csv",
        "download_url": "https://data.ontario.ca/dataset/3ca4505b-091c-4b04-89e8-c316ffaa0d9e/resource/97d82317-539c-479d-9479-4dd9b7e9e08c/download/institutional_locations_en.csv",
        "table_name": "otisloc",
        "ckan_resource_id": "",
    },
    # ── OTIS public release per-table CSVs (used by morie.mrm_otis_*) ──
    # CKAN resource IDs are from data.ontario.ca/dataset/data-on-inmates-in-ontario;
    # fetched on first use by the existing morie CKAN downloader.
    "otisb01": {
        "name": "OTIS b01: Segregation – Detailed Dataset",
        "source": "otis",
        "survey": "b01",
        "year": "2023-2025",
        "format": "csv",
        "type": "correctional",
        "large_file": False,
        "local_path": "data/datasets/OTIS/b01_segregation_detailed_dataset.csv",
        "table_name": "otisb01",
        "ckan_resource_id": "406e6d90-d568-4553-8ca7-bc9f90e133b9",
    },
    "otisb09": {
        "name": "OTIS b09: Individuals in Segregation - Number of Placements",
        "source": "otis",
        "survey": "b09",
        "year": "2023-2025",
        "format": "csv",
        "type": "correctional",
        "large_file": False,
        "local_path": "data/datasets/OTIS/b09_individuals_in_segregation_number_of_times_in_segregation.csv",
        "table_name": "otisb09",
        "ckan_resource_id": "df24e943-d52b-43a8-a10e-a3cc906e26bb",
    },
    "otisc11": {
        "name": "OTIS c11: Individuals in Segregation/RC by Aggregate Length",
        "source": "otis",
        "survey": "c11",
        "year": "2023-2025",
        "format": "csv",
        "type": "correctional",
        "large_file": False,
        "local_path": "data/datasets/OTIS/c11_individuals_in_segregation_and_restrictive_confinement_aggregate_lengths.csv",
        "table_name": "otisc11",
        "ckan_resource_id": "9c7b74a5-53ad-4ef0-a7a6-97772cd01c55",
    },
    "otisa01": {
        "name": "OTIS a01: Restrictive Confinement – Detailed Dataset",
        "source": "otis",
        "survey": "a01",
        "year": "2023-2025",
        "format": "csv",
        "type": "correctional",
        "large_file": False,
        "local_path": "data/datasets/OTIS/a01_restrictive_confinement_detailed_dataset.csv",
        "table_name": "otisa01",
        "ckan_resource_id": "5a0c5804-a055-4031-9743-73f556e43bb4",
    },
    # ── Ontario SIU case-level (used by morie.mrm_siu_*) ──
    # ── TPS per-category open-data events (used by morie.mrm_tps_*) ──
    "tpsassault": {
        "name": "TPS Assault open-data events 2014-present",
        "source": "tps",
        "survey": "assault",
        "year": "2014-present",
        "format": "csv",
        "type": "crime",
        "large_file": True,
        "local_path": "data/datasets/TPS/Assault/CSV",
        "table_name": "tpsassault",
        "ckan_resource_id": "",
        "fetcher": "morie.tps_fetch:fetch_tps_dataframe",
        "fetcher_args": {"category": "Assault"},
    },
    "tpshomicides": {
        "name": "TPS Homicides open-data events 2014-present",
        "source": "tps",
        "survey": "homicides",
        "year": "2014-present",
        "format": "csv",
        "type": "crime",
        "large_file": False,
        "local_path": "data/datasets/TPS/Homicides/CSV",
        "table_name": "tpshomicides",
        "ckan_resource_id": "",
        "fetcher": "morie.tps_fetch:fetch_tps_dataframe",
        "fetcher_args": {"category": "Homicides"},
    },
    "tpsshootings": {
        "name": "TPS Shootings and Firearm Discharges 2014-present",
        "source": "tps",
        "survey": "shootings",
        "year": "2014-present",
        "format": "csv",
        "type": "crime",
        "large_file": True,
        "local_path": "data/datasets/TPS/ShootingAndFirearmDiscarges/CSV",
        "table_name": "tpsshootings",
        "ckan_resource_id": "",
        "fetcher": "morie.tps_fetch:fetch_tps_dataframe",
        "fetcher_args": {"category": "ShootingAndFirearmDiscarges"},
    },
    # ── NAPS (Environment Canada National Air Pollution Surveillance) ─────
    # Fetched on demand via morie.earth.fetch_naps (no auth; Open-Canada CKAN).
    # Data lands at ~/.cache/morie/earth/ as parquet (keyed by query hash).
    # local_path kept for compatibility but not used; loader dispatches via
    # the 'fetcher' field when entry['source'] == 'naps'.
    "naps-no2-on-2023": {
        "name": "NAPS NO2 Ontario 2023",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_no2_on_2023",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "no2", "year": 2023, "province": "ON"},
    },
    "naps-pm25-on-2023": {
        "name": "NAPS PM2.5 Ontario 2023",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_pm25_on_2023",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm25", "year": 2023, "province": "ON"},
    },
    "naps-o3-on-2023": {
        "name": "NAPS O3 Ontario 2023",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_o3_on_2023",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "o3", "year": 2023, "province": "ON"},
    },
    "naps-so2-on-2023": {
        "name": "NAPS SO2 Ontario 2023",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_so2_on_2023",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "so2", "year": 2023, "province": "ON"},
    },
    "naps-co-on-2023": {
        "name": "NAPS CO Ontario 2023",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_co_on_2023",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "co", "year": 2023, "province": "ON"},
    },
    "naps-pm10-on-2023": {
        "name": "NAPS PM10 Ontario 2023",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_pm10_on_2023",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm10", "year": 2023, "province": "ON"},
    },
    "naps-no2-on-2022": {
        "name": "NAPS NO2 Ontario 2022 (trend baseline)",
        "source": "naps",
        "survey": "naps",
        "year": "2022",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_no2_on_2022",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "no2", "year": 2022, "province": "ON"},
    },
    # ────────────────────────────────────────────────────────────────
    # W5 extension (2026-04-17 night): other provinces + trend years.
    # Each entry dispatches to morie.earth.fetch_naps via the 'fetcher'
    # field, so adding another province/year is one dict entry and
    # NAPS CKAN auto-caches under ~/.cache/morie/earth/ as parquet.
    # ────────────────────────────────────────────────────────────────
    # --- 2023, other key provinces (PM2.5 + NO2) ---
    "naps-pm25-qc-2023": {
        "name": "NAPS PM2.5 Quebec 2023",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_pm25_qc_2023",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm25", "year": 2023, "province": "QC"},
    },
    "naps-no2-qc-2023": {
        "name": "NAPS NO2 Quebec 2023",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_no2_qc_2023",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "no2", "year": 2023, "province": "QC"},
    },
    "naps-pm25-bc-2023": {
        "name": "NAPS PM2.5 British Columbia 2023 (wildfire-smoke dominant)",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_pm25_bc_2023",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm25", "year": 2023, "province": "BC"},
    },
    "naps-no2-bc-2023": {
        "name": "NAPS NO2 British Columbia 2023",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_no2_bc_2023",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "no2", "year": 2023, "province": "BC"},
    },
    "naps-pm25-ab-2023": {
        "name": "NAPS PM2.5 Alberta 2023",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_pm25_ab_2023",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm25", "year": 2023, "province": "AB"},
    },
    "naps-no2-ab-2023": {
        "name": "NAPS NO2 Alberta 2023",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_no2_ab_2023",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "no2", "year": 2023, "province": "AB"},
    },
    "naps-pm25-ns-2023": {
        "name": "NAPS PM2.5 Nova Scotia 2023",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_pm25_ns_2023",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm25", "year": 2023, "province": "NS"},
    },
    "naps-no2-ns-2023": {
        "name": "NAPS NO2 Nova Scotia 2023",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_no2_ns_2023",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "no2", "year": 2023, "province": "NS"},
    },
    # --- Ontario PM2.5 trend baseline (2019-2022) ---
    "naps-pm25-on-2022": {
        "name": "NAPS PM2.5 Ontario 2022 (trend baseline)",
        "source": "naps",
        "survey": "naps",
        "year": "2022",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_pm25_on_2022",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm25", "year": 2022, "province": "ON"},
    },
    "naps-pm25-on-2021": {
        "name": "NAPS PM2.5 Ontario 2021 (pandemic-year baseline)",
        "source": "naps",
        "survey": "naps",
        "year": "2021",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_pm25_on_2021",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm25", "year": 2021, "province": "ON"},
    },
    "naps-pm25-on-2020": {
        "name": "NAPS PM2.5 Ontario 2020 (COVID-lockdown air quality)",
        "source": "naps",
        "survey": "naps",
        "year": "2020",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_pm25_on_2020",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm25", "year": 2020, "province": "ON"},
    },
    "naps-pm25-on-2019": {
        "name": "NAPS PM2.5 Ontario 2019 (pre-COVID reference)",
        "source": "naps",
        "survey": "naps",
        "year": "2019",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_pm25_on_2019",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm25", "year": 2019, "province": "ON"},
    },
    # --- Ontario O3 trend (for heat × ozone interaction studies) ---
    "naps-o3-on-2022": {
        "name": "NAPS O3 Ontario 2022 (heat × ozone study base)",
        "source": "naps",
        "survey": "naps",
        "year": "2022",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_o3_on_2022",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "o3", "year": 2022, "province": "ON"},
    },
    "naps-o3-on-2021": {
        "name": "NAPS O3 Ontario 2021",
        "source": "naps",
        "survey": "naps",
        "year": "2021",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_o3_on_2021",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "o3", "year": 2021, "province": "ON"},
    },
    # --- 2023 national (no province filter; all-Canada aggregate) ---
    "naps-pm25-ca-2023": {
        "name": "NAPS PM2.5 Canada 2023 (national, all provinces)",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": True,
        "local_path": "",
        "table_name": "naps_pm25_ca_2023",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm25", "year": 2023, "province": None},
    },
    "naps-no2-ca-2023": {
        "name": "NAPS NO2 Canada 2023 (national, all provinces)",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": True,
        "local_path": "",
        "table_name": "naps_no2_ca_2023",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "no2", "year": 2023, "province": None},
    },
    "naps-o3-ca-2023": {
        "name": "NAPS O3 Canada 2023 (national, all provinces)",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": True,
        "local_path": "",
        "table_name": "naps_o3_ca_2023",
        "ckan_resource_id": "",
        "fetcher": "morie.earth:fetch_naps",
        "fetcher_args": {"pollutant": "o3", "year": 2023, "province": None},
    },
}


_METADATA_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS _morie_metadata (
    table_name TEXT PRIMARY KEY,
    source TEXT,
    survey TEXT,
    year TEXT,
    format TEXT,
    row_count INTEGER,
    col_count INTEGER,
    columns TEXT,
    ingested_at TEXT,
    file_hash TEXT
)
"""


# ---------------------------------------------------------------------------
# Built-in database -- ships with the package
# ---------------------------------------------------------------------------


def morie_db() -> Path:
    """Return path to morie.db -- checks package-bundled location first, then cache.

    Package-bundled DB ships with the installed package (via LFS/setuptools).
    Cache location is used for development and is gitignored.
    """
    # 1. Package-bundled DB (ships with morie package, available in CI/install)
    package_db = Path(__file__).parent / "data" / "morie.db"
    if package_db.exists():
        return package_db
    # 2. Per-user cache location (portable; created on first write)
    return _user_cache_dir() / "morie.db"


def _builtin_db_connect() -> sqlite3.Connection | None:
    """Connect to the built-in database if it exists."""
    db_path = morie_db()
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA busy_timeout=5000")
        return conn
    return None


# ---------------------------------------------------------------------------
# SQLite cache -- shared between Python and R via DBI
# ---------------------------------------------------------------------------


def _resolve_cache_path(db_path: str | Path | None = None) -> Path:
    """Resolve the cache database path, creating parent dirs as needed.

    Resolution order: an explicit ``db_path`` argument, else the
    ``MORIE_CACHE_DB`` environment variable, else ``morie.db`` in the
    per-user cache directory. A relative path is taken relative to the
    per-user cache directory (:func:`_user_cache_dir`) -- never to the
    install location -- so the cache works on any machine.
    """
    explicit = db_path if db_path is not None else os.environ.get("MORIE_CACHE_DB")
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = _user_cache_dir() / p
    else:
        p = _user_cache_dir() / DEFAULT_CACHE_DB
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def cache_connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open (or create) the MORIE SQLite cache database."""
    p = _resolve_cache_path(db_path)
    conn = sqlite3.connect(str(p))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(_METADATA_TABLE_DDL)
    return conn


def cache_store(df: pd.DataFrame, table: str, db_path: str | Path | None = None) -> int:
    """Write a DataFrame to the SQLite cache, replacing any existing table."""
    conn = cache_connect(db_path)
    try:
        try:
            df.to_sql(table, conn, if_exists="replace", index=False)
        except Exception as exc:  # noqa: BLE001
            # a second process caching the same key at the same moment created the table between
            # this one's drop and create: the table is there, so the cache is fine
            if "already exists" not in str(exc):
                raise
            logger.info("%s was cached by another process at the same time", table)
        n = len(df)
        logger.info("Cached %d rows -> %s", n, table)
        return n
    finally:
        conn.close()


def cache_load(table: str, db_path: str | Path | None = None) -> pd.DataFrame | None:
    """Load a table from the SQLite cache. Returns None if not cached."""
    conn = cache_connect(db_path)
    try:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        if not tables:
            return None
        return pd.read_sql(f"SELECT * FROM [{_safe_table_name(table)}]", conn)
    finally:
        conn.close()


def cache_list(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """List all cached tables with row counts."""
    conn = cache_connect(db_path)
    try:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        result = []
        for (name,) in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM [{_safe_table_name(name)}]").fetchone()[0]
            result.append({"table": name, "rows": count})
        return result
    finally:
        conn.close()


def wayback_snapshot_url(url: str, timestamp: str | None = None, timeout: int = 15) -> str | None:
    """The closest Internet Archive snapshot of ``url`` (https), or None.

    Mirrors ``rmoriebricklayer::wayback_snapshot_url()``: asks
    ``archive.org/wayback/available`` and returns the snapshot URL only when
    the archive reports it available.
    """
    from urllib.parse import quote

    api = "https://archive.org/wayback/available?url=" + quote(url, safe="")
    if timestamp:
        api += "&timestamp=" + timestamp
    try:
        res = json.loads(urlopen(api, timeout=timeout).read().decode())
        snap = res.get("archived_snapshots", {}).get("closest") or {}
        if snap.get("available") and snap.get("url"):
            return str(snap["url"]).replace("http://", "https://", 1)
    except Exception:  # noqa: BLE001 - the archive being down is the same as no snapshot
        return None
    return None


def download_with_wayback(url: str, timeout: int = 60) -> tuple[bytes, str]:
    """Fetch ``url``; if the live site fails, the closest Wayback snapshot.

    Returns the bytes and the URL they came from. Raises the live error when
    no snapshot exists either.
    """
    try:
        return urlopen(url, timeout=timeout).read(), url
    except Exception as live_exc:  # noqa: BLE001
        snap = wayback_snapshot_url(url)
        if snap is None:
            raise
        logger.warning("Live download of %s failed (%s); using the Wayback snapshot %s", url, live_exc, snap)
        return urlopen(snap, timeout=timeout).read(), snap


def _urlopen_json_with_retry(url: str, timeout: int, attempts: int = 4) -> dict:
    """GET a JSON document; a 409/429/5xx answer (the datastore under load) is retried with backoff."""
    import time as _time

    delay = 2.0
    for attempt in range(attempts):
        try:
            return json.loads(urlopen(url, timeout=timeout).read().decode())
        except HTTPError as exc:
            if exc.code not in (409, 429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
            logger.warning("CKAN answered %d; retrying in %.0f s", exc.code, delay)
            _time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


# a table this many cells or larger gets stage lines while it is built and cached
_STAGES_MIN_CELLS = 1_000_000


def fetch_ckan_to_cache(
    dataset_key: str = "cpads",
    limit: int = 32000,
    db_path: str | Path | None = None,
    timeout: int = 60,
    max_records: int | None = None,
) -> pd.DataFrame:
    """Fetch a dataset from CKAN and store it in the SQLite cache.

    Parameters
    ----------
    dataset_key : str
        Key in CKAN_DATASETS (e.g., "cpads", "csads", "csus").
    limit : int
        Records per page of the CKAN DataStore API.
    max_records : int, optional
        Stop once this many records have been fetched (``morie download-bootstrap --limit``).
    db_path : str | Path | None
        Override cache database path.
    timeout : int
        HTTP timeout in seconds.

    Returns
    -------
    pd.DataFrame
        The fetched (and optionally canonicalized) DataFrame.
    """
    info, table_name, is_cpads = _ckan_source(dataset_key)

    resource_id = info["resource_id"]
    if not resource_id:
        # Resolve resource_id from package metadata.
        meta_url = info["metadata_url"]
        logger.info("Resolving resource_id from %s", meta_url)
        meta = json.loads(urlopen(meta_url, timeout=timeout).read().decode())
        resources = meta.get("result", {}).get("resources", [])
        csv_resources = [r for r in resources if r.get("format", "").upper() == "CSV"]
        if csv_resources:
            resource_id = csv_resources[0]["id"]
        elif resources:
            resource_id = resources[0]["id"]
        else:
            raise ValueError(f"No resources found for {dataset_key}")

    # The DataStore API caps one request at 32,000 records, so page with
    # ``offset`` until a short page (or the reported total) says we are
    # done. A single request used to return the first 32,000 of the
    # 40,931 CPADS rows and call it the dataset.
    from ._progress import Progress

    records: list[dict] = []
    offset = 0
    prog: Progress | None = None
    logger.info("Fetching %s from CKAN (%d records per page)...", dataset_key, limit)
    while True:
        params = {"resource_id": resource_id, "limit": limit, "offset": offset}
        url = f"{DEFAULT_CKAN_API_BASE}?{urlencode(params)}"
        try:
            payload = _urlopen_json_with_retry(url, timeout)
        except HTTPError as exc:
            # 404: no datastore behind this resource. 500: the datastore
            # cannot serve a full page of it (the 2018-2022 CCS microdata,
            # the CSADS 2022 bootstrap weights). Either way the resource
            # itself is the file to read.
            if exc.code not in (404, 500) or offset:
                raise
            payload = {}
        except (URLError, OSError) as exc:
            # The datastore API is unreachable: the resource file is the
            # route, live or from its Wayback Machine snapshot.
            if offset:
                raise
            logger.warning("CKAN datastore unreachable for %s (%s); reading the resource file", dataset_key, exc)
            payload = {}
        result = payload.get("result", {})
        batch = result.get("records", [])
        records.extend(batch)
        total = result.get("total")
        if batch:
            if prog is None:
                want = total if isinstance(total, int) else None
                if max_records is not None:  # a --limit preview fetches only that many
                    want = max_records if want is None else min(want, max_records)
                prog = Progress(f"{dataset_key} (CKAN datastore)", want, unit="rows")
            prog.update(len(batch))
        if len(batch) < limit or (isinstance(total, int) and len(records) >= total):
            break
        if max_records is not None and len(records) >= max_records:
            break
        offset += len(batch)
    if prog is not None:
        prog.close()

    # building and caching a big table takes longer than fetching it (ocp21: 16 s fetched, then
    # ~110 s silent): say which step is running
    big = len(records) * max(len(records[0]) if records else 0, 1) >= _STAGES_MIN_CELLS
    stages = None
    if big:
        from ._progress import Stages

        stages = Stages(dataset_key, 3 if is_cpads else 2)
        stages.step(f"building the table ({len(records):,} rows x {len(records[0]):,} columns)")
    if records:
        df = pd.DataFrame.from_records(records)
        # Drop CKAN internal column.
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
    else:
        # Bootstrap-weight files and the older StatCan releases are not
        # loaded into the datastore (the API answers 404 or an empty page);
        # the resource itself is a CSV or a zip of CSVs.
        df = _ckan_resource_file(resource_id, dataset_key, timeout)

    logger.info("Fetched %d rows x %d cols for %s", len(df), len(df.columns), dataset_key)

    if max_records is not None:
        # a --limit preview is not the dataset: caching it made later pulls and modules
        # silently use the first rows (10 of 61,096 bootstrap replicates)
        logger.info("%s: first %d rows fetched (--limit); not cached", dataset_key, len(df))
        return df
    # Cache under the name load_dataset() looks up next time.
    if stages is not None:
        stages.step("caching it so the next pull is fast")
    cache_store(df, table_name, db_path)

    # If CPADS, also canonicalize and cache the canonical version.
    if is_cpads and has_raw_cpads_columns(df):
        if stages is not None:
            stages.step("adding the canonical CPADS columns")
        canonical = canonicalize_cpads_frame(df)
        cache_store(canonical, "cpads_canonical", db_path)
        return canonical

    return df


def _download_file(url: str, dest: Path, timeout: int = 60, label: str | None = None) -> str:
    """Stream ``url`` to ``dest``; when the live site fails, its closest Wayback snapshot.

    Returns the URL the bytes came from. The same fallback the R arm has in
    ``morie_download(attempt_wayback = TRUE)``.
    """
    from ._progress import download_url

    def _stream(src: str) -> None:
        # open.canada.ca's front end rejects a bare "Mozilla/5.0" agent; a dropped transfer is resumed
        download_url(
            src,
            dest,
            label or dest.name,
            timeout=timeout,
            headers={"User-Agent": "morie/1 (+https://rmorie.com)"},
            opener=lambda req, timeout: urlopen(req, timeout=timeout),
        )

    try:
        _stream(url)
        return url
    except Exception as live_exc:  # noqa: BLE001 - any live failure may have an archived copy
        snap = wayback_snapshot_url(url)
        if snap is None:
            raise
        logger.warning("Live download of %s failed (%s); using the Wayback snapshot %s", url, live_exc, snap)
        _stream(snap)
        return snap


def _download_url_table(entry: dict, matched: str, timeout: int = 60) -> pd.DataFrame:
    """Tier 3c: the catalog's ``download_url`` (a CSV/XLSX, or a zip whose ``zip_member`` is the table)."""
    import zipfile

    url = entry["download_url"]
    dest = _user_cache_dir() / "direct" / matched / url.rsplit("/", 1)[-1]
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        _download_file(url, tmp, timeout, label=matched)
        tmp.replace(dest)
    if zipfile.is_zipfile(dest):
        member = entry.get("zip_member") or ""
        with zipfile.ZipFile(dest) as zf:
            names = zf.namelist()
            pick = [n for n in names if n.rsplit("/", 1)[-1].lower() == member.lower()]
            if not pick:
                raise RuntimeError(f"{dest.name} for {matched} has no member {member!r}; it holds {', '.join(names)}")
            with zf.open(pick[0]) as fh:
                return pd.read_csv(fh, low_memory=False)
    if entry.get("format") == "xlsx":
        return _xlsx_data_sheet(dest)
    return pd.read_csv(dest, low_memory=False)


_COVER_SHEET = re.compile(
    r"^(instructions?|notes?( to readers?)?|(table of )?contents|about|read ?me|cover|footnotes?|glossary|definitions|methodology)$",
    re.IGNORECASE,
)


def _xlsx_data_sheet(path) -> pd.DataFrame:
    """The data sheet of a workbook: cover sheets ("Instructions", "Notes to readers") skipped, the most cells wins.

    CIHI data tables open on an Instructions sheet; reading sheet 0 returned that cover text as the dataset.
    """
    import zipfile

    from .fn._frame_core import _xlsx_sheet_map

    with zipfile.ZipFile(path) as zf:
        names = [nm for nm, _ in _xlsx_sheet_map(zf)]
    candidates = [nm for nm in names if not _COVER_SHEET.match((nm or "").strip())] or names
    best, best_cells = None, -1
    for nm in candidates:
        try:
            df = pd.read_excel(path, sheet_name=nm)
        except Exception:  # noqa: BLE001 - an unreadable sheet is skipped; the others still count
            continue
        cells = int(df.shape[0]) * int(df.shape[1])
        if cells > best_cells:
            best, best_cells = df, cells
    if best is None:
        raise RuntimeError(f"no readable sheet in {Path(path).name}")
    # header cells wrapped inside the workbook ("Number of \nhospital stays") become one-line names
    best.columns = [" ".join(str(c).split()) for c in best.columns]
    return _xlsx_promote_header(best)


_PLACEHOLDER = re.compile(r"^(Unnamed: \d+|\.\.\.\d+)$")


def _xlsx_cell_set(v) -> bool:
    return v is not None and not (isinstance(v, float) and v != v) and str(v).strip() != ""


def _xlsx_number(vals: list):
    """The column as numbers when every set cell is one, else unchanged."""
    out = []
    for v in vals:
        if not _xlsx_cell_set(v):
            out.append(float("nan"))
            continue
        if isinstance(v, int | float) and not isinstance(v, bool):
            out.append(v)
            continue
        try:
            f = float(str(v).strip())
        except ValueError:
            return vals
        out.append(int(f) if f.is_integer() and "." not in str(v) else f)
    return out


def _xlsx_promote_header(df: pd.DataFrame) -> pd.DataFrame:
    """A sheet whose first row is a title, its real header a few rows down (CIHI data tables).

    The title in A1 becomes the first column name and the rest are placeholders. The first row
    filled across most columns is the header; the table ends at the first blank row (what
    follows is notes, or further tables stacked on the same tab). Same rule as rmorie.
    """
    names = [str(c) for c in df.columns]
    n_rows, n_cols = df.shape
    if n_cols < 2 or not n_rows:
        return df
    if sum(bool(_PLACEHOLDER.match(c)) or not c.strip() for c in names[1:]) < 0.5 * (n_cols - 1):
        return df
    cols = [df[c].tolist() for c in df.columns]
    # the share is of the columns the table uses: a stray note far to the right widens the sheet
    # (cihi820b: a 5-column table on a 13-column tab) without being part of it
    head = min(n_rows, 20)
    used = sum(1 for col in cols if any(_xlsx_cell_set(col[i]) for i in range(head)))
    need = max(2, -(-4 * used // 5))
    hdr = next((i for i in range(head) if sum(_xlsx_cell_set(col[i]) for col in cols) >= need), None)
    if hdr is None:
        return df
    end = n_rows
    for i in range(hdr + 1, n_rows):
        if not any(_xlsx_cell_set(col[i]) for col in cols):
            end = i
            if any(_xlsx_cell_set(col[j]) for col in cols for j in range(i, n_rows)):
                logger.info("rows after the table's first blank line (notes, or further tables) are left out")
            break
    # an unnamed column with nothing in the table is not part of it
    cols = [col for col in cols if _xlsx_cell_set(col[hdr]) or any(_xlsx_cell_set(v) for v in col[hdr + 1 : end])]
    new, seen = [], {}
    for j, col in enumerate(cols):
        nm = " ".join(str(col[hdr]).split()) if _xlsx_cell_set(col[hdr]) else f"...{j + 1}"
        k = seen.get(nm, 0)
        seen[nm] = k + 1
        new.append(nm if k == 0 else f"{nm}.{k}")
    return pd.DataFrame({nm: _xlsx_number(col[hdr + 1 : end]) for nm, col in zip(new, cols)})


def _direct_or_hosted(entry: dict, matched: str, db_path, timeout: int = 60) -> pd.DataFrame:
    """Tier 3c: portal first, the data.rmorie.com copy second; cache whichever answered."""
    err: Exception | None = None
    df = None
    if entry.get("download_url"):
        try:
            df = _download_url_table(entry, matched, timeout)
        except Exception as exc:  # noqa: BLE001 - the hosted copy is the fallback
            err = exc
            logger.warning("Direct download of %s failed (%s)", matched, exc)
    hk = entry.get("hosted_key")
    if df is None and hk:
        from .datahub import load_hosted_dataset
        from .hosted import hosted_key

        if hosted_key():
            df = load_hosted_dataset(hk, db_path=db_path)
        elif err is not None:
            raise RuntimeError(
                f"{matched}: the portal download failed ({err}); the data.rmorie.com copy ({hk}) "
                "opens with your MORIE key: run `morie login` (GitHub) or `morie login --email you@example.com` once."
            ) from err
        else:
            raise RuntimeError(
                f"{matched} is served from data.rmorie.com as {hk}: run `morie login` (GitHub) or `morie login --email you@example.com` once, then "
                f"`morie pull {matched}` (or `morie pull {hk}`)."
            )
    if df is None:
        assert err is not None
        raise err
    try:
        cache_store(df, entry["table_name"], db_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not cache %s: %s", matched, exc)
    return df


class RObjectSavedError(NotImplementedError):
    """The dataset is an R object (RData/rds): Python cannot open it, but it is saved at ``path`` for R."""

    def __init__(self, key: str, path: Path):
        self.key, self.path = key, Path(path)
        super().__init__(
            f"{key} is an R object ({self.path.suffix}), saved at {self.path}. Open it in R with "
            f"rmorie::morie_load_dataset('{key}'), or {'readRDS' if str(path).lower().endswith('.rds') else 'load'}() on that file."
        )


def _load_hosted_file(entry: dict, matched: str):
    """Tier 3b: fetch an R object (RData/rds) from data.rmorie.com into the data directory and say where it is."""
    from .datahub import _get_to_file

    rel = Path(entry["local_path"])
    tail = Path(*rel.parts[1:]) if rel.parts and rel.parts[0] == "data" else rel
    dest = _data_dir_candidates()[0] / tail
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _get_to_file("/files/" + entry["hosted_file"], dest, label=matched)
    raise RObjectSavedError(matched, dest)


def _ckan_resource_file(resource_id: str, dataset_key: str, timeout: int = 60) -> pd.DataFrame:
    """Download a CKAN resource that has no datastore and read it as a table.

    The resource URL comes from ``resource_show``. A zip (StatCan's
    ``CSV.zip`` releases hold the microdata, the bootstrap weights and the
    PDF codebooks together) yields the CSV that matches the catalog entry:
    the bootstrap file for a ``bootstrap`` key, the other one otherwise.
    The download is kept under the cache directory.
    """
    import zipfile

    meta_url = f"{DEFAULT_CKAN_API_BASE.rsplit('/', 1)[0]}/resource_show?id={resource_id}"
    meta = json.loads(urlopen(meta_url, timeout=timeout).read().decode())
    url = (meta.get("result") or {}).get("url") or ""
    if not url:
        raise RuntimeError(f"CKAN returned 0 records for {dataset_key} and resource {resource_id} has no file URL")
    dest = _user_cache_dir() / "ckan" / resource_id / url.rsplit("/", 1)[-1]
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s for %s", url, dataset_key)
        # open.canada.ca's front end rejects a bare "Mozilla/5.0" agent
        tmp = dest.with_suffix(dest.suffix + ".part")
        _download_file(url, tmp, timeout, label=dataset_key)
        tmp.replace(dest)
    if not zipfile.is_zipfile(dest):
        return pd.read_csv(dest, low_memory=False)
    entry = DATASET_CATALOG.get(dataset_key, {})
    want_boot = entry.get("type") == "bootstrap"
    with zipfile.ZipFile(dest) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        member = entry.get("zip_member")
        if member:  # the catalog names the member: StatCan's CSV.zip holds the microdata and the weights together
            pick = [n for n in csvs if n.rsplit("/", 1)[-1] == member]
            if not pick:
                raise RuntimeError(
                    f"{dest.name} for {dataset_key} has no member {member!r}; it holds {', '.join(csvs)}"
                )
        else:
            is_boot = lambda n: bool(re.search(r"bsw|bwt|boot", n.rsplit("/", 1)[-1], re.I))  # noqa: E731
            pick = [n for n in csvs if is_boot(n) == want_boot] or csvs
        if not pick:
            raise RuntimeError(f"{dest.name} for {dataset_key} holds no CSV")
        with zf.open(pick[0]) as fh:
            return pd.read_csv(fh, low_memory=False)


def _data_dir_candidates() -> list[Path]:
    """Where a catalog ``local_path`` (``data/datasets/...``) may live.

    ``MORIE_DATA_DIR`` first (a directory holding ``datasets/``), then the
    per-user data directory of :mod:`morie._datapaths`, then a source
    checkout, then the working directory.
    """
    from ._datapaths import _user_data_dir

    out: list[Path] = []
    env = os.environ.get("MORIE_DATA_DIR", "").strip()
    if env:
        out.append(Path(env).expanduser())
    out.append(_user_data_dir())
    out.append(_project_root() / "data")
    out.append(Path.cwd() / "data")
    return out


def _own_file_target(rel: str) -> Path:
    """Where a user puts an own-file dataset: under ``MORIE_DATA_DIR`` when set, else the per-user data dir."""
    p = Path(rel)
    tail = Path(*p.parts[1:]) if p.parts and p.parts[0] == "data" else p
    return _data_dir_candidates()[0] / tail


def _find_local_file(rel: str) -> Path | None:
    """Resolve a catalog ``local_path`` through :func:`_data_dir_candidates`."""
    p = Path(rel)
    if p.is_absolute():
        return p if p.exists() else None
    tail = Path(*p.parts[1:]) if p.parts and p.parts[0] == "data" else p
    for base in _data_dir_candidates():
        for cand in (base / tail, base / p):
            if cand.exists():
                return cand
    return None


RMORIEDATA_VERSION = "0.3.3"
RMORIEDATA_TARBALL = f"https://cran.r-project.org/src/contrib/rmoriedata_{RMORIEDATA_VERSION}.tar.gz"


def _rmoriedata_extdata(timeout: int = 120) -> Path:
    """The ``inst/extdata`` of rmoriedata, fetched once from CRAN into the user cache.

    rmoriedata is the family's data package on CRAN; its tables are plain
    CSV files, so Python reads them without R. The tarball is 6.8 MB.
    """
    root = _user_cache_dir() / "rmoriedata" / RMORIEDATA_VERSION
    ext = root / "extdata"
    if (ext / "_catalog.csv").exists():
        return ext
    import tarfile
    import tempfile

    root.mkdir(parents=True, exist_ok=True)
    from ._progress import download_url

    logger.info("Fetching rmoriedata %s from CRAN (%s)...", RMORIEDATA_VERSION, RMORIEDATA_TARBALL)
    with tempfile.TemporaryDirectory() as tmp:
        tgz = Path(tmp) / "rmoriedata.tar.gz"
        download_url(
            RMORIEDATA_TARBALL,
            tgz,
            f"rmoriedata {RMORIEDATA_VERSION} (CRAN)",
            timeout=timeout,
            opener=lambda req, timeout: urlopen(req, timeout=timeout),
        )
        with tarfile.open(tgz) as tf:
            members = [m for m in tf.getmembers() if "/inst/extdata/" in m.name and not m.name.endswith("/")]
            for m in members:
                target = ext / m.name.split("/inst/extdata/", 1)[1]
                if ".." in target.parts:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                fh = tf.extractfile(m)
                if fh is not None:
                    target.write_bytes(fh.read())
    if not (ext / "_catalog.csv").exists():
        raise RuntimeError("the rmoriedata tarball carried no extdata catalog")
    return ext


def list_rmoriedata(timeout: int = 120) -> list[dict]:
    """The rmoriedata catalog: one dict per slug (slug, source_path, kind, n_rows, n_cols)."""
    import csv

    ext = _rmoriedata_extdata(timeout=timeout)
    with (ext / "_catalog.csv").open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:  # counts are numbers (r["n_rows"] > 100 raised TypeError on the CSV strings)
        for k in ("n_rows", "n_cols"):
            v = (r.get(k) or "").strip()
            r[k] = int(v) if v.lstrip("-").isdigit() else None
    return rows


def load_rmoriedata(slug: str, timeout: int = 120) -> pd.DataFrame:
    """Load one rmoriedata table by slug (``list_rmoriedata()`` lists them).

    Reads the CSV (or CSV.gz) that rmoriedata ships, applying the column
    names from its schema exactly as ``rmoriedata::morie_data_load()`` does.
    """
    import csv
    import gzip

    ext = _rmoriedata_extdata(timeout=timeout)
    rows = list_rmoriedata(timeout=timeout)
    hit = [r for r in rows if r.get("slug") == slug]
    if not hit:
        known = ", ".join(sorted(r["slug"] for r in rows if r.get("kind") == "table"))
        raise KeyError(f"rmoriedata has no table {slug!r}. Tables: {known}")
    entry = hit[0]
    if entry.get("kind") != "table":
        raise ValueError(f"rmoriedata slug {slug!r} is a {entry.get('kind')}, not a table")
    src = ext / entry["source_path"]
    if not src.exists():
        raise FileNotFoundError(f"rmoriedata file missing from the tarball: {src}")
    if src.suffix == ".gz":
        plain = src.with_suffix("")
        if not plain.exists():
            with gzip.open(src, "rb") as fin, plain.open("wb") as fout:
                fout.write(fin.read())
        src = plain
    df = pd.read_csv(src, low_memory=False)
    schema = ext / "_schema.csv"
    if schema.exists():
        with schema.open(newline="", encoding="utf-8-sig") as fh:
            names = [r["name"] for r in csv.DictReader(fh) if r.get("slug") == slug]
        if names and len(names) == len(df.columns):
            df.columns = names
    return df


CIHI_INDICATOR_LIBRARY_URL = (
    "https://www.cihi.ca/sites/default/files/document/indicator-library-all-indicator-data-en.xlsx"
)


def fetch_cihi_indicator_library(timeout: int = 120) -> pd.DataFrame:
    """CIHI's Indicator Library, all indicators: the public workbook, fetched once.

    The workbook is 72 MB and 809,000 rows by 33 columns. Loading it as a
    worksheet object needs more than 10 GB, so it is streamed row by row
    into a CSV next to it (once) and read from there. A copy you already
    have under ``MORIE_DATA_DIR`` is used instead of downloading.
    """
    import csv
    from urllib.request import Request

    xlsx = _find_local_file("data/datasets/cihi/indicator-library-all-indicator-data-en.xlsx")
    if xlsx is None:
        xlsx = _user_cache_dir() / "cihi" / "indicator-library-all-indicator-data-en.xlsx"
        if not xlsx.exists():
            xlsx.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Fetching the CIHI indicator library (72 MB) from %s", CIHI_INDICATOR_LIBRARY_URL)
            req = Request(CIHI_INDICATOR_LIBRARY_URL, headers={"User-Agent": "morie/1 (+https://rmorie.com)"})
            import tempfile

            from ._progress import stream_to_file

            fd, part = tempfile.mkstemp(dir=xlsx.parent, prefix=xlsx.name + ".", suffix=".part")
            os.close(fd)
            try:
                with urlopen(req, timeout=timeout) as resp:
                    stream_to_file(resp, part, "CIHI indicator library")
                os.replace(part, xlsx)
            finally:
                if os.path.exists(part):
                    os.remove(part)
    csv_path = _user_cache_dir() / "cihi" / "indicator-library-all-indicator-data-en.csv"
    if not csv_path.exists() or csv_path.stat().st_mtime < xlsx.stat().st_mtime:
        import openpyxl

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
        ws = wb.worksheets[0]
        import tempfile

        fd, tmp_name = tempfile.mkstemp(dir=csv_path.parent, prefix=csv_path.name + ".", suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            from ._progress import Progress

            with (
                open(fd, "w", newline="", encoding="utf-8") as fh,
                Progress("CIHI workbook -> CSV", unit="rows") as prog,
            ):
                w = csv.writer(fh)
                for row in ws.iter_rows(values_only=True):
                    w.writerow(["" if v is None else v for v in row])
                    prog.update(1)
            wb.close()
            tmp.replace(csv_path)  # atomic: a concurrent pull either sees the old file or the new one
        finally:
            tmp.unlink(missing_ok=True)
    return pd.read_csv(csv_path, low_memory=False)


def _ckan_source(dataset_key: str) -> tuple[dict[str, str], str, bool]:
    """Resolve a CKAN_DATASETS key or a DATASET_CATALOG key with a resource id.

    Returns the source record, the cache table to store under, and whether
    the rows are raw CPADS (so the canonical frame is cached as well).
    ``load_dataset("ocp21")`` reaches here with the catalog key; it used to
    be rejected because only the three short keys were known.
    """
    info = CKAN_DATASETS.get(dataset_key)
    if info:
        return dict(info), f"{dataset_key}_raw", dataset_key == "cpads"
    entry = DATASET_CATALOG.get(dataset_key)
    if entry and entry.get("ckan_resource_id"):
        info = {"name": entry["name"], "resource_id": entry["ckan_resource_id"], "metadata_url": ""}
        return info, entry["table_name"], entry.get("survey") == "cpads"
    known = sorted(set(CKAN_DATASETS) | {k for k, e in DATASET_CATALOG.items() if e.get("ckan_resource_id")})
    raise ValueError(f"Unknown CKAN dataset: {dataset_key}. Known: {known}")


def cached_cpads() -> pd.DataFrame | None:
    """The real CPADS PUMF already in the built-in store (``morie pull cpads``), or None. No network."""
    builtin = _builtin_db_connect()
    if builtin is None:
        return None
    try:
        for tbl in ("cpads_canonical", "ocp21"):
            hit = builtin.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tbl,)).fetchone()
            if hit:
                return pd.read_sql(f"SELECT * FROM [{_safe_table_name(tbl)}]", builtin)
    except Exception:
        return None
    finally:
        builtin.close()
    return None


def load_cpads(db_path: str | Path | None = None, timeout: int = 60) -> pd.DataFrame:
    """Load CPADS data: try local files, then cache, then CKAN API.

    Resolution order:
    1. Local RDS (via R bridge) or CSV files in standard locations
    2. SQLite cache (data/cache/morie.db)
    3. CKAN API fetch -> cache -> return
    """
    # 1. Try local files.
    local_candidates = [
        "data/datasets/oc/CPADS/2021-2022/cpads-2021-2022-pumf2.csv",
        "data/cache/cpads_pumf_wrangled.rds",
    ]
    for path in local_candidates:
        p = Path(path)
        if not p.exists():
            continue
        if p.suffix == ".csv":
            df = pd.read_csv(p)
            if has_raw_cpads_columns(df):
                return canonicalize_cpads_frame(df)
            validate_cpads_frame(df, strict=True)
            return df
        if p.suffix == ".rds":
            # RDS files need R -- skip in Python, prefer CSV or cache.
            continue

    # 2. Try built-in morie.db (ships with package).
    builtin = _builtin_db_connect()
    if builtin is not None:
        try:
            # Look for the canonical CPADS table in the built-in DB
            for tbl in ("cpads_canonical", "ocp21"):
                hit = builtin.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (tbl,),
                ).fetchone()
                if hit:
                    df = pd.read_sql(f"SELECT * FROM [{_safe_table_name(tbl)}]", builtin)
                    logger.info("Loaded CPADS from built-in DB table %s (%d rows)", tbl, len(df))
                    if has_raw_cpads_columns(df):
                        return canonicalize_cpads_frame(df)
                    return df
        finally:
            builtin.close()

    # 3. Try SQLite cache.
    cached = cache_load("cpads_canonical", db_path)
    if cached is not None:
        logger.info("Loaded CPADS from cache (%d rows)", len(cached))
        return cached

    # 4. Fetch from CKAN API and cache.
    logger.info("CPADS not found locally or in cache. Fetching from CKAN...")
    return fetch_ckan_to_cache("cpads", db_path=db_path, timeout=timeout)


# ---------------------------------------------------------------------------
# Unified load interface
# ---------------------------------------------------------------------------


def _fuzzy_match_key(key: str) -> str | None:
    """Match a key like 'cpads', 'ocp21', or 'naps-no2-on-2023' to a
    catalog entry. Handles both hyphen and underscore separators in the
    input by normalising BOTH the query and each catalog key to the same
    form (underscores + lowercase + no whitespace) before comparing."""

    def _norm(s: str) -> str:
        return s.lower().replace("-", "_").replace(" ", "")

    key_lower = _norm(key)
    # Exact match on either the raw key or the normalised form.
    if key in DATASET_CATALOG:
        return key
    if key_lower in DATASET_CATALOG:
        return key_lower
    # Normalised equality against every catalog key.
    for full_key in DATASET_CATALOG:
        if _norm(full_key) == key_lower:
            return full_key
    # Substring matches on key / survey / name (also normalised).
    for full_key, entry in DATASET_CATALOG.items():
        if key_lower in _norm(full_key):
            return full_key
        if key_lower == _norm(entry.get("survey", "")):
            return full_key
        if key_lower in entry.get("name", "").lower().replace(" ", ""):
            return full_key
    return None


def synthetic_mapq_panel(n: int = 400, seed: int = 2026) -> pd.DataFrame:
    """Deterministic synthetic MAPQII panel: a toy stand-in for TKARONTOMAPQ.xlsx.

    The same design as rmorie's ``.morie_mapq_synth_panel()``: 20 Likert items (1-5) in four
    subscales (EE, EA, UA, ER) with a planted one-factor-per-subscale structure, ``gender_male``
    and ``age``, subscale scores, and a Knowledge Scale score ``ks_score`` driven by epistemic
    attitudes, gender and age, so the psychometric and DML stages have a real signal to recover.
    Participant-level MAPQ data are not distributed; this panel is.
    """
    import random

    from morie.fn._mapq_const import SUBSCALES

    rng = random.Random(seed)
    cols: dict[str, list] = {
        "gender_male": [1 if rng.random() < 0.5 else 0 for _ in range(n)],
        "age": [rng.randint(18, 65) for _ in range(n)],
    }
    for items in SUBSCALES.values():
        latent = [rng.gauss(0.0, 1.0) for _ in range(n)]
        for item in items:
            cols[item] = [min(5, max(1, round(3 + 0.9 * z + rng.gauss(0.0, 0.8)))) for z in latent]
    for name, items in SUBSCALES.items():
        cols[f"{name.lower()}_score"] = [sum(cols[i][r] for i in items) for r in range(n)]
    cols["ks_score"] = [
        10 + 0.4 * cols["ea_score"][r] + 1.5 * cols["gender_male"][r] + 0.02 * cols["age"][r] + rng.gauss(0.0, 1.0)
        for r in range(n)
    ]
    return pd.DataFrame(cols)


# own-file datasets with a synthetic toy stand-in used when the real file is absent
SYNTHETIC_OWN_FILES = {"mapq": synthetic_mapq_panel}


def _load_dataset_raw(
    key: str,
    *,
    db_path: str | Path | None = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """Load a dataset by catalog key.

    Resolution order:
    1. Built-in morie.db (ships with package)
    2. User cache data/cache/morie.db
    3. Local file (ingest to cache on the fly)
    4. CKAN API (if resource ID available)
    5. Error

    Supports fuzzy matching: ``load_dataset("cpads")`` resolves to ``ocp21``.
    """
    matched = _fuzzy_match_key(key)
    if matched is None:
        from .datahub import is_hosted_key, load_hosted_dataset

        if is_hosted_key(key):
            # a curated table at data.rmorie.com (db/table), opened by the MORIE key
            return load_hosted_dataset(key, db_path=db_path)
        available = ", ".join(sorted(DATASET_CATALOG))
        raise KeyError(
            f"Unknown dataset key: {key!r}. Available: {available}; "
            "curated tables at data.rmorie.com use db/table keys (morie list-datasets shows them after `morie login`, GitHub or --email)."
        )

    entry = DATASET_CATALOG[matched]
    table_name = entry["table_name"]

    if dataset_route(entry).startswith("own file"):
        # your own research file: read it where it is, every time, and keep it out of the cache
        local_path = _find_local_file(entry["local_path"])
        if local_path is None and matched in SYNTHETIC_OWN_FILES:
            import warnings

            warnings.warn(
                f"{matched}: your file is not at {_own_file_target(entry['local_path'])}; "
                "returning the synthetic toy panel (n = 400, planted structure) so the analyses run. "
                "Its numbers demonstrate the pipeline, they are not findings.",
                UserWarning,
                stacklevel=2,
            )
            return SYNTHETIC_OWN_FILES[matched]()
        if local_path is None:
            raise FileNotFoundError(dataset_recommendation(matched, entry))
        logger.info("Reading %s from your file: %s", matched, local_path)
        if entry["format"] == "csv":
            return pd.read_csv(local_path, low_memory=False)
        if entry["format"] == "xlsx":
            return pd.read_excel(local_path)
        raise NotImplementedError(f"Format {entry['format']} not supported for on-the-fly ingest")

    # 1. Built-in database (ships with package).
    builtin = _builtin_db_connect()
    if builtin is not None:
        try:
            tables = builtin.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            if tables:
                df = pd.read_sql(f"SELECT * FROM [{_safe_table_name(table_name)}]", builtin)
                logger.info("Loaded %s from built-in DB (%d rows)", matched, len(df))
                if matched == "ocp21" and has_raw_cpads_columns(df):
                    return canonicalize_cpads_frame(df)  # the same frame the cache tier returns
                return df
        finally:
            builtin.close()

    # 2. User cache. A misconfigured or unwritable cache path must not
    #    crash the load -- skip the cache tier and carry on.
    try:
        cached = cache_load(table_name, db_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cache tier skipped for %s: %s", matched, exc)
        cached = None
    if cached is not None:
        logger.info("Loaded %s from cache (%d rows)", matched, len(cached))
        if matched == "ocp21" and has_raw_cpads_columns(cached):
            # the first load cached the canonical frame alongside the raw one: same shape every call
            try:
                canonical = cache_load("cpads_canonical", db_path)
            except Exception:  # noqa: BLE001
                canonical = None
            if canonical is not None and len(canonical) == len(cached):
                return canonical
            return canonicalize_cpads_frame(cached)
        return cached

    # 2b. Fetcher dispatch (NAPS, OpenAQ, Earth Engine, ArcGIS).
    #     Entry has ``source`` in {"naps", "openaq", "ee", "arcgis"} OR a
    #     ``fetcher`` key of the form "module.path:fn_name". Call the fn
    #     with ``fetcher_args`` and cache the resulting DataFrame.
    fetcher_spec = entry.get("fetcher")
    if fetcher_spec:
        import importlib

        mod_name, fn_name = fetcher_spec.split(":", 1)
        try:
            mod = importlib.import_module(mod_name)
            fetcher_fn = getattr(mod, fn_name)
        except (ImportError, AttributeError) as exc:
            raise ImportError(f"Cannot resolve fetcher {fetcher_spec!r} for dataset {matched!r}: {exc}") from exc
        args = dict(entry.get("fetcher_args") or {})
        logger.info("Fetching %s via %s(**%s) ...", matched, fetcher_spec, args)
        df = fetcher_fn(**args)
        if df is not None and len(df):
            try:
                cache_store(df, table_name, db_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not cache %s: %s", matched, exc)
        return df

    # 2c. rmoriedata (CRAN): read the slug from the package's extdata,
    #     fetched once from CRAN as a source tarball. No R needed.
    slug = entry.get("rmoriedata")
    if slug:
        df = load_rmoriedata(slug, timeout=timeout)
        try:
            cache_store(df, table_name, db_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not cache %s: %s", matched, exc)
        return df

    # 3a. Research files that are not tables (R environments) kept at data.rmorie.com live in the
    #     data directory in effect (MORIE_DATA_DIR when set): a copy saved earlier under another
    #     directory must not win over the one the user chose.
    if entry.get("hosted_file"):
        return _load_hosted_file(entry, matched)

    # 3. Local file: the catalog path is relative to a data directory.
    #    An installed package has no source tree, so the cascade is
    #    MORIE_DATA_DIR, the per-user data directory, the source checkout
    #    (when there is one), then the working directory.
    local_path = _find_local_file(entry["local_path"])
    if local_path is not None:
        logger.info("Ingesting %s from local file: %s", matched, local_path)
        if entry["format"] == "csv":
            df = pd.read_csv(local_path, low_memory=False)
        elif entry["format"] == "xlsx":
            df = pd.read_excel(local_path)
        elif str(entry["format"]).lower() in ("rdata", "rda", "rds"):
            raise RObjectSavedError(matched, local_path)
        else:
            raise NotImplementedError(f"Format {entry['format']} not supported for on-the-fly ingest")
        try:
            cache_store(df, table_name, db_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not cache %s: %s", matched, exc)
        return df

    # 3c. A direct portal download (a file, or one member of a zip), then the
    #     data.rmorie.com copy of the same table when the portal fails or the
    #     catalog only names the copy.
    if entry.get("download_url") or entry.get("hosted_key"):
        return _direct_or_hosted(entry, matched, db_path, timeout)

    # 4. Open data portals: Ontario's catalogue for the OTIS tables (their
    #    downloader lives in morie.otis_datasets), open.canada.ca for the
    #    rest of the CKAN-backed keys.
    rid = entry.get("ckan_resource_id", "")
    if rid and entry.get("source") == "otis":
        from .otis_datasets import load_otis_dataset

        logger.info("Fetching %s from data.ontario.ca...", matched)
        df = load_otis_dataset(entry["survey"])
        try:
            cache_store(df, table_name, db_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not cache %s: %s", matched, exc)
        return df
    if rid:
        logger.info("Fetching %s from CKAN API...", matched)
        return fetch_ckan_to_cache(matched, db_path=db_path, timeout=timeout)

    raise FileNotFoundError(f"Dataset {matched!r} could not be loaded.\n" + dataset_recommendation(matched, entry))


def list_datasets(db_path: str | Path | None = None, *, hosted: bool = True) -> list[dict]:
    """List all datasets with their cache status.

    Returns a list of dicts with keys: key, name, source, survey, year,
    type, cached (bool), rows (int or None).

    When *db_path* is explicitly provided, only that database is queried
    (the built-in DB is skipped).  This keeps unit tests deterministic.
    """
    cached_tables: dict[str, int] = {}
    if db_path is None:
        # Check built-in database first (only when no explicit db_path)
        builtin = _builtin_db_connect()
        if builtin is not None:
            try:
                for (name,) in builtin.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                    count = builtin.execute(f"SELECT COUNT(*) FROM [{_safe_table_name(name)}]").fetchone()[0]
                    cached_tables[name] = count
            except Exception:
                pass
            finally:
                builtin.close()
    # Also check user cache (or the explicit db_path)
    try:
        for item in cache_list(db_path):
            if item["table"] not in cached_tables:
                cached_tables[item["table"]] = item["rows"]
    except Exception:
        pass

    result = []
    for key, entry in DATASET_CATALOG.items():
        tbl = entry["table_name"]
        result.append(
            {
                "key": key,
                "name": entry["name"],
                "source": entry["source"],
                "survey": entry["survey"],
                "year": entry["year"],
                "type": entry["type"],
                "cached": tbl in cached_tables,
                "rows": cached_tables.get(tbl),
                "route": dataset_route(entry),
            }
        )
    from .datahub import DataHubAuthError, cached_manifest, hosted_entries, hosted_manifest

    try:
        from .hosted import hosted_key

        # hosted=False: the copy already on disk only (selftest promises no downloads)
        manifest = hosted_manifest() if hosted and hosted_key() else cached_manifest()
    except DataHubAuthError:
        manifest = cached_manifest()
    except Exception as exc:  # noqa: BLE001 - offline: the local list still prints
        logger.warning("data.rmorie.com manifest unavailable (%s)", exc)
        manifest = cached_manifest()
    for row in hosted_entries(manifest):
        row["cached"] = row["table_name"] in cached_tables
        if row["cached"]:
            row["rows"] = cached_tables.get(row["table_name"])
        result.append(row)
    return result


def dataset_route(entry: dict) -> str:
    """How a catalog entry is obtained: a portal it downloads from, rmoriedata, or your own file."""
    if entry.get("rmoriedata"):
        return "rmoriedata (CRAN)"
    fetcher = entry.get("fetcher") or ""
    if fetcher:
        return {
            "naps": "ECCC NAPS",
            "statcan": "Statistics Canada",
            "cihi": "CIHI",
            "tps": "Toronto Police ArcGIS",
        }.get(entry.get("source", ""), "fetched on demand")
    if entry.get("ckan_resource_id"):
        return "data.ontario.ca" if entry.get("source") == "otis" else "open.canada.ca"
    if entry.get("download_url"):
        host = urlparse(entry["download_url"]).netloc
        return host + (" (or data.rmorie.com)" if entry.get("hosted_key") else "")
    if entry.get("hosted_key"):
        return "data.rmorie.com (your MORIE key)"
    if entry.get("hosted_file"):
        return "data.rmorie.com file (an R object: rmorie loads it, morie saves it)"
    if entry.get("source") in CKAN_DATASETS:
        return "open.canada.ca"
    return f"own file: {_own_file_target(entry.get('local_path', ''))}"


def _cache_rows(table: str, db_path: str | Path | None = None) -> int | None:
    """Row count of a cached table without loading it; None when it is not cached."""
    conn = cache_connect(db_path)
    try:
        if not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
            return None
        return int(conn.execute(f"SELECT COUNT(*) FROM [{_safe_table_name(table)}]").fetchone()[0])
    finally:
        conn.close()


def dataset_info(key: str) -> dict:
    """Return full metadata for a dataset: a catalog key, or a curated ``db/table`` key."""
    from .datahub import cached_manifest, hosted_entries, hosted_manifest, is_hosted_key

    if is_hosted_key(key):
        manifest = cached_manifest()
        if manifest is None:
            try:
                manifest = hosted_manifest()
            except Exception:  # noqa: BLE001 - no key or no network: said below
                manifest = None
        row = next((d for d in (manifest or {}).get("datasets", []) if d.get("key") == key), None)
        if row is None:
            hint = (
                ""
                if manifest
                else " (no data.rmorie.com manifest at hand: run `morie login`, then `morie list-datasets`)"
            )
            raise KeyError(f"Unknown dataset key: {key!r}{hint}")
        info = hosted_entries({"datasets": [row]})[0]
        info["meta"] = row.get("meta") or {}
    else:
        matched = _fuzzy_match_key(key)
        if matched is None:
            raise KeyError(f"Unknown dataset key: {key!r}")
        info = dict(DATASET_CATALOG[matched])
        info["key"] = matched
        info["local_exists"] = Path(info["local_path"]).exists()
    rows = _cache_rows(info["table_name"])
    info["cached"] = rows is not None
    info["cached_rows"] = rows
    return info


class DatasetRegistry:
    """
    A registry to manage, catalog, and load secure epidemiological datasets.
    """

    def __init__(self, data_dir: str = "data/datasets/"):
        """
        Initialize the dataset registry.

        :param data_dir: The root file directory containing secure datasets, defaults to "data/datasets/".
        :type data_dir: str
        """
        self.data_dir = os.fspath(data_dir)
        self.catalog = {
            "cpads_2021_2022_pumf": {
                "name": "CPADS 2021-2022 PUMF",
                "path": "cpads/2021_2022/pumf.csv",
                "format": "csv",
                "type": "survey",
                "source_kind": "local_private_file",
                "landing_page": "https://open.canada.ca/data/en/dataset/736fa9b2-62e4-4e31-aea4-51869605b363",
                "documentation_url": "https://open.canada.ca/data/en/dataset/736fa9b2-62e4-4e31-aea4-51869605b363",
                "ckan_api_base": DEFAULT_CKAN_API_BASE,
                "ckan_resource_id": "d2639429-c304-45a6-90b3-770562f4d46d",
                "required_variables": list(CPADS_REQUIRED_VARIABLES),
            }
        }

    def register_dataset(self, name: str, metadata: dict[str, Any]):
        """
        Register a dataset in the internal catalog.

        :param name: Unique identifier for the dataset.
        :type name: str
        :param metadata: A dictionary mapping metadata values (like name, path, format).
        :type metadata: dict
        """
        required_keys = {"name", "path", "format", "type"}
        missing_keys = required_keys.difference(metadata)
        if missing_keys:
            raise ValueError("Dataset metadata is missing required keys: " + ", ".join(sorted(missing_keys)))

        self.catalog[name] = dict(metadata)

    def list_datasets(self) -> dict[str, dict[str, Any]]:
        """
        Return a copy of the registered dataset catalog.

        :return: Dataset metadata keyed by registry identifier.
        :rtype: dict[str, dict[str, Any]]
        """
        return deepcopy(self.catalog)

    def get_dataset_metadata(self, name: str) -> dict[str, Any]:
        """Return a copy of the metadata for one registered dataset."""
        if name not in self.catalog:
            raise ValueError(f"Dataset {name} not found in registry.")
        return dict(self.catalog[name])

    def fetch_ckan_records(
        self,
        name: str,
        *,
        limit: int = 5,
        query: str | None = None,
        timeout: int = 30,
    ) -> dict[str, Any]:
        """
        Fetch records for a dataset backed by a CKAN DataStore resource.
        """
        info = self.get_dataset_metadata(name)
        resource_id = info.get("ckan_resource_id")
        if not resource_id:
            raise ValueError(f"Dataset {name} does not define a CKAN resource id.")

        params = {
            "resource_id": resource_id,
            "limit": int(limit),
        }
        if query:
            params["q"] = query

        base_url = info.get("ckan_api_base", DEFAULT_CKAN_API_BASE)
        url = f"{base_url}?{urlencode(params)}"
        payload = urlopen(url, timeout=timeout).read().decode("utf-8")
        return json.loads(payload)

    def fetch_ckan_dataframe(
        self,
        name: str,
        *,
        limit: int = 100,
        query: str | None = None,
        timeout: int = 30,
    ) -> pd.DataFrame:
        """
        Fetch a CKAN result set and return the records as a DataFrame.
        """
        payload = self.fetch_ckan_records(
            name,
            limit=limit,
            query=query,
            timeout=timeout,
        )
        records = payload.get("result", {}).get("records", [])
        return pd.DataFrame.from_records(records)

    def cpads_contract(self) -> dict[str, Any]:
        """Return the canonical local-private CPADS contract."""
        return cpads_contract()

    def register_local_cpads(self, path: str | Path, *, name: str = "cpads_local") -> dict[str, Any]:
        """Register a user-provided local CPADS file."""
        resolved = Path(path).expanduser()
        metadata = {
            "name": "Local CPADS analysis file",
            "path": os.fspath(resolved),
            "format": infer_file_format(resolved),
            "type": "survey",
            "source_kind": "local_private_file",
            "required_variables": list(CPADS_REQUIRED_VARIABLES),
        }
        self.catalog[name] = metadata
        return dict(metadata)

    def validate_cpads_frame(self, frame: pd.DataFrame, *, strict: bool = True) -> list[str]:
        """Validate a DataFrame against the canonical CPADS variable contract."""
        return validate_cpads_frame(frame, strict=strict)

    def load(self, name: str) -> pd.DataFrame:
        """
        Load a registered dataset securely.

        :param name: Unique identifier for the dataset in the catalog.
        :type name: str
        :raises ValueError: If the dataset name is not in the registry catalog.
        :raises FileNotFoundError: If the underlying file could not be queried or synced locally.
        :raises NotImplementedError: If the specified file format is not supported.
        :return: A pandas DataFrame containing the dataset content.
        :rtype: pandas.DataFrame
        """
        if name not in self.catalog:
            raise ValueError(f"Dataset {name} not found in registry.")

        info = self.catalog[name]
        raw_path = info["path"]
        path = raw_path if os.path.isabs(raw_path) else os.path.join(self.data_dir, raw_path)

        # In a real environment, we would securely fetch via requests
        # but here we mock the filesystem load for the beta release
        if not os.path.exists(path):
            raise FileNotFoundError(f"Underlying file {path} not synced to {self.data_dir}")

        if info["format"] == "csv":
            frame = pd.read_csv(path)
        elif info["format"] == "excel":
            frame = pd.read_excel(path, engine="openpyxl")
        elif info["format"] == "rds":
            raise NotImplementedError(
                "RDS loading is not supported from Python. Provide a CSV/Excel export or use the R package."
            )
        else:
            raise NotImplementedError("Format not supported.")

        required_variables = info.get("required_variables")
        if required_variables:
            if has_raw_cpads_columns(frame):
                frame = canonicalize_cpads_frame(frame)
            else:
                validate_cpads_frame(frame, strict=True)
        return frame


def dataset_recommendation(key: str, entry: "dict | None" = None) -> str:
    """Human-readable guidance on how to obtain a catalogued dataset.

    Surfaced in :func:`load_dataset` errors and :func:`check_datasets`
    output, so a user who hits an unavailable dataset is told where it
    comes from and what to do -- morie ships code, not the data, but it
    can always point at the source.

    Parameters
    ----------
    key : str
        Catalogue key (e.g. ``"cchs22"``, ``"siu"``).
    entry : dict, optional
        The catalogue entry; looked up from :data:`DATASET_CATALOG`
        when omitted.

    Returns
    -------
    str
        A multi-line recommendation.
    """
    entry = entry if entry is not None else DATASET_CATALOG.get(key, {})
    if not entry:
        return f"Dataset {key!r} is not in the morie catalogue. Run morie.check_datasets() to list catalogued datasets."

    name = entry.get("name", key)
    src = entry.get("source", "?")
    local_path = entry.get("local_path", "")
    rid = entry.get("ckan_resource_id")
    fetcher = entry.get("fetcher")
    lines = [f"How to obtain {key!r} -- {name} (source: {src}):"]

    if rid:
        lines.append(
            "  Published on the Open Government portal. morie fetches it "
            "automatically when online; if you are seeing this, the portal "
            "was unreachable -- retry when connected."
        )
        lines.append(
            f"  CKAN datastore API: https://open.canada.ca/data/api/3/action/datastore_search?resource_id={rid}&limit=5"
        )
    elif fetcher:
        lines.append(
            f"  Fetched on demand via {fetcher}. morie downloads it "
            "automatically when online; if you are seeing this, the remote "
            "source was unreachable -- retry when connected."
        )
    else:
        lines.append(
            "  morie has no public remote source for this dataset -- it is "
            "your own or otherwise restricted data, not redistributed by "
            "morie."
        )
        if local_path:
            rel = Path(local_path)
            tail = Path(*rel.parts[1:]) if rel.parts and rel.parts[0] == "data" else rel
            lines.append(f"  Place the data file at: {(_data_dir_candidates()[0] / tail)}")
            lines.append(
                f"  (catalog path {local_path}: the file goes under the directory MORIE_DATA_DIR names, else under "
                "the per-user data directory; it is read in place and never copied into the cache)"
            )
    return "\n".join(lines)


def check_datasets(
    *,
    probe_remote: bool = False,
    db_path: "str | Path | None" = None,
    timeout: int = 15,
):
    """Audit the availability of every catalogued dataset.

    Classifies each entry in :data:`DATASET_CATALOG` by where it loads
    from:

    * ``bundled`` -- shipped in the built-in morie database.
    * ``cached``  -- present in the user cache.
    * ``local``   -- a local file exists at ``local_path``.
    * ``remote``  -- not available locally, but a CKAN resource id or a
      fetcher is registered, so it can be fetched on demand.
    * ``MISSING`` -- no working source; the entry needs attention (a
      missing file together with an empty ``ckan_resource_id`` and no
      fetcher).

    With ``probe_remote=True`` every ``remote`` dataset is actually
    fetched once, to confirm the link is still live (``remote-ok``) or
    not (``remote-dead``).

    Returns
    -------
    RichResult
        A per-dataset table and per-tier counts; the headline value is
        the number of datasets that load with no attention needed.
    """
    from morie.fn._richresult import RichResult

    builtin = _builtin_db_connect()
    builtin_tables: set = set()
    if builtin is not None:
        try:
            builtin_tables = {
                r[0] for r in builtin.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
        finally:
            builtin.close()

    rows: list = []
    tiers: dict = {}
    warnings: list = []
    missing: list = []
    recommendations: dict = {}

    for key, entry in DATASET_CATALOG.items():
        table = entry.get("table_name", key)
        detail = ""

        # Each availability probe is wrapped: a checker must never crash,
        # even if the cache path or project root is misconfigured.
        in_builtin = table in builtin_tables
        in_cache = False
        if not in_builtin:
            try:
                in_cache = cache_load(table, db_path) is not None
            except Exception:  # noqa: BLE001
                in_cache = False
        has_local = False
        if not in_builtin and not in_cache:
            try:
                lp = Path(entry.get("local_path", "") or "")
                if str(lp) and not lp.is_absolute():
                    lp = _project_root() / lp
                has_local = bool(entry.get("local_path")) and lp.exists()
            except Exception:  # noqa: BLE001
                has_local = False
        has_remote = bool(entry.get("fetcher") or entry.get("ckan_resource_id"))

        if in_builtin:
            tier = "bundled"
        elif in_cache:
            tier = "cached"
        elif has_local:
            tier = "local"
        elif has_remote:
            tier = "remote"
        else:
            tier = "MISSING"
            detail = "no built-in table, no local file, no CKAN id / fetcher"

        if tier == "remote" and probe_remote:
            try:
                df = load_dataset(key, db_path=db_path, timeout=timeout)
                tier, detail = "remote-ok", f"{len(df)} rows fetched"
            except Exception as exc:  # noqa: BLE001
                tier = "remote-dead"
                detail = str(exc).splitlines()[0][:90]

        tiers[tier] = tiers.get(tier, 0) + 1
        if tier in ("MISSING", "remote-dead"):
            warnings.append(f"{key} ({entry.get('name', key)}): {tier} — {detail}")
            missing.append(key)
            recommendations[key] = dataset_recommendation(key, entry)
        link = entry.get("ckan_resource_id") or (
            "fetcher:" + entry["fetcher"] if entry.get("fetcher") else entry.get("local_path", "")
        )
        rows.append([key, str(entry.get("name", ""))[:36], tier, str(link)[:42]])

    ok = sum(tiers.get(t, 0) for t in ("bundled", "cached", "local", "remote", "remote-ok"))
    needs = tiers.get("MISSING", 0) + tiers.get("remote-dead", 0)
    interp = f"{ok} of {len(DATASET_CATALOG)} catalogued datasets have a working source" + (
        f"; {needs} need attention (see warnings)." if needs else " — every catalogued dataset is reachable."
    )
    return RichResult(
        title="morie Dataset Availability Audit",
        summary_lines=[
            ("Catalogued datasets", len(DATASET_CATALOG)),
            ("Reachable", ok),
            ("Need attention", needs),
            ("Remote probed", probe_remote),
        ],
        tables=[
            {
                "title": "Per-dataset availability:",
                "headers": ["key", "name", "tier", "link / source"],
                "rows": rows,
            }
        ],
        warnings=warnings,
        interpretation=interp,
        payload={
            "value": ok,
            "tiers": tiers,
            "n_catalog": len(DATASET_CATALOG),
            "needs_attention": needs,
            "missing": missing,
            "recommendations": recommendations,
        },
    )


def _post_load(key: str, df):
    """Column clean-ups a dataset needs however it arrived (portal, data.rmorie.com, cache)."""
    if key == "siu" and df is not None and "sex_gender_affected" in getattr(df, "columns", ()):
        # the published corpus holds page text cut at the wrong place in this column
        # ("ual assault. the unit's jurisdiction ..."): the few real categories, missing kept missing
        from .siu.analyze import _sex

        df["sex_gender_affected"] = [
            None if v is None or (isinstance(v, float) and v != v) or v == "" else _sex(v)
            for v in df["sex_gender_affected"].tolist()
        ]
    return df


@functools.wraps(_load_dataset_raw)
def load_dataset(key: str, *args, **kwargs):
    df = _load_dataset_raw(key, *args, **kwargs)
    matched = _fuzzy_match_key(key) or key
    return _post_load(matched, df)
