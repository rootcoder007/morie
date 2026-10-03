# SPDX-License-Identifier: AGPL-3.0-or-later
"""Canadian Institute for Health Information (CIHI) ingest.

CIHI publishes indicator data-table workbooks as ``.xlsx`` files at
stable URLs under ``cihi.ca/sites/default/files/document/...`` (the
"Accessible version" / "data file" link on each indicator page).

:func:`fetch_cihi_xlsx` downloads one and returns a DataFrame.  It is
wired into :data:`morie.data.DATASET_CATALOG` through the ``fetcher``
mechanism, so ``morie.load_dataset()`` can fetch a CIHI indicator
table on demand.
"""

from __future__ import annotations

from typing import Any

__all__ = ["fetch_cihi_xlsx"]


def _pick_data_sheet(xl, **read_excel_kwargs):
    """Return the largest sheet of a workbook as a DataFrame.

    CIHI workbooks lead with a small "Notes"/"Introduction" sheet; the
    data lives on a later, much larger sheet.  Picking the sheet with
    the most cells skips the notes page without hard-coding names.
    """
    best_df, best_cells = None, -1
    for name in xl.sheet_names:
        df = xl.parse(name, **read_excel_kwargs)
        cells = df.shape[0] * df.shape[1]
        if cells > best_cells:
            best_df, best_cells = df, cells
    return best_df


def fetch_cihi_xlsx(url: str, *, sheet: Any = None, timeout: float = 120.0, **read_excel_kwargs: Any):
    """Download a CIHI indicator ``.xlsx`` data table and return a DataFrame.

    Parameters
    ----------
    url : str
        Direct URL of the CIHI ``.xlsx`` data table.
    sheet : int or str, optional
        Worksheet index or name to read.  When omitted (the default),
        the largest sheet is used -- CIHI workbooks lead with a small
        notes sheet, so the largest sheet is the data.  Pass an explicit
        ``sheet`` via the catalog entry's ``fetcher_args`` to override.
    timeout : float
        HTTP timeout in seconds.
    **read_excel_kwargs
        Forwarded to :func:`pandas.read_excel`.

    Returns
    -------
    pandas.DataFrame
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - httpx is a core dep
        raise ImportError("fetch_cihi_xlsx needs httpx") from exc

    import tempfile

    from morie._progress import Progress

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/cihi.xlsx"
        with httpx.Client(timeout=timeout, follow_redirects=True) as client, client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length") or 0) or None
            with open(path, "wb") as fh, Progress(url.rsplit("/", 1)[-1], total) as prog:
                for chunk in resp.iter_bytes(1 << 20):
                    fh.write(chunk)
                    prog.update(len(chunk))
        return _read_xlsx_streaming(path, sheet, **read_excel_kwargs)


def _read_xlsx_streaming(path: str, sheet: Any = None, **read_excel_kwargs: Any):
    """Read one sheet row by row (openpyxl read-only mode) so a large workbook
    never becomes millions of cell objects at once; the largest sheet by
    declared dimension is the data sheet when ``sheet`` is not given."""
    try:
        import openpyxl
    except ImportError:
        from morie.fn import _frame_core as pd

        xl = pd.ExcelFile(path)
        return _pick_data_sheet(xl, **read_excel_kwargs) if sheet is None else xl.parse(sheet, **read_excel_kwargs)
    from morie.fn import _frame_core as pd

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        if sheet is None:

            def _cells(ws):
                try:
                    return (ws.max_row or 0) * (ws.max_column or 0)
                except Exception:  # noqa: BLE001
                    return 0

            ws = max(wb.worksheets, key=_cells)
        else:
            ws = wb[sheet] if isinstance(sheet, str) else wb.worksheets[int(sheet)]
        rows = ws.iter_rows(values_only=True)
        header = None
        records = []
        for row in rows:
            if header is None:
                if row:
                    last = max((i for i, v in enumerate(row) if v is not None), default=-1)
                    row = tuple(row[: last + 1])  # the sheet may declare 16,384 columns; count the used ones
                if row and sum(v is not None for v in row) >= max(1, len(row) // 2):
                    header = [str(v) if v is not None else f"col{i}" for i, v in enumerate(row)]
                continue
            if row is None or all(v is None for v in row):
                continue
            records.append(list(row[: len(header)]) + [None] * max(0, len(header) - len(row)))
        if header is None:
            return pd.DataFrame()
        return pd.DataFrame(records, columns=header)
    finally:
        wb.close()
