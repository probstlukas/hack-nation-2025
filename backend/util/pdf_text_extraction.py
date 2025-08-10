from functools import lru_cache
import pdfplumber
import pandas as pd
from typing import List, Optional, Union
import re
from typing import List

from tqdm import tqdm


def table_to_dataframe(
    table: List[List[Optional[Union[str, None]]]], header: bool = True
) -> pd.DataFrame:
    """
    Convert a pdfplumber table (List of Lists) to a clean pandas DataFrame.

    Args:
        table: Raw table as List of Lists with str or None.
        header: If True, use the first row as column headers.

    Returns:
        pd.DataFrame: Cleaned DataFrame with no None cells.
    """
    # Find max row length to pad shorter rows
    max_cols = max(len(row) for row in table)

    def clean_cell(content: str):
        s = content.strip()
        s = re.sub(r"[\r\n]+", " ", s)
        return s

    # Pad rows to have equal length and replace None with empty string
    cleaned_rows = [
        [(clean_cell(cell) if cell is not None else "") for cell in row]
        + [""] * (max_cols - len(row))
        for row in table
    ]

    if header:
        df = pd.DataFrame(cleaned_rows[1:], columns=cleaned_rows[0])
    else:
        df = pd.DataFrame(cleaned_rows)

    # Optional: Strip whitespace from all string cells
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    return df


def is_table_empty(table: list[list[str | None]]) -> bool:
    for row in table:
        for cell in row:
            if cell and cell.strip() != "":
                return False
    return True


def is_table_useless(table: list[list[str | None]], threshold=0.9) -> bool:
    total_cells = 0
    empty_cells = 0

    for row in table:
        for cell in row:
            total_cells += 1
            if not cell or cell.strip() == "":
                empty_cells += 1

    if total_cells == 0:
        return True  # no data at all

    empty_ratio = empty_cells / total_cells
    return empty_ratio >= threshold


@lru_cache(maxsize=128)
def parse_pdf_with_nice_tables(pdf_path: str):
    with pdfplumber.open(pdf_path) as pdf:
        all_pages_text = []
        for i, page in enumerate(tqdm(pdf.pages)):
            raw_page_text = page.extract_text()

            page_tables = [
                table_to_dataframe(table).to_csv(index=False, sep="|")
                for table in page.extract_tables()
            ]

            page_text = f"Page Number {i}\n"
            if raw_page_text:
                page_text += raw_page_text

            if len(page_tables) > 0:
                page_tables_text = "\n".join(page_tables)
                page_text += "\n\n" + page_tables_text

            all_pages_text.append(page_text)

        full_document_text = "\n\n".join(all_pages_text)
    return full_document_text
