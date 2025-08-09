"""
Utilities for Stage 1: load PDF filings, extract text, split into chunks, and parse metadata.

Functions:
- get_pdf_text(doc): Backward-compatible loader returning a list[Document] (LangChain)
- load_pages(doc_or_path): Return list[Document] pages via PyMuPDFLoader
- get_text(doc_or_path): Return a single concatenated string with page separators
- get_page_texts(doc_or_path): Return list[str] per page
- chunk_documents(doc_or_path, chunk_size=1024, overlap=30): Return chunked Documents for vector store
- parse_stock_and_year(name): Extract (company, year?) using FinanceBench index if available; year can be None
- detect_doc_type(name): Infer filing type from FinanceBench index or filename (10K, 10Q, 8K, EARNINGS, ANNUALREPORT)
- load_financebench_index(jsonl_path?): Load mapping doc_name -> metadata from datasets/financebench/financebench_document_information.jsonl
- get_doc_metadata(name): Get full metadata dict for a doc from the index

Conventions:
- Prefers a project-local pdfs/ directory rooted at CWD for ad-hoc experiments.
- Also supports the repo dataset location datasets/financebench/pdfs.
- doc parameter may be either a bare doc name (without .pdf) or a full path.
"""
from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# LangChain loader import with compatibility between versions
try:  # langchain >= 0.1+ split
    from langchain_community.document_loaders import PyMuPDFLoader  # type: ignore
except Exception:  # pragma: no cover
    from langchain.document_loaders import PyMuPDFLoader  # type: ignore

# Text splitter + Document imports compatible across versions
try:  # modern packages
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
except Exception:  # fallback to old path
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

try:  # modern Document location
    from langchain_core.documents import Document  # type: ignore
except Exception:  # fallback
    from langchain.schema import Document  # type: ignore

##############################################################################
# PATHS (align with both notebook experiments and repo dataset layout)
##############################################################################
REPO_ROOT = Path(__file__).resolve().parents[1]
FINANCEBENCH_ROOT = REPO_ROOT / "datasets" / "financebench"
PATH_PDFS_CWD = Path(os.getcwd()) / "pdfs"
PATH_PDFS_FB = FINANCEBENCH_ROOT / "pdfs"


def _ensure_pdf_path(doc_or_path: str | Path) -> Path:
    """Resolve a doc name or path to an existing .pdf path.

    Accepts either:
    - "AMD_2017_10K" (no extension) -> {CWD}/pdfs/AMD_2017_10K.pdf or datasets/financebench/pdfs/...
    - ".../pdfs/AMD_2017_10K.pdf" (full path)
    Raises FileNotFoundError if not found.
    """
    p = Path(doc_or_path)
    if p.suffix.lower() == ".pdf" and p.exists():
        return p

    # If bare name or missing extension, check multiple candidate roots
    name = p.name
    if not name.endswith(".pdf"):
        name = name + ".pdf"

    for root in (PATH_PDFS_CWD, PATH_PDFS_FB):
        candidate = root / name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"PDF not found for '{doc_or_path}'. Looked under: {PATH_PDFS_CWD} and {PATH_PDFS_FB}"
    )


##############################################################################
# FinanceBench metadata index (doc_name -> {company, doc_period, doc_type, ...})
##############################################################################
INDEX_JSONL_DEFAULT = FINANCEBENCH_ROOT / "financebench_document_information.jsonl"


def load_financebench_index(jsonl_path: Path | None = None) -> Dict[str, Dict]:
    """Load the FinanceBench JSONL index as a dict keyed by lowercase doc_name.

    Returns an empty dict if the file is missing.
    """
    path = Path(jsonl_path) if jsonl_path else INDEX_JSONL_DEFAULT
    if not path.exists():
        return {}
    index: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = str(rec.get("doc_name", "")).strip()
            if not name:
                continue
            index[name.lower()] = rec
    return index


def get_doc_metadata(name: str | Path, index: Dict[str, Dict] | None = None) -> Optional[Dict]:
    """Get metadata dict for a document using the FinanceBench index.

    name can be a full path or a bare doc name; matching is done on the stem.
    """
    if index is None:
        index = load_financebench_index()
    key = Path(name).stem.lower()
    return index.get(key)


def _normalize_doc_type(value: Optional[str]) -> str:
    if not value:
        return "UNKNOWN"
    up = value.upper()
    if up in {"10K", "10-Q", "10Q", "8-K", "8K"}:
        return up.replace("-", "")  # normalize 10-Q -> 10Q, 8-K -> 8K
    if "EARNINGS" in up:
        return "EARNINGS"
    if "ANNUAL" in up:
        return "ANNUALREPORT"
    return "UNKNOWN"


def parse_stock_and_year(name: str | Path) -> Tuple[str, Optional[int]]:
    """Parse company and year using FinanceBench index if available, else fallback.

    Example: 'AMD_2017_10K.pdf' -> ("AMD", 2017)
             'PEPSICO_2023Q2_EARNINGS.pdf' -> ("PepsiCo" or "PEPSICO", 2023)
    """
    meta = get_doc_metadata(name)
    if meta:
        company = str(meta.get("company", "")).strip() or Path(name).stem.split("_", 1)[0]
        year = meta.get("doc_period")
        try:
            year_int: Optional[int] = int(year) if year is not None else None
        except Exception:
            year_int = None
        return company, year_int

    # Fallback to filename heuristics
    stem = Path(name).stem
    stock = stem.split("_", 1)[0] if "_" in stem else stem
    m = re.search(r"(20\d{2}|19\d{2})", stem)
    year = int(m.group(1)) if m else None
    return stock, year


def detect_doc_type(name: str | Path) -> str:
    """Guess filing/report type, preferring FinanceBench index when present.

    Returns one of: '10K', '10Q', '8K', 'EARNINGS', 'ANNUALREPORT', or 'UNKNOWN'.
    """
    meta = get_doc_metadata(name)
    if meta:
        return _normalize_doc_type(str(meta.get("doc_type")))

    up = Path(name).stem.upper()
    if "10K" in up:
        return "10K"
    if "10Q" in up:
        return "10Q"
    if re.search(r"\b8K\b|_8K_", up):
        return "8K"
    if "EARNINGS" in up:
        return "EARNINGS"
    if "ANNUALREPORT" in up or "ANNUAL_REPORT" in up or "ANNUAL" in up:
        return "ANNUALREPORT"
    return "UNKNOWN"


def load_pages(doc_or_path: str | Path) -> List[Document]:
    """Load a PDF into LangChain Documents, one per page, using PyMuPDFLoader."""
    pdf_path = _ensure_pdf_path(doc_or_path)
    loader = PyMuPDFLoader(str(pdf_path))
    return loader.load()


def get_page_texts(doc_or_path: str | Path) -> List[str]:
    """Return raw text for each page as list[str]."""
    docs = load_pages(doc_or_path)
    return [d.page_content for d in docs]


def _page_separators(page_idx: int, total: int, file_name: str) -> str:
    return f"\n\n===== Page {page_idx+1}/{total} : {file_name} =====\n\n"


def _clean_text(text: str) -> str:
    """Light cleanup for downstream analysis without being too aggressive."""
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_text(doc_or_path: str | Path, add_separators: bool = True, clean: bool = True) -> str:
    """Return the full document text as a single string.

    - add_separators: add page headers to preserve structure
    - clean: apply light normalization
    """
    pdf_path = _ensure_pdf_path(doc_or_path)
    pages = load_pages(pdf_path)
    total = len(pages)

    parts: List[str] = []
    for i, d in enumerate(pages):
        txt = d.page_content
        if clean:
            txt = _clean_text(txt)
        if add_separators:
            parts.append(_page_separators(i, total, Path(pdf_path).name))
        parts.append(txt)

    return "".join(parts)


def chunk_documents(doc_or_path: str | Path, chunk_size: int = 1024, overlap: int = 30) -> List[Document]:
    """Split a PDF into chunked Documents for vector stores.

    Returns list[Document] with metadata preserved from the original pages.
    """
    pages = load_pages(doc_or_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks: List[Document] = []
    for d in pages:
        # Split page content and propagate metadata (page number, source path, etc.)
        for chunk in splitter.split_text(d.page_content):
            chunks.append(Document(page_content=chunk, metadata=dict(d.metadata)))
    return chunks


##############################################################################
# Structured extraction: financial metrics from filings/earnings PDFs
##############################################################################

def _units_multiplier_from_text(text: str) -> float:
    """Heuristically detect unit scale from surrounding text.

    Looks for phrases like "($ in millions)", "Amounts in thousands", etc.
    Returns a multiplier to convert to raw currency units.
    """
    if not text:
        return 1.0
    up = text.lower()
    if re.search(r"\bin\s*billions\b|\b\$\s*in\s*billions\b", up):
        return 1e9
    if re.search(r"\bin\s*millions\b|\b\$\s*in\s*millions\b", up):
        return 1e6
    if re.search(r"\bin\s*thousands\b|\b\$\s*in\s*thousands\b", up):
        return 1e3
    return 1.0


def _to_number(raw: str) -> Optional[float]:
    """Parse a table cell numeric value.

    Handles commas, parentheses for negatives, em-dash/NA as None.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if s in {"", "—", "–", "-", "N/A", "NA", "n/a"}:
        return None
    neg = False
    # Parentheses denote negatives in financials
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    # Remove currency symbols and footnote markers
    s = re.sub(r"[^0-9.,-]", "", s)
    if s.count(",") > 0:
        s = s.replace(",", "")
    try:
        val = float(s)
        return -val if neg else val
    except Exception:
        return None


_METRIC_SYNONYMS = {
    "revenue": [
        "revenue", "total revenue", "net sales", "sales and other revenue", "sales", "net revenues"
    ],
    "cost_of_revenue": [
        "cost of revenue", "cost of sales", "costs of goods sold", "cogs", "costs and expenses"
    ],
    "gross_profit": [
        "gross profit", "gross margin"
    ],
    "operating_income": [
        "operating income", "income from operations", "operating earnings", "operating loss"
    ],
    "net_income": [
        "net income", "net earnings", "net loss", "income (loss) attributable to", "net income attributable"
    ],
    "eps_basic": [
        "earnings per share—basic", "earnings per share - basic", "basic earnings per share", "basic eps"
    ],
    "eps_diluted": [
        "earnings per share—diluted", "earnings per share - diluted", "diluted earnings per share", "diluted eps"
    ],
}


def _label_to_metric(label: str) -> Optional[str]:
    lab = re.sub(r"\s+", " ", label.strip().lower())
    # Remove trailing footnote refs like (1), (a)
    lab = re.sub(r"\s*\([^)]*\)$", "", lab)
    for key, synonyms in _METRIC_SYNONYMS.items():
        for s in synonyms:
            if s in lab:
                return key
    return None


def extract_financial_metrics(doc_or_path: str | Path) -> dict:
    """Extract key financial metrics from a PDF using table parsing with pdfplumber.

    Returns a dict with keys when found: revenue, cost_of_revenue, gross_profit,
    operating_income, net_income, eps_basic, eps_diluted, currency (if detectable),
    units_multiplier, and basic metadata (company, year, type, file).

    Falls back to regex-on-text if tables fail.
    """
    pdf_path = _ensure_pdf_path(doc_or_path)

    # Metadata
    meta = get_doc_metadata(pdf_path)
    company, year = (meta.get("company"), meta.get("doc_period")) if meta else parse_stock_and_year(pdf_path)
    try:
        year = int(year) if year is not None else None
    except Exception:
        year = None
    doc_type = detect_doc_type(pdf_path)

    best = {
        "company": company,
        "year": year,
        "doc_type": doc_type,
        "file": Path(pdf_path).name,
        "currency": "USD",  # default assumption when "$" present; many reports are USD
        "units_multiplier": 1.0,
    }

    # Lazy import to avoid hard dependency if user only wants text
    try:
        import pdfplumber  # type: ignore
    except Exception:
        # Fallback to regex-only
        text = get_text(pdf_path, add_separators=False, clean=True)
        best.update(_extract_metrics_regex(text))
        return best

    most_fields = 0
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                # Units from nearby text
                page_text = page.extract_text() or ""
                mult = _units_multiplier_from_text(page_text)

                # Extract tables
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    tables = []

                for tbl in tables:
                    if not tbl or len(tbl) < 2:
                        continue
                    # Heuristic: first column is labels, pick the numeric column with most parseable values
                    rows = [[(c or "").strip() for c in r] for r in tbl]
                    # Normalize row lengths
                    max_len = max(len(r) for r in rows)
                    rows = [r + [""] * (max_len - len(r)) for r in rows]

                    # choose value column index
                    numeric_counts = []
                    for j in range(1, max_len):
                        cnt = 0
                        for i in range(1, len(rows)):
                            if _to_number(rows[i][j]) is not None:
                                cnt += 1
                        numeric_counts.append((cnt, j))
                    if not numeric_counts:
                        continue
                    numeric_counts.sort(reverse=True)
                    _, val_col = numeric_counts[0]

                    found = {}
                    for i in range(1, len(rows)):
                        label = rows[i][0]
                        metric = _label_to_metric(label)
                        if not metric:
                            continue
                        val_raw = _to_number(rows[i][val_col])
                        if val_raw is None:
                            continue
                        # EPS values are per-share, do not scale by units multiplier
                        if metric.startswith("eps_"):
                            found[metric] = float(val_raw)
                        else:
                            found[metric] = float(val_raw) * (mult or 1.0)

                    # If we found enough fields, keep the best result seen
                    if len(found) > most_fields:
                        most_fields = len(found)
                        best.update({"units_multiplier": mult or 1.0})
                        best.update(found)

    except Exception:
        pass  # fall back below

    # If table parsing was weak, try regex over full text as backup
    if most_fields < 2:
        text = get_text(pdf_path, add_separators=False, clean=True)
        best.update(_extract_metrics_regex(text))

    return best


def _extract_metrics_regex(text: str) -> dict:
    """Simpler regex backup for revenue/net income/EPS.

    Not perfect, but provides signal if tables fail.
    """
    UNIT_MUL = {
        "billion": 1e9, "bn": 1e9, "b": 1e9,
        "million": 1e6, "m": 1e6,
        "thousand": 1e3, "k": 1e3,
        None: 1.0, "": 1.0
    }

    def _to_amt(v, u):
        try:
            return float(v.replace(",", "")) * UNIT_MUL.get((u or "").lower(), 1.0)
        except Exception:
            return None

    REV_PAT = re.compile(r"(?:revenue|net sales|total sales)[^$]{0,80}\$?([0-9][\d,\.]*)\s*(billion|bn|b|million|m|thousand|k)?", re.I)
    NI_PAT  = re.compile(r"(?:net (?:income|earnings)|profit)[^$]{0,80}\$?([0-9][\d,\.]*)\s*(billion|bn|b|million|m|thousand|k)?", re.I)
    LOSS_NEAR = re.compile(r"(loss|negative)", re.I)
    EPS_PAT = re.compile(r"(?:diluted|basic)?\s*EPS[^$\n]{0,40}\$?([0-9][\d\.]+)", re.I)

    out = {}
    m = REV_PAT.search(text)
    if m:
        out["revenue"] = _to_amt(m.group(1), m.group(2))

    n = NI_PAT.search(text)
    if n:
        amt = _to_amt(n.group(1), n.group(2))
        if amt is not None:
            start = max(n.start()-40, 0); end = min(n.end()+40, len(text))
            if LOSS_NEAR.search(text[start:end]):
                amt = -abs(amt)
            out["net_income"] = amt

    e = EPS_PAT.search(text)
    if e:
        try:
            out["eps_diluted"] = float(e.group(1))
        except Exception:
            pass

    return out


def extract_financials_dataframe(paths: Optional[List[str | Path]] = None):
    """Aggregate structured metrics across multiple PDFs into a DataFrame.

    If paths is None, scans datasets/financebench/pdfs.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pandas is required for extract_financials_dataframe") from e

    if paths is None:
        roots = [PATH_PDFS_CWD, PATH_PDFS_FB]
        pdfs: List[Path] = []
        for r in roots:
            if r.exists():
                pdfs.extend(sorted(r.glob("*.pdf")))
        paths = pdfs

    records = []
    for p in paths:
        try:
            rec = extract_financial_metrics(p)
            records.append(rec)
        except Exception:
            continue

    df = pd.DataFrame.from_records(records)
    # Ensure numeric dtype for known metrics
    for col in [
        "revenue", "cost_of_revenue", "gross_profit", "operating_income", "net_income", "eps_basic", "eps_diluted"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# Backward-compatible helper for the evaluation notebook
def get_pdf_text(doc):
    """Return list[Document] pages for the given doc name (no extension) or path."""
    return load_pages(doc)


##############################################################################
# Helpers: currency + unit detection, sectioning, and sentiment analysis
##############################################################################

def detect_currency(text: str) -> Optional[str]:
    """Best-effort currency detection from text.

    Returns ISO-like code when possible (e.g., 'USD'), else None.
    """
    if not text:
        return None
    up = text.upper()
    if re.search(r"\bUSD\b|US\$|U\.S\.\s*DOLLARS|UNITED\s+STATES\s+DOLLARS", up):
        return "USD"
    if "$" in text:
        # Most filings here are USD; use USD when dollar symbol appears
        return "USD"
    if re.search(r"\bEUR\b|EURO\b", up):
        return "EUR"
    if re.search(r"\bGBP\b|POUNDS?\b", up):
        return "GBP"
    if re.search(r"\bJPY\b|YEN\b", up):
        return "JPY"
    return None


def _global_units_multiplier_from_doc_text(text: str) -> float:
    """Infer a document-level units multiplier from the first couple of pages of text."""
    return _units_multiplier_from_text(text)


def split_sections(full_text: str) -> dict:
    """Split a 10-K/10-Q-like document into coarse sections using common headings.

    Returns dict of {section_name: text} for keys like 'MD&A', 'Risk Factors', 'Business', 'Financial Statements',
    plus generic 'Guidance'/'Outlook' blocks for earnings docs when present.
    """
    text = full_text
    up = text.upper()

    # Anchors for 10-K/Q items
    anchors = [
        ("BUSINESS", r"\bITEM\s+1\.?\s+BUSINESS\b"),
        ("RISK_FACTORS", r"\bITEM\s+1A\.?\s+RISK\s+FACTORS\b"),
        ("PROPERTIES", r"\bITEM\s+2\.?\s+PROPERTIES\b"),
        ("LEGAL_PROCEEDINGS", r"\bITEM\s+3\.?\s+LEGAL\s+PROCEEDINGS\b"),
        ("MD&A", r"\bITEM\s+7\.?\s+MANAGEMENT'S\s+DISCUSSION\s+AND\s+ANALYSIS\b|\bMD&A\b"),
        ("MARKET_RISK", r"\bITEM\s+7A\.?\s+QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES\s+ABOUT\s+MARKET\s+RISK\b"),
        ("FINANCIAL_STATEMENTS", r"\bITEM\s+8\.?\s+FINANCIAL\s+STATEMENTS\b|CONSOLIDATED\s+STATEMENTS\b"),
    ]

    # Find start positions
    spans = []
    for name, pat in anchors:
        m = re.search(pat, up)
        if m:
            spans.append((name, m.start()))
    # Add generic sections for earnings: Guidance/Outlook
    for name, pat in [("GUIDANCE", r"\bGUIDANCE\b"), ("OUTLOOK", r"\bOUTLOOK\b")]:
        m = re.search(pat, up)
        if m:
            spans.append((name, m.start()))

    if not spans:
        return {"FULL": text}

    spans.sort(key=lambda x: x[1])
    out = {}
    for i, (name, start) in enumerate(spans):
        end = spans[i + 1][1] if i + 1 < len(spans) else len(text)
        out[name] = text[start:end].strip()
    return out


def analyze_sentiment(text_or_path: str | Path, method: str = "vader") -> dict:
    """Compute sentiment overall and per-section.

    method='vader' uses NLTK VADER (lightweight). If 'finbert' and transformers are installed,
    method='finbert' will attempt to use a finance-tuned model for headline-style sentiment.
    """
    # Load text
    if isinstance(text_or_path, (str, Path)) and Path(text_or_path).suffix.lower() == ".pdf":
        full_text = get_text(text_or_path, add_separators=False, clean=True)
    else:
        full_text = str(text_or_path)

    sections = split_sections(full_text)

    result = {"method": method, "overall": None, "sections": {}}

    if method.lower() == "finbert":
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
            import torch  # type: ignore
            model_name = "yiyanghkust/finbert-tone"
            tok = AutoTokenizer.from_pretrained(model_name)
            mdl = AutoModelForSequenceClassification.from_pretrained(model_name)

            def score(txt: str) -> dict:
                # Truncate long text for efficiency; FinBERT is trained on sentence/short text
                chunk = txt[:1024]
                inputs = tok(chunk, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    logits = mdl(**inputs).logits[0]
                probs = torch.softmax(logits, dim=-1).tolist()
                labels = ["negative", "neutral", "positive"]
                return {k: v for k, v in zip(labels, probs)}

            # Overall
            result["overall"] = score(full_text)
            # Sections
            for k, v in sections.items():
                result["sections"][k] = score(v)
            return result
        except Exception:
            # fallback to vader
            method = "vader"

    # VADER
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
        import nltk  # type: ignore
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except Exception:
            nltk.download('vader_lexicon', quiet=True)
        sia = SentimentIntensityAnalyzer()
    except Exception as e:  # pragma: no cover
        raise RuntimeError("NLTK VADER is required for sentiment analysis. Install nltk.") from e

    def score_vader(txt: str) -> dict:
        d = sia.polarity_scores(txt)
        return {"negative": d.get("neg"), "neutral": d.get("neu"), "positive": d.get("pos"), "compound": d.get("compound")}

    result["method"] = "vader"
    result["overall"] = score_vader(full_text)
    for k, v in sections.items():
        result["sections"][k] = score_vader(v)
    return result


##############################################################################
# Strengthen regex fallback: require currency/units indicators to avoid tiny matches
##############################################################################

def _has_amount_indicators(prefix: str, number_str: str, unit: Optional[str]) -> bool:
    has_dollar = "$" in prefix
    has_unit = bool(unit)
    has_comma = "," in number_str
    has_large = len(re.sub(r"[^0-9]", "", number_str)) >= 4  # at least 4 digits total
    return has_dollar or has_unit or has_comma or has_large


# Patch the regex extractor to use document-level units/currency
_old_extract_metrics_regex = _extract_metrics_regex


def _extract_metrics_regex(text: str) -> dict:  # type: ignore[override]
    UNIT_MUL = {
        "billion": 1e9, "bn": 1e9, "b": 1e9,
        "million": 1e6, "m": 1e6,
        "thousand": 1e3, "k": 1e3,
        None: 1.0, "": 1.0
    }

    def _to_amt(v, u):
        try:
            return float(v.replace(",", "")) * UNIT_MUL.get((u or "").lower(), 1.0)
        except Exception:
            return None

    # Capture 1-3 digits with thousands separators or decimals; avoid picking lone small integers without indicators
    NUM = r"([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]{4,})"
    REV_PAT = re.compile(rf"((?:revenue|net sales|total sales)[^$\n]{{0,100}})(?:\$\s*)?{NUM}\s*(billion|bn|b|million|m|thousand|k)?", re.I)
    NI_PAT  = re.compile(rf"((?:net (?:income|earnings)|profit)[^$\n]{{0,100}})(?:\$\s*)?{NUM}\s*(billion|bn|b|million|m|thousand|k)?", re.I)
    EPS_PAT = re.compile(r"(?:earnings per share|EPS)[^\n]{0,60}?([0-9]{1,2}\.[0-9]{1,3})", re.I)
    LOSS_NEAR = re.compile(r"(loss|negative)", re.I)

    out = {}

    m = REV_PAT.search(text)
    if m:
        prefix, num, unit = m.group(1), m.group(2), m.group(3)
        if _has_amount_indicators(prefix, num, unit):
            out["revenue"] = _to_amt(num, unit)

    n = NI_PAT.search(text)
    if n:
        prefix, num, unit = n.group(1), n.group(2), n.group(3)
        if _has_amount_indicators(prefix, num, unit):
            amt = _to_amt(num, unit)
            if amt is not None:
                start = max(n.start()-40, 0); end = min(n.end()+40, len(text))
                if LOSS_NEAR.search(text[start:end]):
                    amt = -abs(amt)
                out["net_income"] = amt

    e = EPS_PAT.search(text)
    if e:
        try:
            out["eps_diluted"] = float(e.group(1))
        except Exception:
            pass

    return out


##############################################################################
# Enhance extract_financial_metrics to use document-level units/currency as defaults
##############################################################################
_old_extract_financial_metrics = extract_financial_metrics


def extract_financial_metrics(doc_or_path: str | Path) -> dict:  # type: ignore[override]
    pdf_path = _ensure_pdf_path(doc_or_path)

    # Metadata
    meta = get_doc_metadata(pdf_path)
    company, year = (meta.get("company"), meta.get("doc_period")) if meta else parse_stock_and_year(pdf_path)
    try:
        year = int(year) if year is not None else None
    except Exception:
        year = None
    doc_type = detect_doc_type(pdf_path)

    # Read quick text from first pages to guess currency and units
    head_text = get_text(pdf_path, add_separators=False, clean=True)[:4000]
    doc_units = _global_units_multiplier_from_doc_text(head_text)
    currency = detect_currency(head_text) or "USD"

    best = {
        "company": company,
        "year": year,
        "doc_type": doc_type,
        "file": Path(pdf_path).name,
        "currency": currency,
        "units_multiplier": doc_units or 1.0,
    }

    # Try table extraction
    try:
        import pdfplumber  # type: ignore
    except Exception:
        # Fallback to regex-only
        text = get_text(pdf_path, add_separators=False, clean=True)
        best.update(_extract_metrics_regex(text))
        return best

    most_fields = 0
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                # Prefer page-specific units if present, else doc-level
                mult = _units_multiplier_from_text(page_text) or doc_units or 1.0

                try:
                    tables = page.extract_tables() or []
                except Exception:
                    tables = []

                for tbl in tables:
                    if not tbl or len(tbl) < 2:
                        continue
                    rows = [[(c or "").strip() for c in r] for r in tbl]
                    max_len = max(len(r) for r in rows)
                    rows = [r + [""] * (max_len - len(r)) for r in rows]

                    numeric_counts = []
                    for j in range(1, max_len):
                        cnt = 0
                        for i in range(1, len(rows)):
                            if _to_number(rows[i][j]) is not None:
                                cnt += 1
                        numeric_counts.append((cnt, j))
                    if not numeric_counts:
                        continue
                    numeric_counts.sort(reverse=True)
                    _, val_col = numeric_counts[0]

                    found = {}
                    for i in range(1, len(rows)):
                        label = rows[i][0]
                        metric = _label_to_metric(label)
                        if not metric:
                            continue
                        val_raw = _to_number(rows[i][val_col])
                        if val_raw is None:
                            continue
                        if metric.startswith("eps_"):
                            found[metric] = float(val_raw)
                        else:
                            found[metric] = float(val_raw) * (mult or 1.0)

                    if len(found) > most_fields:
                        most_fields = len(found)
                        best.update({"units_multiplier": mult or 1.0})
                        best.update(found)
    except Exception:
        pass

    if most_fields < 2:
        text = get_text(pdf_path, add_separators=False, clean=True)
        # Apply regex with awareness of doc-level units (implicitly via units words)
        best.update(_extract_metrics_regex(text))

    return best


__all__ = [
    "get_pdf_text",
    "load_pages",
    "get_text",
    "get_page_texts",
    "chunk_documents",
    "parse_stock_and_year",
    "detect_doc_type",
    "load_financebench_index",
    "get_doc_metadata",
    "extract_financial_metrics",
    "extract_financials_dataframe",
    "detect_currency",
    "split_sections",
    "analyze_sentiment",
]

if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="Demo: extract text and financial metrics from a PDF")
    parser.add_argument("doc", help="Doc name (e.g., AMD_2017_10K) or full path to a .pdf")
    parser.add_argument("--head", type=int, default=600, help="Show first N characters of text (default: 600)")
    parser.add_argument("--no-text", action="store_true", help="Do not print text preview")
    args = parser.parse_args()

    pdf_path = _ensure_pdf_path(args.doc)

    company, year = parse_stock_and_year(pdf_path)
    doc_type = detect_doc_type(pdf_path)

    print(f"File: {pdf_path}")
    print(f"Company: {company} | Year: {year} | Type: {doc_type}")

    if not args.no_text:
        txt = get_text(pdf_path, add_separators=False, clean=True)
        preview = txt[: args.head].replace("\n", " ")
        print(f"Text preview ({len(txt)} chars total):\n{preview}...")

    metrics = extract_financial_metrics(pdf_path)
    print("\nExtracted metrics:")
    print(_json.dumps({k: v for k, v in metrics.items() if k not in {"file"}}, indent=2, ensure_ascii=False))