"""
FinDocGPT - AI for Financial Document Analysis & Investment Strategy
FastAPI Backend for Stage 1: Document Q&A Feature

This is the API backend for the FinDocGPT application implementing
the AkashX.ai challenge requirements. Frontend is a separate React app.
"""
import os
import asyncio
# Load .env early so environment variables are available
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass
import random
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import logging
from datetime import datetime, date  # Added for recency weighting

# Import our PDF processing utilities
from backend.util.pdf2text import (
    load_financebench_index, 
    get_doc_metadata,
    parse_stock_and_year,
    detect_doc_type,
    get_text,
    chunk_documents,
    extract_financial_metrics,
    PATH_PDFS_CWD,
    PATH_PDFS_FB,
    PATH_PDFS_GENERIC,
)
from backend.util.text2sentiment import text2sentiment, SentimentSummary
from backend.util.retrieval import answer_question
from backend.util.forecast import run_forecast
from backend.util.news_sentiment import analyze_company_news_sentiment, NewsSentimentResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
if not NEWS_API_KEY:
    logger = logging.getLogger(__name__)
    logger.warning("NEWS_API_KEY is not set; news sentiment features may be disabled.")

# Pydantic models for request/response validation
class FinancialMetrics(BaseModel):
    revenue: Optional[float] = None
    cost_of_revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    eps_basic: Optional[float] = None
    eps_diluted: Optional[float] = None
    currency: Optional[str] = "USD"
    units_multiplier: Optional[float] = None

class Document(BaseModel):
    id: str
    name: str
    company: str
    sector: str
    doc_type: str
    period: str
    year: str
    doc_link: Optional[str] = None
    financial_metrics: Optional[FinancialMetrics] = None

class DocumentText(BaseModel):
    text: str
    full_length: int
    truncated: bool

class QARequest(BaseModel):
    document_id: str = Field(..., description="Document ID to ask about")
    question: str = Field(..., min_length=1, description="Question to ask about the document")

class QAResponse(BaseModel):
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: List[str]
    processing_time: float

class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    message: Optional[str] = None

class InvestmentRecommendation(BaseModel):
    action: str
    confidence: float
    reasoning: str
    target_price: Optional[float] = None
    risk_level: str = "MEDIUM"
    components: Optional[Dict[str, Any]] = None

class InvestmentRequest(BaseModel):
    document_id: Optional[str] = Field(None, description="Optional ID of a FinanceBench document")
    document_ids: Optional[List[str]] = Field(None, description="Optional list of FinanceBench document IDs to aggregate")
    ticker: Optional[str] = Field(None, description="Optional market ticker to run forecast (e.g. AAPL)")
    period: str = Field("5y", description="History period for forecast")
    horizon: int = Field(5, ge=1, le=30, description="Forecast horizon in trading days")
    model: str = Field("lstm", description="Forecast model: rf|prophet|lstm")
    include_news: bool = Field(True, description="Include recent news sentiment")

# Utility: derive an approximate document date from FinanceBench metadata
def _doc_date_for_id(doc_id: str) -> Optional[datetime]:
    try:
        meta = FINANCEBENCH_INDEX.get(doc_id) or {}
        year = meta.get("year")
        period = str(meta.get("period", ""))
        y = None
        if isinstance(year, (int, float)):
            y = int(year)
        else:
            try:
                y = int(str(year)) if year is not None else None
            except Exception:
                y = None
        if y is None:
            # Try to infer year from id like COMPANY_2019_10K.pdf or COMPANY_2023Q2_10Q.pdf
            import re
            m = re.search(r"(20\d{2}|19\d{2})", doc_id)
            if m:
                y = int(m.group(1))
        if y is None:
            return None
        # Quarter handling
        import re
        q = re.search(r"Q([1-4])", period or "")
        if not q:
            q = re.search(r"Q([1-4])", doc_id)
        if q:
            qn = int(q.group(1))
            month_day = "03-31" if qn == 1 else "06-30" if qn == 2 else "09-30" if qn == 3 else "12-31"
            return datetime.strptime(f"{y}-{month_day}", "%Y-%m-%d")
        return datetime.strptime(f"{y}-12-31", "%Y-%m-%d")
    except Exception:
        return None

def _company_from_doc_id(doc_id: str) -> str:
    try:
        meta = FINANCEBENCH_INDEX.get(doc_id) or {}
        company = meta.get("company")
        if company:
            return str(company)
    except Exception:
        pass
    try:
        name, _year = parse_stock_and_year(doc_id)
        return name
    except Exception:
        return ""

def _normalize_company(name: str) -> str:
    return str(name or "").replace("_", " ").replace("-", " ").strip().upper()

async def initialize_app():
    """Initialize the application by loading FinanceBench index"""
    global FINANCEBENCH_INDEX
    try:
        FINANCEBENCH_INDEX = load_financebench_index()
        logger.info(f"Loaded {len(FINANCEBENCH_INDEX)} documents from FinanceBench index")
    except Exception as e:
        logger.error(f"Failed to load FinanceBench index: {e}")
        FINANCEBENCH_INDEX = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler"""
    await initialize_app()
    yield

# Initialize FastAPI app
app = FastAPI(
    title="FinDocGPT API",
    description="AI-powered financial document analysis and investment strategy platform",
    version="1.0.0",
    contact={
        "name": "AkashX.ai Challenge",
        "url": "https://akashx.ai",
    },
    license_info={
        "name": "HackNation 2025",
    },
    lifespan=lifespan
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/", response_model=Dict[str, Any])
async def root():
    """API root endpoint"""
    return {
        'message': 'FinDocGPT API Server',
        'version': '1.0.0',
        'sponsor': 'AkashX.ai',
        'challenge': 'HackNation 2025',
        'features': ['Document Q&A', 'Financial Forecasting (Coming Soon)', 'Investment Strategy (Coming Soon)'],
        'docs_url': '/docs',
        'redoc_url': '/redoc'
    }

@app.get("/api/documents", response_model=APIResponse)
async def get_documents(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of documents to return"),
    offset: int = Query(0, ge=0, description="Number of documents to skip"),
    company: Optional[str] = Query(None, description="Filter by company name"),
    sector: Optional[str] = Query(None, description="Filter by sector"),
    doc_type: Optional[str] = Query(None, description="Filter by document type")
):
    """Get list of available documents with metadata and optional filtering"""
    try:
        documents = []
        
        for doc_name, metadata in FINANCEBENCH_INDEX.items():
            # Apply filters
            if company and company.lower() not in metadata.get('company', '').lower():
                continue
            if sector and sector.lower() != metadata.get('gics_sector', '').lower():
                continue
            if doc_type and doc_type.lower() != metadata.get('doc_type', '').lower():
                continue
            
            # Extract financial metrics if available (skip for now to prevent errors)
            financial_metrics = None
            # try:
            #     metrics = extract_financial_metrics(doc_name)
            #     if metrics:
            #         financial_metrics = FinancialMetrics(**metrics)
            # except Exception as e:
            #     logger.warning(f"Could not extract financial metrics for {doc_name}: {e}")
            
            # Parse additional metadata
            company_name, year = parse_stock_and_year(doc_name)
            doc_type_parsed = detect_doc_type(doc_name)
            
            document = Document(
                id=doc_name,
                name=doc_name,
                company=metadata.get('company', company_name),
                sector=str(metadata.get('gics_sector', 'Unknown') or 'Unknown'),
                doc_type=str(metadata.get('doc_type', doc_type_parsed) or doc_type_parsed),
                period=str(metadata.get('doc_period', year) or year),
                year=str(year),
                doc_link=metadata.get('doc_link'),
                financial_metrics=financial_metrics
            )
            documents.append(document)
        
        # Apply pagination
        total = len(documents)
        paginated_documents = documents[offset:offset + limit]
        
        return APIResponse(
            success=True,
            data={
                'documents': [doc.dict() for doc in paginated_documents],
                'total': total,
                'limit': limit,
                'offset': offset
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch documents")

@app.get("/api/documents/{document_id}", response_model=APIResponse)
async def get_document(document_id: str):
    """Get detailed information about a specific document"""
    try:
        resolved = _resolve_doc_id(document_id)
        if not resolved:
            raise HTTPException(status_code=404, detail="Document not found")
        metadata = FINANCEBENCH_INDEX[resolved]
        
        # Extract financial metrics (skip for now to prevent errors)
        financial_metrics = None
        # try:
        #     metrics = extract_financial_metrics(document_id)
        #     if metrics:
        #         financial_metrics = FinancialMetrics(**metrics)
        # except Exception as e:
        #     logger.warning(f"Could not extract financial metrics for {document_id}: {e}")
        
        # Parse additional metadata
        company_name, year = parse_stock_and_year(resolved)
        doc_type_parsed = detect_doc_type(resolved)
        
        document = Document(
            id=resolved,
            name=resolved,
            company=metadata.get('company', company_name),
            sector=metadata.get('gics_sector', 'Unknown'),
            doc_type=metadata.get('doc_type', doc_type_parsed),
            period=str(metadata.get('doc_period', year)),
            year=str(year),
            doc_link=metadata.get('doc_link'),
            financial_metrics=financial_metrics
        )
        
        return APIResponse(success=True, data={'document': document.dict()})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch document")

@app.get("/api/documents/{document_id}/text", response_model=APIResponse)
async def get_document_text(document_id: str, max_length: int = Query(50000, ge=1000, le=200000)):
    """Get the full text content of a document"""
    try:
        resolved = _resolve_doc_id(document_id)
        if not resolved:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get document text
        text = get_text(resolved)
        full_length = len(text)
        truncated = False
        
        # Truncate if too long for display
        if len(text) > max_length:
            text = text[:max_length] + "\n\n... (truncated for display)"
            truncated = True
        
        document_text = DocumentText(
            text=text,
            full_length=full_length,
            truncated=truncated
        )
        
        return APIResponse(success=True, data=document_text.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting text for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch document text")

def _summary_to_scores(summary: SentimentSummary) -> Dict[str, float]:
    """Convert SentimentSummary to frontend SentimentScore shape.

    Maps TextBlob polarity (-1..1) into a distribution for negative/neutral/positive,
    and uses polarity as compound.
    """
    polarity = summary.textblob_polarity
    abs_p = abs(polarity)
    neutral = max(0.0, 1.0 - abs_p)
    positive = max(0.0, polarity)
    negative = max(0.0, -polarity)
    # Normalize to sum to 1 (avoid tiny drift)
    total = positive + negative + neutral
    if total > 0:
        positive, negative, neutral = positive/total, negative/total, neutral/total
    return {
        "negative": round(negative, 4),
        "neutral": round(neutral, 4),
        "positive": round(positive, 4),
        "compound": round(polarity, 4),
    }

def _resolve_doc_id(document_id: str) -> Optional[str]:
    """Resolve a document id case-insensitively against FINANCEBENCH_INDEX keys."""
    if document_id in FINANCEBENCH_INDEX:
        return document_id
    target = document_id.lower()
    for key in FINANCEBENCH_INDEX.keys():
        if key.lower() == target:
            return key
    return None

def _extract_simple_sections(text: str) -> Dict[str, str]:
    """Heuristically extract a few common 10-K/10-Q sections for sentiment.

    Lightweight and fast; not bulletproof but useful for a quick pass.
    """
    lowered = text.lower()
    sections: Dict[str, str] = {}
    # naive markers
    markers = [
        ("Risk Factors", ["risk factors"]),
        ("MD&A", ["management's discussion", "management’s discussion", "discussion and analysis"]),
        ("Financial Statements", ["financial statements"]) ,
    ]
    for title, keys in markers:
        for k in keys:
            idx = lowered.find(k)
            if idx != -1:
                # take a window around the marker
                start = max(0, idx - 2000)
                end = min(len(text), idx + 10000)
                sections[title] = text[start:end]
                break
    return sections

@app.get("/api/documents/{document_id}/sentiment", response_model=APIResponse)
async def get_document_sentiment(document_id: str, max_length: int = Query(80000, ge=1000, le=300000)):
    """Compute sentiment for a document (overall + a few common sections)."""
    try:
        if document_id not in FINANCEBENCH_INDEX:
            raise HTTPException(status_code=404, detail="Document not found")

        raw_text = get_text(document_id)
        text = raw_text[:max_length]

        # Overall
        overall_summary = text2sentiment([text])[0]
        overall_scores = _summary_to_scores(overall_summary)

        # Sections (heuristic)
        sections_raw = _extract_simple_sections(text)
        sections_scores: Dict[str, Dict[str, float]] = {}
        if sections_raw:
            section_texts = list(sections_raw.values())
            summaries = text2sentiment(section_texts)
            for (title, _), summ in zip(sections_raw.items(), summaries):
                sections_scores[title] = _summary_to_scores(summ)

        method_desc = "TextBlob polarity with optional HF pipeline fallback"

        return APIResponse(
            success=True,
            data={
                "sentiment": {
                    "method": method_desc,
                    "overall": overall_scores,
                    "sections": sections_scores,
                }
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing sentiment for {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute sentiment")

@app.get("/api/companies/{company_name}/news-sentiment", response_model=APIResponse)
async def get_company_news_sentiment(
    company_name: str, 
    days_back: int = Query(30, ge=1, le=90, description="Number of days to look back for news")
):
    """Get news-based sentiment analysis for a company."""
    try:
        # Analyze company sentiment from news
        result = analyze_company_news_sentiment(
            company_name=company_name,
            api_key=NEWS_API_KEY,
            days_back=days_back
        )
        
        # Convert to API response format
        response_data = {
            "company": result.company,
            "total_articles": result.total_articles,
            "date_range": result.date_range,
            "overall_sentiment": result.overall_sentiment,
            "sentiment_trend": result.sentiment_trend,
            "summary": result.summary,
            "articles": [
                {
                    "title": article.title,
                    "description": article.description,
                    "url": article.url,
                    "published_at": article.published_at.isoformat(),
                    "source": article.source,
                    "sentiment_score": article.sentiment_score,
                    "sentiment_label": article.sentiment_label
                }
                for article in result.articles[:10]  # Limit to 10 most recent
            ]
        }
        
        return APIResponse(
            success=True,
            data={"news_sentiment": response_data}
        )
        
    except Exception as e:
        logger.error(f"Error getting news sentiment for {company_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch news sentiment")

@app.get("/api/documents/{document_id}/enhanced-sentiment", response_model=APIResponse)
async def get_enhanced_document_sentiment(
    document_id: str, 
    max_length: int = Query(80000, ge=1000, le=300000),
    include_news: bool = Query(True, description="Include news-based sentiment analysis")
):
    """Get enhanced sentiment analysis combining document and news sentiment."""
    try:
        if document_id not in FINANCEBENCH_INDEX:
            raise HTTPException(status_code=404, detail="Document not found")

        # Get document metadata
        metadata = FINANCEBENCH_INDEX[document_id]
        company_name = metadata.get('company', 'Unknown')
        
        # Document sentiment analysis
        raw_text = get_text(document_id)
        text = raw_text[:max_length]
        overall_summary = text2sentiment([text])[0]
        overall_scores = _summary_to_scores(overall_summary)

        sections_raw = _extract_simple_sections(text)
        sections_scores: Dict[str, Dict[str, float]] = {}
        if sections_raw:
            section_texts = list(sections_raw.values())
            summaries = text2sentiment(section_texts)
            for (title, _), summ in zip(sections_raw.items(), summaries):
                sections_scores[title] = _summary_to_scores(summ)

        response_data = {
            "document_sentiment": {
                "method": "TextBlob polarity with optional HF pipeline fallback",
                "overall": overall_scores,
                "sections": sections_scores,
            }
        }
        
        # News sentiment analysis (if requested and company name available)
        if include_news and company_name != 'Unknown':
            try:
                news_result = analyze_company_news_sentiment(
                    company_name=company_name,
                    api_key=NEWS_API_KEY,
                    days_back=14  # Last 2 weeks for faster response
                )
                
                response_data["news_sentiment"] = {
                    "total_articles": news_result.total_articles,
                    "overall_sentiment": news_result.overall_sentiment,
                    "summary": news_result.summary,
                    "recent_articles": [
                        {
                            "title": article.title,
                            "sentiment_label": article.sentiment_label,
                            "sentiment_score": article.sentiment_score,
                            "published_at": article.published_at.isoformat()
                        }
                        for article in news_result.articles[:5]
                    ]
                }
            except Exception as news_error:
                logger.warning(f"Could not fetch news sentiment for {company_name}: {news_error}")
                response_data["news_sentiment"] = {
                    "error": "Could not fetch recent news sentiment"
                }

        return APIResponse(
            success=True,
            data=response_data
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing enhanced sentiment for {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute enhanced sentiment")

@app.get("/api/documents/{document_id}/pdf")
async def get_document_pdf(document_id: str):
    """Serve the PDF file for a document (case-insensitive filename match)."""
    try:
        if document_id not in FINANCEBENCH_INDEX:
            # We still try to serve the PDF if present even if index key mismatch
            logger.warning(f"Document id {document_id} not in index; attempting filesystem match")

        pdfs_dir = Path("datasets/financebench/pdfs")
        if not pdfs_dir.exists():
            raise HTTPException(status_code=500, detail="PDF directory missing")

        # Try direct exact match first
        direct = pdfs_dir / f"{document_id}.pdf"
        pdf_path: Optional[Path] = None
        if direct.exists():
            pdf_path = direct
        else:
            # Case-insensitive stem match
            target = document_id.lower()
            for p in pdfs_dir.iterdir():
                if p.suffix.lower() == ".pdf" and p.stem.lower() == target:
                    pdf_path = p
                    break

        if not pdf_path or not pdf_path.exists():
            raise HTTPException(status_code=404, detail="PDF file not found")

        # Force inline display in browser instead of download
        return FileResponse(
            path=str(pdf_path),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="{pdf_path.name}"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving PDF for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve PDF file")

@app.post("/api/qa", response_model=APIResponse)
async def ask_question(request: QARequest):
    """Ask a question about a document using TF-IDF retrieval baseline."""
    try:
        start_time = time.time()
        if request.document_id not in FINANCEBENCH_INDEX:
            raise HTTPException(status_code=404, detail="Document not found")

        retrieval = answer_question(request.document_id, request.question, top_k=4)
        processing_time = time.time() - start_time

        response = QAResponse(
            answer=retrieval["answer"],
            confidence=float(retrieval["confidence"]),
            sources=retrieval["sources"],
            processing_time=round(processing_time, 2),
        )
        return APIResponse(success=True, data=response.dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing Q&A request: {e}")
        raise HTTPException(status_code=500, detail="Failed to process question")

# Stage 2 & 3 placeholder endpoints
@app.post("/api/forecast", response_model=APIResponse)
async def create_forecast(ticker: str = Query(..., min_length=1), period: str = Query("5y"), horizon: int = Query(5, ge=1, le=30), model: str = Query("lstm", description="Model to use: rf|prophet|lstm")):
    """Run a lightweight price forecasting pipeline using selectable model.

    rf: RandomForestRegressor (default)
    prophet: Facebook Prophet (if installed)
    lstm: Keras LSTM (if installed)
    """
    try:
        result = run_forecast(ticker=ticker.upper(), period=period, horizon=horizon, model=model)
        data = {
            "ticker": result.ticker,
            "model": result.model,
            "mae": result.mae,
            "horizon_days": result.horizon_days,
            "last_price": result.last_price,
            "predictions": result.predictions.assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
            "history": result.history.assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
        }
        return APIResponse(success=True, data=data)
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate forecast")

@app.post("/api/investment-recommendation", response_model=APIResponse)
async def get_investment_recommendation(req: InvestmentRequest):
    """Compute an investment recommendation by combining:
    - Document sentiment (Stage 1, if document provided)
    - Optional recent news sentiment (from document company or ticker)
    - Optional price forecast (Stage 2)
    Returns BUY/SELL/HOLD with confidence, reasoning, and optional target price.
    """
    try:
        doc_score: Optional[float] = None
        pos = 0.0
        neg = 0.0
        news_part = None
        news_score = None
        forecast_part = None
        forecast_score = None
        # Collect per-document sentiment for charting
        doc_breakdown: List[Dict[str, Any]] = []

        # 1) Document sentiment (optional, supports multiple docs)
        company = ""
        used_documents: List[str] = []
        if req.document_ids:
            # Normalize requested ids to actual FinanceBench entries by company name matching when user passes a company string
            normalized_ids: List[str] = []
            for did in req.document_ids:
                if did in FINANCEBENCH_INDEX:
                    normalized_ids.append(did)
            # If none were valid doc IDs and the client meant a company, try match by company substring (case-insensitive)
            if not normalized_ids and len(req.document_ids) == 1 and not req.document_id:
                q = str(req.document_ids[0]).strip().lower()
                # Treat as company needle
                for key, meta in FINANCEBENCH_INDEX.items():
                    comp = str(meta.get('company', '')).strip().lower()
                    if not comp:
                        continue
                    if q in comp or comp in q:
                        normalized_ids.append(key)
                # De-duplicate and optionally cap to avoid overload
                normalized_ids = list(dict.fromkeys(normalized_ids))[:50]
            doc_ids_to_use = normalized_ids if normalized_ids else req.document_ids

            # Prefer most recent documents to improve reliability and latency
            try:
                pairs = []
                for did in doc_ids_to_use:
                    dt = _doc_date_for_id(did)
                    pairs.append((did, dt or datetime(1970, 1, 1)))
                pairs.sort(key=lambda x: x[1], reverse=True)
                # Cap to top 12 most recent docs to avoid timeouts on large companies
                doc_ids_to_use = [p[0] for p in pairs[:12]]
            except Exception:
                # Fallback: keep original order but cap to 12
                doc_ids_to_use = (doc_ids_to_use or [])[:12]

            # Aggregate document sentiment with recency weighting (newest docs weigh more)
            weighted_sum = 0.0
            total_w = 0.0
            for did in doc_ids_to_use:
                try:
                    # Load text robustly
                    text = _safe_get_text(did)
                    if not text:
                        logger.warning(f"No text loaded for {did}; skipping")
                        continue
                    doc_summary = text2sentiment([text])[0]
                    # Use TextBlob polarity directly (-1..1)
                    pol = float(getattr(doc_summary, 'textblob_polarity', 0.0))
                    s = pol

                    # Recency weight: w = 1 / (1 + age_years)
                    dt = _doc_date_for_id(did)
                    if dt is not None:
                        age_years = max(0.0, (datetime.utcnow() - dt).days / 365.25)
                        w = 1.0 / (1.0 + age_years)
                    else:
                        w = 1.0

                    weighted_sum += float(s) * w
                    total_w += w

                    # Track used document canonical id
                    resolved_id = _resolve_doc_id(did) or did
                    used_documents.append(resolved_id)
                    if not company:
                        company = _company_from_doc_id(resolved_id)

                    # Add to breakdown for charting
                    doc_breakdown.append({
                        "id": resolved_id,
                        "score": s,
                        "weight": w,
                        "date": dt.isoformat() if dt else None,
                        "year": dt.year if dt else None,
                    })
                except Exception as e:
                    logger.warning(f"Doc sentiment failed for {did}: {e}")
                    continue
            if total_w > 0:
                doc_score = float(weighted_sum / total_w)
        elif req.document_id:
            try:
                text = _safe_get_text(req.document_id)
                if text:
                    doc_summary = text2sentiment([text])[0]
                    pol = float(getattr(doc_summary, 'textblob_polarity', 0.0))
                    doc_score = float(pol)
                    resolved_single = _resolve_doc_id(req.document_id) or req.document_id
                    used_documents.append(resolved_single)
                    dt = _doc_date_for_id(resolved_single)
                    doc_breakdown.append({
                        "id": resolved_single,
                        "score": float(pol),
                        "weight": 1.0,
                        "date": dt.isoformat() if dt else None,
                        "year": dt.year if dt else None,
                    })
                company = _company_from_doc_id(req.document_id)
            except Exception as e:
                logger.warning(f"Document sentiment failed for {req.document_id}: {e}")
                doc_score = None

        # 2) Determine ticker/company
        ticker = (req.ticker or "").strip().upper()
        if not ticker and company:
            # Resolve ticker from company name via Yahoo Finance search (no hard-coded mapping)
            try:
                res = _resolve_ticker_for_company(company)
                if res and res.get("symbol"):
                    ticker = str(res["symbol"]).upper()
            except Exception:
                pass
        if not company and ticker:
            # Resolve company long name from ticker
            try:
                res = _resolve_ticker_for_company(ticker)
                if res and res.get("name"):
                    company = str(res["name"]).strip()
                else:
                    company = ticker
            except Exception:
                company = ticker

        # Fallback: if no explicit document IDs but we have a company (e.g., user typed "apple"),
        # auto-select that company's FinanceBench documents and compute doc sentiment.
        if not used_documents and not req.document_ids and not req.document_id and company:
            try:
                comp_norm = _normalize_company(company)
                candidate_ids = []
                for key, meta in FINANCEBENCH_INDEX.items():
                    comp = _normalize_company(str(meta.get("company", "")))
                    if comp == comp_norm or comp_norm in comp or comp in comp_norm:
                        candidate_ids.append(key)
                if candidate_ids:
                    pairs = [(did, _doc_date_for_id(did) or datetime(1970, 1, 1)) for did in candidate_ids]
                    pairs.sort(key=lambda x: x[1], reverse=True)
                    doc_ids_to_use = [p[0] for p in pairs[:12]]

                    weighted_sum = 0.0
                    total_w = 0.0
                    for did in doc_ids_to_use:
                        try:
                            text = _safe_get_text(did)
                            if not text:
                                continue
                            doc_summary = text2sentiment([text])[0]
                            pol = float(getattr(doc_summary, 'textblob_polarity', 0.0))
                            s = pol

                            dt = _doc_date_for_id(did)
                            if dt is not None:
                                age_years = max(0.0, (datetime.utcnow() - dt).days / 365.25)
                                w = 1.0 / (1.0 + age_years)
                            else:
                                w = 1.0

                            weighted_sum += float(s) * w
                            total_w += w

                            resolved_id = _resolve_doc_id(did) or did
                            used_documents.append(resolved_id)
                            doc_breakdown.append({
                                "id": resolved_id,
                                "score": s,
                                "weight": w,
                                "date": dt.isoformat() if dt else None,
                                "year": dt.year if dt else None,
                            })
                        except Exception:
                            continue
                    if total_w > 0 and doc_score is None:
                        doc_score = float(weighted_sum / total_w)
            except Exception:
                pass

        # 3) News sentiment (optional)
        if req.include_news and company:
            try:
                news: NewsSentimentResult = analyze_company_news_sentiment(company, api_key=NEWS_API_KEY)
                overall = news.overall_sentiment or {"positive": 0, "neutral": 1, "negative": 0}
                news_score = float(overall.get("positive", 0) - overall.get("negative", 0))
                news_part = {
                    "company": company,
                    "total_articles": news.total_articles,
                    "overall": overall,
                    "summary": news.summary,
                }
            except Exception as e:
                logger.warning(f"News sentiment failed: {e}")
                news_score = None

        # 4) Forecast (optional if ticker available)
        if ticker:
            try:
                fc = run_forecast(ticker=ticker, period=req.period, horizon=req.horizon, model=req.model)
                last_price = float(fc.last_price)
                last_pred = float(fc.predictions.iloc[-1]["pred"]) if not fc.predictions.empty else last_price
                change_pct = 0.0 if last_price == 0 else (last_pred - last_price) / last_price
                change_pct = max(-0.15, min(0.15, change_pct))
                forecast_score = float(change_pct)
                forecast_part = {
                    "ticker": fc.ticker,
                    "model": fc.model,
                    "mae": fc.mae,
                    "last_price": last_price,
                    "target_price": last_pred,
                    "horizon_days": fc.horizon_days,
                }
            except Exception as e:
                logger.warning(f"Forecast failed for {ticker}: {e}")
                forecast_score = None

        # 5) Combine scores with available weights
        weights = {"doc": 0.6, "news": 0.25, "forecast": 0.15}
        total_w = 0.0
        composite = 0.0
        if doc_score is not None:
            composite += weights["doc"] * doc_score
            total_w += weights["doc"]
        if news_score is not None:
            composite += weights["news"] * news_score
            total_w += weights["news"]
        if forecast_score is not None:
            composite += weights["forecast"] * forecast_score
            total_w += weights["forecast"]
        composite = composite / total_w if total_w > 0 else 0.0

        # 6) Decision thresholds
        action = "HOLD"
        if composite >= 0.15:
            action = "BUY"
        elif composite <= -0.15:
            action = "SELL"

        # Confidence and risk level (cleaned; do not override action from news presence)
        confidence = float(min(1.0, max(0.0, abs(composite) * 1.5)))

        risk_level = "MEDIUM"
        mae_ratio = None
        if forecast_part and forecast_part.get("mae") and forecast_part.get("last_price"):
            mae_ratio = float(forecast_part["mae"]) / max(1e-9, float(forecast_part["last_price"]))
            if mae_ratio > 0.06:
                risk_level = "HIGH"
            elif mae_ratio < 0.03:
                risk_level = "LOW"
        if news_part and news_part.get("overall"):
            overall = news_part["overall"]
            if overall.get("neutral", 0) < 0.3:
                risk_level = "HIGH" if risk_level != "HIGH" else risk_level

        reasoning_bits = []
        if doc_score is not None:
            reasoning_bits.append(f"Aggregated document sentiment score {doc_score:+.2f}.")
        if news_score is not None and news_part:
            overall = news_part["overall"]
            reasoning_bits.append(
                f"News tone P/N/N = {overall.get('positive',0):.2f}/{overall.get('negative',0):.2f}/{overall.get('neutral',0):.2f}."
            )
        if forecast_part is not None:
            reasoning_bits.append(
                f"Forecast suggests {forecast_score:+.2%} over {forecast_part['horizon_days']}d (model {forecast_part['model']}, MAE {forecast_part['mae']:.2f})."
            )
        # If no documents were used, add explicit note about data sources used
        if not used_documents:
            sources = []
            if news_score is not None:
                sources.append("news sentiment")
            if forecast_part is not None:
                sources.append("price forecast")
            src_text = " and ".join(sources) if sources else "available data"
            subject = company or ticker or "this ticker"
            reasoning_bits.append(f"No FinanceBench documents found for {subject}. Recommendation based on {src_text}.")
        if not reasoning_bits:
            reasoning_bits.append("Insufficient data; defaulting to HOLD.")

        components: Dict[str, Any] = {
            "doc_score": doc_score,
            "news_score": news_score,
            "forecast_score": forecast_score,
            "forecast": forecast_part,
            "used_documents": used_documents,
        }
        if doc_breakdown:
            # Sort breakdown by date/year descending for convenience
            try:
                components["doc_breakdown"] = sorted(doc_breakdown, key=lambda d: (d.get("date") or ""), reverse=False)
            except Exception:
                components["doc_breakdown"] = doc_breakdown

        return APIResponse(success=True, data=InvestmentRecommendation(
            action=action,
            confidence=confidence,
            reasoning="\n".join(reasoning_bits),
            target_price=forecast_part.get("target_price") if forecast_part else None,
            risk_level=risk_level,
            components=components
        ).dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Investment recommendation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute investment recommendation")

# --- Ticker/Company lookup utilities ---

def _yahoo_search(query: str) -> Dict[str, Any]:
    """Query Yahoo Finance search endpoint for symbols matching query."""
    import json
    import urllib.parse
    import urllib.request

    url = "https://query2.finance.yahoo.com/v1/finance/search?" + urllib.parse.urlencode({
        "q": query,
        "quotesCount": 10,
        "newsCount": 0,
    })
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data or {}


def _resolve_ticker_for_company(name_or_symbol: str) -> Optional[Dict[str, Any]]:
    """Resolve a company name or symbol to a best-match ticker using Yahoo search.
    Returns dict with symbol, name, exchange if found.
    """
    q = (name_or_symbol or "").strip()
    if not q:
        return None
    try:
        # Try yfinance search if available (newer versions)
        try:
            import yfinance as yf  # type: ignore
            if hasattr(yf, "search"):
                res = yf.search(q)
                quotes = res.get("quotes", []) if isinstance(res, dict) else []
            else:
                quotes = []
        except Exception:
            quotes = []
        if not quotes:
            res = _yahoo_search(q)
            quotes = res.get("quotes", []) if isinstance(res, dict) else []
        # Pick first equity-like result
        for it in quotes:
            sym = it.get("symbol")
            longname = it.get("longname") or it.get("longName") or it.get("shortname") or it.get("shortName")
            exch = it.get("exchDisp") or it.get("exchangeDisp") or it.get("exchange")
            quote_type = (it.get("quoteType") or "").lower()
            if sym and (quote_type in ("equity", "etf", "mutualfund", "index") or True):
                return {"symbol": str(sym).upper(), "name": longname or sym, "exchange": exch}
    except Exception:
        return None
    return None

@app.get("/api/lookup", response_model=APIResponse)
async def lookup_symbol(query: str):
    """Lookup ticker symbol(s) by company name or symbol."""
    try:
        result = _resolve_ticker_for_company(query)
        # Also return top candidates for potential UI use
        candidates = []
        try:
            data = _yahoo_search(query)
            for it in data.get("quotes", [])[:10]:
                candidates.append({
                    "symbol": it.get("symbol"),
                    "name": it.get("longname") or it.get("longName") or it.get("shortname") or it.get("shortName"),
                    "exchange": it.get("exchDisp") or it.get("exchangeDisp") or it.get("exchange"),
                })
        except Exception:
            pass
        return APIResponse(success=True, data={"query": query, "best": result, "candidates": candidates})
    except Exception:
        return APIResponse(success=False, error="Lookup failed")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "documents_loaded": len(FINANCEBENCH_INDEX)}

def _candidate_pdf_paths(name: str) -> list:
    """Generate candidate PDF paths for a given FinanceBench doc_name, trying common locations/cases."""
    from pathlib import Path as _P
    stem = _P(name).stem
    # Try as-provided, lowercase, uppercase
    names = [f"{stem}.pdf", f"{stem.lower()}.pdf", f"{stem.upper()}.pdf"]
    roots = [PATH_PDFS_CWD, PATH_PDFS_GENERIC, PATH_PDFS_FB]
    out = []
    for root in roots:
        for nm in names:
            out.append(str((root / nm)))
    return out


def _safe_get_text(doc_id: str) -> str:
    """Best-effort loader for a FinanceBench document by id, trying canonical, original doc_name,
    and multiple file system case variants across known roots. Returns empty string on failure.
    """
    try:
        resolved = _resolve_doc_id(doc_id) or doc_id
        meta = FINANCEBENCH_INDEX.get(resolved) or {}
        candidates = []
        original = meta.get("doc_name")
        if original:
            candidates.append(str(original))
            candidates.extend(_candidate_pdf_paths(original))
        candidates.append(str(resolved))
        candidates.extend(_candidate_pdf_paths(resolved))
        seen = set()
        for name in candidates:
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            try:
                txt = get_text(name)
                if txt:
                    return txt
            except Exception:
                continue
    except Exception:
        pass
    return ""