"""
FinDocGPT - AI for Financial Document Analysis & Investment Strategy
FastAPI Backend for Stage 1: Document Q&A Feature

This is the API backend for the FinDocGPT application implementing
the AkashX.ai challenge requirements. Frontend is a separate React app.
"""
import os
import asyncio
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

# Import our PDF processing utilities
from util.pdf2text import (
    load_financebench_index, 
    get_doc_metadata,
    parse_stock_and_year,
    detect_doc_type,
    get_text,
    chunk_documents,
    extract_financial_metrics
)
from util.text2sentiment import text2sentiment, SentimentSummary
from util.retrieval import answer_question
from util.forecast import run_forecast
from util.news_sentiment import analyze_company_news_sentiment, NewsSentimentResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Global variables for caching
FINANCEBENCH_INDEX = {}
DOCUMENT_CACHE = {}

# NewsAPI configuration
NEWS_API_KEY = "6979aad5e683411b9316510e0dd423e7"

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
                sector=metadata.get('gics_sector', 'Unknown'),
                doc_type=metadata.get('doc_type', doc_type_parsed),
                period=str(metadata.get('doc_period', year)),
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
async def create_forecast(ticker: str = Query(..., min_length=1), period: str = Query("5y"), horizon: int = Query(5, ge=1, le=30)):
    """Run a lightweight price forecasting pipeline using yfinance + RandomForest.

    Inspired by the approach in a public NVDA prediction notebook [predict_nvda.ipynb].
    """
    try:
        result = run_forecast(ticker=ticker.upper(), period=period, horizon=horizon)
        data = {
            "ticker": result.ticker,
            "model": result.model,
            "mae": result.mae,
            "horizon_days": result.horizon_days,
            "last_price": result.last_price,
            "predictions": result.predictions.assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
        }
        return APIResponse(success=True, data=data)
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate forecast")

@app.post("/api/investment-recommendation", response_model=APIResponse)
async def get_investment_recommendation():
    """Placeholder for investment recommendation functionality"""
    return APIResponse(
        success=False, 
        message='Investment strategy feature coming soon in Stage 3!',
        data={'stage': 3}
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "documents_loaded": len(FINANCEBENCH_INDEX)}

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 5001))
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=port, 
        reload=True,
        log_level="info"
    )