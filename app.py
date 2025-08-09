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
        if document_id not in FINANCEBENCH_INDEX:
            raise HTTPException(status_code=404, detail="Document not found")
        
        metadata = FINANCEBENCH_INDEX[document_id]
        
        # Extract financial metrics (skip for now to prevent errors)
        financial_metrics = None
        # try:
        #     metrics = extract_financial_metrics(document_id)
        #     if metrics:
        #         financial_metrics = FinancialMetrics(**metrics)
        # except Exception as e:
        #     logger.warning(f"Could not extract financial metrics for {document_id}: {e}")
        
        # Parse additional metadata
        company_name, year = parse_stock_and_year(document_id)
        doc_type_parsed = detect_doc_type(document_id)
        
        document = Document(
            id=document_id,
            name=document_id,
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
        if document_id not in FINANCEBENCH_INDEX:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get document text
        text = get_text(document_id)
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

@app.get("/api/documents/{document_id}/pdf")
async def get_document_pdf(document_id: str):
    """Serve the PDF file for a document"""
    try:
        if document_id not in FINANCEBENCH_INDEX:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Construct PDF file path
        pdf_filename = f"{document_id}.pdf"
        pdf_path = Path("datasets/financebench/pdfs") / pdf_filename
        
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        return FileResponse(
            path=str(pdf_path),
            media_type="application/pdf",
            filename=pdf_filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving PDF for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve PDF file")

@app.post("/api/qa", response_model=APIResponse)
async def ask_question(request: QARequest):
    """Ask a question about a document (mock implementation for Stage 1)"""
    try:
        start_time = time.time()
        
        if request.document_id not in FINANCEBENCH_INDEX:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get document metadata for context
        metadata = FINANCEBENCH_INDEX[request.document_id]
        company = metadata.get('company', 'Unknown')
        doc_type = metadata.get('doc_type', 'Unknown')
        period = metadata.get('doc_period', 'Unknown')
        
        # Mock Q&A system - In production, this would use a real LLM
        question_lower = request.question.lower()
        
        if any(word in question_lower for word in ['revenue', 'sales', 'income']):
            try:
                metrics = extract_financial_metrics(request.document_id)
                if metrics and metrics.get('revenue'):
                    revenue = metrics['revenue']
                    currency = metrics.get('currency', 'USD')
                    units = metrics.get('units_multiplier', 1)
                    
                    if units >= 1e9:
                        revenue_str = f"{currency} {revenue/1e9:.1f} billion"
                    elif units >= 1e6:
                        revenue_str = f"{currency} {revenue/1e6:.1f} million"
                    else:
                        revenue_str = f"{currency} {revenue:,.0f}"
                    
                    answer = f"According to the {doc_type} filing, {company} reported revenue of {revenue_str} for the period {period}."
                else:
                    answer = f"I found the {company} {doc_type} filing from {period}, but could not extract specific revenue figures. The financial data may be in a format that requires manual review."
            except Exception as e:
                answer = f"I found the {company} {doc_type} filing from {period}. However, I encountered an issue extracting specific financial metrics: {str(e)}"
        
        elif any(word in question_lower for word in ['risk', 'challenge', 'threat']):
            answer = f"Based on the {company} {doc_type} filing from {period}, key risk factors typically include market competition, regulatory changes, economic conditions, and operational challenges. For specific risk details, please refer to the 'Risk Factors' section in the document."
        
        elif any(word in question_lower for word in ['business', 'what does', 'company do']):
            answer = f"{company} is a company in the {metadata.get('gics_sector', 'business')} sector. Based on their {doc_type} filing from {period}, they operate in multiple business segments. For detailed business descriptions, please refer to the 'Business Overview' section of the document."
        
        elif any(word in question_lower for word in ['cash', 'cash flow', 'liquidity']):
            answer = f"According to the {company} {doc_type} filing from {period}, cash flow information can be found in the Cash Flow Statement section. This includes operating cash flow, investing activities, and financing activities."
        
        elif any(word in question_lower for word in ['debt', 'liability', 'borrowing']):
            answer = f"Debt and liability information for {company} can be found in the Balance Sheet section of their {doc_type} filing from {period}. This includes both short-term and long-term debt obligations."
        
        else:
            # Generic response for other questions
            answer = f"I found information about {company} in their {doc_type} filing from {period}. For specific details about '{request.question}', I recommend reviewing the relevant sections of the document. This is a mock Q&A system - in production, I would provide more detailed analysis using advanced language models."
        
        processing_time = time.time() - start_time
        
        response = QAResponse(
            answer=answer,
            confidence=random.uniform(0.75, 0.95),
            sources=[f"Page sections from {company} {doc_type} {period}"],
            processing_time=round(processing_time, 2)
        )
        
        return APIResponse(success=True, data=response.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing Q&A request: {e}")
        raise HTTPException(status_code=500, detail="Failed to process question")

# Stage 2 & 3 placeholder endpoints
@app.post("/api/forecast", response_model=APIResponse)
async def create_forecast():
    """Placeholder for financial forecasting functionality"""
    return APIResponse(
        success=False, 
        message='Financial forecasting feature coming soon in Stage 2!',
        data={'stage': 2}
    )

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