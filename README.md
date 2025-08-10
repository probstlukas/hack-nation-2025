# FinDocGPT - AI for Financial Document Analysis & Investment Strategy

[![AkashX.ai](https://img.shields.io/badge/Sponsored%20by-AkashX.ai-blue)](https://akashx.ai)
[![HackNation 2025](https://img.shields.io/badge/HackNation-2025-green)](https://hacknation.com)

An AI-powered platform for financial document analysis and investment strategy development, built for the AkashX.ai challenge at HackNation 2025.

## 🎯 Challenge Overview

This project implements a comprehensive 3-stage AI system for financial analysis, directly mapped to the AkashX.ai FinDocGPT challenge:

### Stage 1: Insights & Analysis (Document Q&A) ✅
- Process financial documents from *FinanceBench* dataset
- Natural language Q&A interface with RAG for document analysis
- Sentiment analysis for overall and key sections

### Stage 2: Financial Forecasting ✅
- Predict near‑term price movements using historical data from *yfinance*
- Interactive chart with time window selection, error metric, and forecast vs. history

### Stage 3: Investment Strategy & Decision-Making ✅
- Generate actionable Buy/Sell/Hold with confidence and risk
- Combines document sentiment, recent news sentiment, and forecast signal
- Returns transparency details: used documents, per-doc breakdown, component scores

## 🏗️ Architecture

```
             ┌───────────────────────────┐
             │       React Frontend      │
             │         (TypeScript)      │
             └───────────────────────────┘
                           ▲
                           │  REST/JSON
                           │
             ┌───────────────────────────┐
             │       FastAPI Backend     │
             │           (Python)        │
             └───────────────────────────┘
              ▲                          ▲
              │ Ingest PDFs              │ Market Data
              │                          │
┌───────────────────────────┐   ┌───────────────────────────┐
│       FinanceBench        │   │          yfinance         │
│       Dataset (PDFs)      │   │      (Yahoo Finance)      │
└───────────────────────────┘   └───────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+

### 1) Start Backend

1. **Set up environment variables**  
   - Copy the example file and edit it with your own values:
     ```bash
     cp .env.example .env
     ```
   - Open `.env` and set your keys:
     - `NEWS_API_KEY` → Enables news sentiment analysis. Get from [newsapi.org](https://newsapi.org/).  
     - `OPENAI_API_KEY` →  Get from [platform.openai.com](https://platform.openai.com/).  
     - `PORT` → *(optional)* Server port (default: `5001`).  

2. Create a conda environment (optional)
    ```
    conda create -n hack-nation python=3.10 -y
    ```

3. **Run the start script**
   This automatically activates the conda env, installs all dependencies and starts frontend/backend server.
   ```bash
   ./start.sh

Backend runs at http://localhost:5001 (FastAPI docs at `/docs`).
App runs at http://localhost:3000.

## 📚 Key Routes
- `/` — Dashboard
- `/document/:id` — Document Analysis (Q&A, PDF, Sentiment)
- `/forecasting` — Forecasting (RF/Prophet/LSTM)
- `/investment-strategy` — Investment Strategy (Stage 3)

## 🧠 APIs (selected)
- `GET /api/documents` — List available documents (FinanceBench)
- `GET /api/documents/{id}` — Document metadata
- `GET /api/documents/{id}/text` — Extracted document text
- `GET /api/documents/{id}/sentiment` — TextBlob sentiment for document
- `GET /api/documents/{id}/pdf` — Serve the PDF inline
- `GET /api/companies/{company}/news-sentiment` — Aggregated recent news sentiment (requires NEWS_API_KEY)
- `POST /api/forecast` — Body: `{ ticker, period, horizon, model }` → forecasts + MAE
- `POST /api/investment-recommendation` — Body: `{ company|ticker, document_ids? }` → action/confidence/risk + transparency
- `GET /api/lookup?q=` — Resolve ticker/company candidates (for UX)

## Notes
- News sentiment requires `NEWS_API_KEY`; otherwise that component is omitted gracefully.
- PDFs are served inline via `/api/documents/:id/pdf`.

## 📊 Dataset

The project uses the bundled **FinanceBench** dataset containing:

- **300+ Financial Documents**: 10-K, 10-Q, 8-K filings, earnings reports
- **Major Companies**: Apple, Microsoft, Amazon, Google, Tesla, and more
- **Q&A Examples**: Pre-made questions and expert answers
- **Multiple Years**: 2015-2023 data coverage
- **Diverse Sectors**: Technology, Healthcare, Finance, Consumer, etc.

### Sample Companies
- Technology: Apple, Microsoft, AMD, Intel, Adobe, Oracle
- Consumer: Amazon, Costco, Nike, McDonald's, Coca-Cola
- Healthcare: Johnson & Johnson, Pfizer, CVS Health
- Finance: JPMorgan, American Express, PayPal
- And many more...

## 🛠️ Features

### Stage 1 — Document Q&A
- **Document Library**: Browse 300+ financial documents
- **Smart Search**: Filter by company, sector, document type
- **Document Viewer**: Read full document text
- **Q&A Chat Interface**: Ask questions about any document
- **Financial Metrics Extraction**: Automatic revenue, profit analysis via Q&A
  - Note: For newly opened documents, the first query may take longer or fail as the system generates the vector database for retrieval. This is a one-time process per document. Wait about a minute, then try again.
- **Sentiment Analysis**: Market sentiment from financial text
- **Responsive Design**: Works on desktop and mobile

### Stage 2 — Forecasting
- Historical price ingestion and model selection (LSTM/RF/Prophet)
- MAE surfaced in UI; interactive brush/zoom; dashed forecast line; directional coloring

### Stage 3 — Investment Strategy
- Synthesizes document sentiment, news sentiment, and forecast
- Returns action (BUY/SELL/HOLD), confidence, risk level, target price (when applicable)
- Transparency: `used_documents` and `doc_breakdown` to show which docs influenced the decision

## 🔧 Technical Stack

### Frontend
- **React 18** with TypeScript
- **React Router** for navigation
- **Axios** for API communication
- **Lucide React** for icons
- **Custom CSS** with modern design system

### Backend
- **FastAPI** web framework
- **CORS** for cross-origin requests
- **PyMuPDF** for PDF text extraction
- **LangChain** for document processing
- **NLTK** for sentiment analysis

### Key Libraries
- `pymupdf`: PDF text extraction
- `langchain`: Document chunking and processing
- `nltk`: Natural language processing
- `pdfplumber`: Enhanced PDF table extraction

## 📁 Project Structure

```
hack-nation-2025/
├── backend/
│   ├── main.py               # FastAPI app entry
│   └── util/                 # PDF, retrieval, sentiment, forecast utils
├── datasets/
│   └── financebench/         # Document metadata and PDFs
├── frontend/                 # React + TypeScript app
│   ├── src/pages/            # Pages: Dashboard, DocumentAnalysis, Forecasting, InvestmentStrategy
│   ├── src/services/api.ts   # API client
│   └── public/
├── requirements.txt          # Python deps
├── start.sh                  # Dev helper to launch backend
└── README.md
```

## 🎨 User Interface

### Dashboard
- **Company Overview**: Browse financial documents by company
- **Sector Filtering**: Filter by industry sector
- **Document Types**: 10-K, 10-Q, 8-K, Earnings reports
- **Search Functionality**: Find specific companies or documents

### Document Analysis
- **Dual-Pane Interface**: Chat Q&A + Document viewer
- **Real-time Q&A**: Ask questions and get AI responses
- **Financial Metrics**: Automatic extraction of key numbers
- **Confidence Scores**: AI confidence in responses
- **Quick Questions**: Pre-built common questions

### Demo Guide
1) Document Analysis: Open `/document/:id` from Dashboard; ask questions; view sentiment and PDF side-by-side.
2) Forecasting: Open `/forecasting`; enter ticker (e.g., AAPL), choose model and horizon; run forecast; inspect MAE and overlay.
3) Investment Strategy: Open `/investment-strategy`; enter ticker/company (e.g., Apple or AAPL); review action, confidence, risk; see which documents were used and the sentiment-over-time chart.

## 🧪 Try These Prompts

Try these sample questions:

### Financial Metrics
- "What is Apple's revenue for 2022?"
- "How much debt does Microsoft have?"
- "What are Amazon's key risk factors?"

### Business Analysis
- "What is Tesla's main business?"
- "How does Google make money?"
- "What are Netflix's competitive advantages?"

### Performance Questions
- "How is McDonald's cash flow?"
- "What drove Apple's growth in 2022?"
- "What are Intel's biggest challenges?"

## 🔮 Future Enhancements

### Stage 1: Financial Document Analysis and Q&A
- **Improved Metric Extraction**: Use pattern matching and keyword search (reports are partially structured), prompt engineering or a combination of both
- **Improved RAG**: Employ different kinds of document indices for better content retrieval (e.g., LLM-generated summaries)

### Stage 2: Advanced Forecasting
- **Machine Learning Models**: Use more sophisticated models and include financial metrics from the reports as features
- **Risk Modeling**: VaR, Monte Carlo simulations
- **Uncertainty Estimates on Forecasting**: Use an ensemble of different prediction models and model accuracy from historical data

### Stage 3: Investment Strategy
- **Portfolio Theory**: Modern portfolio optimization
- **ESG Integration**: Environmental, Social, Governance factors
- **Backtesting**: Historical strategy performance
- **Real-time Alerts**: Market change notifications

## 📝 API Documentation
See “APIs (selected)” above and live docs at `http://localhost:5001/docs`.


## Authors
- Lukas Probst
- Christopher von Klitzing


## 📄 License
This project is developed for the AkashX.ai HackNation 2025 challenge.

---

**Sponsored by AkashX.ai** | **HackNation 2025 Challenge**

— Building the future of AI-powered financial analysis —
