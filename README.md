# FinDocGPT - AI for Financial Document Analysis & Investment Strategy

[![AkashX.ai](https://img.shields.io/badge/Sponsored%20by-AkashX.ai-blue)](https://akashx.ai)
[![HackNation 2025](https://img.shields.io/badge/HackNation-2025-green)](https://hacknation.com)

An AI-powered platform for financial document analysis and investment strategy development, built for the AkashX.ai challenge at HackNation 2025.

## 🎯 Challenge Overview

This project implements a comprehensive 3-stage AI system for financial analysis:

### Stage 1: Insights & Analysis (Document Q&A) ✅
- Status: IMPLEMENTED
- Process financial documents and extract key insights
- Natural language Q&A interface for document analysis
- Sentiment analysis for overall and key sections

### Stage 2: Financial Forecasting ✅
- Status: IMPLEMENTED (baseline + Prophet/LSTM optional)
- Predict near‑term price movements using historical data
- Model options: RandomForest (default), Prophet, LSTM
- Interactive chart with zoom and overlays

### Stage 3: Investment Strategy & Decision-Making ✅
- Status: IMPLEMENTED (initial)
- Generate actionable Buy/Sell/Hold with confidence and risk
- Combines document sentiment, optional news sentiment, and forecast
- UI live at /investment-strategy

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ React Frontend  │    │ FastAPI Backend  │    │  FinanceBench   │
│ (TypeScript)    │◄──►│  (Python)        │◄──►│  Dataset (PDFs) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+

### 1) Start Backend
- Create a `.env` in repo root (or export in shell):
  - NEWS_API_KEY=your_key
  - PORT=5001 (optional)
- Run server:

```bash
./start.sh
```

Backend runs at http://localhost:5001 (FastAPI docs at /docs).

### 2) Start Frontend
- Optional `frontend/.env`:
  - REACT_APP_API_URL=http://localhost:5001
- From frontend/:

```bash
npm install
npm start
```

App runs at http://localhost:3000.

## 📚 Key Routes
- /              Dashboard
- /document/:id  Document Analysis (Q&A, PDF, Sentiment)
- /forecasting   Forecasting (RF/Prophet/LSTM)
- /investment-strategy  Investment Strategy (Stage 3 UI)

## 🧠 APIs (selected)
- GET /api/documents
- GET /api/documents/{id}
- GET /api/documents/{id}/text
- GET /api/documents/{id}/sentiment
- GET /api/companies/{company}/news-sentiment
- POST /api/forecast?ticker=...&period=...&horizon=...&model=...
- POST /api/investment-recommendation

## Notes
- News sentiment requires NEWS_API_KEY set; otherwise that part is skipped.
- Prophet/LSTM paths are optional and may take longer on first use.
- PDFs are served inline via /api/documents/:id/pdf.

## 📊 Dataset

The project uses the **FinanceBench** dataset containing:

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

### Current Features (Stage 1)
- ✅ **Document Library**: Browse 300+ financial documents
- ✅ **Smart Search**: Filter by company, sector, document type
- ✅ **Document Viewer**: Read full document text
- ✅ **Q&A Chat Interface**: Ask questions about any document
- ✅ **Financial Metrics Extraction**: Automatic revenue, profit analysis
- ✅ **Sentiment Analysis**: Market sentiment from financial text
- ✅ **Responsive Design**: Works on desktop and mobile

### Planned Features

#### Stage 2: Financial Forecasting
- 🔄 Time series forecasting models
- 🔄 External data integration (Yahoo Finance, Alpha Vantage)
- 🔄 Risk assessment and volatility prediction
- 🔄 Interactive forecasting dashboard

#### Stage 3: Investment Strategy
- 🔄 Buy/Sell/Hold recommendations
- 🔄 Portfolio optimization
- 🔄 Risk-adjusted returns calculation
- 🔄 Backtesting capabilities

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
- **Pandas** for data analysis

### Key Libraries
- `pymupdf`: PDF text extraction
- `langchain`: Document chunking and processing
- `nltk`: Natural language processing
- `pandas`: Financial data analysis
- `pdfplumber`: Enhanced PDF table extraction

## 📁 Project Structure

```
hack-nation-2025/
├── app.py                 # FastAPI backend
├── requirements.txt       # Python dependencies
├── util/
│   └── pdf2text.py       # PDF processing utilities
├── datasets/
│   └── financebench/     # Financial documents and metadata
├── frontend/             # React TypeScript frontend
│   ├── src/
│   │   ├── components/   # Reusable UI components
│   │   ├── pages/        # Main application pages
│   │   ├── services/     # API service layer
│   │   ├── types/        # TypeScript type definitions
│   │   └── utils/        # Utility functions
│   └── public/
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

### Stage Placeholders
- **Professional Design**: Consistent with main application
- **Feature Previews**: Detailed upcoming functionality
- **Technical Roadmap**: Clear development timeline

## 🧪 Testing the Q&A System

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

### Stage 2: Advanced Forecasting
- **Machine Learning Models**: LSTM, ARIMA for time series
- **External APIs**: Real-time market data integration
- **Risk Modeling**: VaR, Monte Carlo simulations
- **Visualization**: Interactive charts and forecasts

### Stage 3: Investment Intelligence
- **Portfolio Theory**: Modern portfolio optimization
- **ESG Integration**: Environmental, Social, Governance factors
- **Backtesting**: Historical strategy performance
- **Real-time Alerts**: Market change notifications

## 📝 API Documentation

### Document Endpoints
- `GET /api/documents` - List all available documents
- `GET /api/documents/{id}` - Get document details
- `GET /api/documents/{id}/text` - Get document text content
- `GET /api/documents/{id}/sentiment` - Get sentiment analysis

### Q&A Endpoints
- `POST /api/qa` - Ask a question about a document

### Stage 2 & 3 Placeholders
- `POST /api/forecast` - Financial forecasting (placeholder)
- `POST /api/investment-recommendation` - Investment advice (placeholder)

## 🤝 Contributing

This project was built for the AkashX.ai challenge at HackNation 2025. The current implementation focuses on Stage 1 (Document Q&A) with a solid foundation for Stage 2 and 3 development.

## 📄 License

This project is developed for the AkashX.ai HackNation 2025 challenge.

---

**Sponsored by AkashX.ai** | **HackNation 2025 Challenge**

*Building the future of AI-powered financial analysis*