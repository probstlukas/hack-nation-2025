import axios from 'axios';
import { 
  Document, 
  DocumentText, 
  SentimentAnalysis, 
  QARequest, 
  QAResponse, 
  ApiResponse,
  ForecastRequest,
  InvestmentRecommendation
} from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 seconds for Q&A operations
});

// Response interceptor to handle API responses
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

export const apiService = {
  // Document operations
  async getDocuments(): Promise<Document[]> {
    const response = await api.get<ApiResponse<{ documents: Document[]; total: number }>>(
      '/api/documents?limit=1000'
    );
    return response.data.data?.documents || [];
  },

  async getDocument(documentId: string): Promise<Document> {
    const response = await api.get<ApiResponse<{ document: Document }>>(
      `/api/documents/${documentId}`
    );
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.error || 'Failed to fetch document');
    }
    return response.data.data.document;
  },

  async getDocumentText(documentId: string): Promise<DocumentText> {
    const response = await api.get<ApiResponse<DocumentText>>(
      `/api/documents/${documentId}/text`
    );
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.error || 'Failed to fetch document text');
    }
    return response.data.data;
  },

  async getDocumentSentiment(documentId: string): Promise<SentimentAnalysis> {
    const response = await api.get<ApiResponse<{ sentiment: SentimentAnalysis }>>(
      `/api/documents/${documentId}/sentiment`
    );
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.error || 'Failed to fetch sentiment analysis');
    }
    return response.data.data.sentiment;
  },

  async getEnhancedDocumentSentiment(documentId: string, includeNews: boolean = true): Promise<any> {
    const response = await api.get<ApiResponse<any>>(
      `/api/documents/${documentId}/enhanced-sentiment?include_news=${includeNews}`
    );
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.error || 'Failed to fetch enhanced sentiment analysis');
    }
    return response.data.data;
  },

  // Q&A operations
  async askQuestion(request: QARequest): Promise<QAResponse> {
    const response = await api.post<ApiResponse<QAResponse>>('/api/qa', request);
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.error || 'Failed to get answer');
    }
    return response.data.data;
  },

  // Forecasting
  async createForecast(request: Partial<ForecastRequest> & { ticker?: string; period?: string; horizon?: number; model?: string }): Promise<any> {
    const ticker = (request as any).ticker || '';
    const period = (request as any).period || '5y';
    const horizon = (request as any).horizon ?? (request as any).forecast_periods ?? 5;
    const model = (request as any).model || 'lstm';
    if (!ticker) throw new Error('ticker is required');
    const url = `/api/forecast?ticker=${encodeURIComponent(ticker)}&period=${encodeURIComponent(period)}&horizon=${encodeURIComponent(String(horizon))}&model=${encodeURIComponent(model)}`;
    const response = await api.post<ApiResponse<any>>(url);
    return response.data;
  },

  async getCompanyNewsSentiment(companyName: string, daysBack: number = 30): Promise<any> {
    const url = `/api/companies/${encodeURIComponent(companyName)}/news-sentiment?days_back=${encodeURIComponent(String(daysBack))}`;
    const response = await api.get<ApiResponse<any>>(url);
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.error || 'Failed to fetch company news sentiment');
    }
    return response.data.data.news_sentiment;
  },

  // Stage 3: Investment Strategy
  async getInvestmentRecommendation(request: { document_id: string; ticker?: string; period?: string; horizon?: number; model?: string; include_news?: boolean; }): Promise<InvestmentRecommendation> {
    const payload = {
      document_id: request.document_id,
      ticker: request.ticker,
      period: request.period ?? '5y',
      horizon: request.horizon ?? 5,
      model: request.model ?? 'lstm',
      include_news: request.include_news ?? true,
    };
    const response = await api.post<ApiResponse<InvestmentRecommendation>>(
      '/api/investment-recommendation',
      payload
    );
    if (!response.data.success || !response.data.data) {
      throw new Error(response.data.error || 'Failed to get investment recommendation');
    }
    return response.data.data as unknown as InvestmentRecommendation;
  },

  // Health check
  async healthCheck(): Promise<any> {
    const response = await api.get('/');
    return response.data;
  },
};

export default apiService;
