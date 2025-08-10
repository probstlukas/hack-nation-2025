export interface Document {
  id: string;
  name: string;
  company: string;
  sector: string;
  doc_type: string;
  period: number | string;
  year: number | string;
  doc_link?: string;
  financial_metrics?: FinancialMetrics;
}

export interface FinancialMetrics {
  revenue?: number;
  cost_of_revenue?: number;
  gross_profit?: number;
  operating_income?: number;
  net_income?: number;
  eps_basic?: number;
  eps_diluted?: number;
  currency?: string;
  units_multiplier?: number;
}

export interface DocumentText {
  text: string;
  full_length: number;
  truncated: boolean;
}

export interface SentimentAnalysis {
  method: string;
  overall: SentimentScore;
  sections: Record<string, SentimentScore>;
}

export interface SentimentScore {
  negative: number;
  neutral: number;
  positive: number;
  compound?: number;
}

export interface QARequest {
  document_id: string;
  question: string;
}

export interface QAResponse {
  answer: string;
  confidence: number;
  sources: string[];
  processing_time: number;
}

export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  confidence?: number;
  sources?: string[];
  processing_time?: number;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface LoadingState {
  isLoading: boolean;
  error?: string;
}

// Stage 2 & 3 placeholder types
export interface ForecastRequest {
  document_ids: string[];
  forecast_periods: number;
  metrics: string[];
}

export interface InvestmentRecommendation {
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  reasoning: string;
  target_price?: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
}
