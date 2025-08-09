import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { 
  ArrowLeft, 
  MessageSquare, 
  FileText, 
  Send, 
  Loader2, 
  BarChart3,
  TrendingUp 
} from 'lucide-react';
import Navbar from '../components/Navbar';
import { apiService } from '../services/api';
import { Document, SentimentAnalysis } from '../types';

const DocumentAnalysis: React.FC = () => {
  const { documentId } = useParams<{ documentId: string }>();
  const navigate = useNavigate();
  const [document, setDocument] = useState<Document | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'chat' | 'document' | 'sentiment'>('chat');
  
  // Chat state
  const [messages, setMessages] = useState<Array<{ id: string; type: 'user' | 'assistant'; content: string }>>([]);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [questionLoading, setQuestionLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatInputRef = useRef<HTMLTextAreaElement>(null);
  
  // Sentiment state
  const [sentimentData, setSentimentData] = useState<SentimentAnalysis | null>(null);
  const [sentimentLoading, setSentimentLoading] = useState(false);

  useEffect(() => {
    const fetchDocument = async () => {
      if (!documentId) return;
      try {
        setLoading(true);
        const doc = await apiService.getDocument(documentId);
        setDocument(doc);
        setError(null);
      } catch (err) {
        setError('Failed to load document');
        console.error('Error fetching document:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchDocument();
  }, [documentId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const formatNumber = (num: number, currency: string = 'USD') => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency,
      notation: 'compact',
      maximumFractionDigits: 1,
    }).format(num);
  };

  const handleSubmitQuestion = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentQuestion.trim() || questionLoading || !document) return;

    const userMessage = {
      id: Date.now().toString(),
      type: 'user' as const,
      content: currentQuestion,
    };

    setMessages(prev => [...prev, userMessage]);
    setQuestionLoading(true);
    
    const questionToSend = currentQuestion;
    setCurrentQuestion('');

    try {
      const response = await apiService.askQuestion({
        document_id: document.id,
        question: questionToSend
      });
      
      const assistantMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant' as const,
        content: response.answer,
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error asking question:', error);
      const errorMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant' as const,
        content: 'Sorry, I encountered an error while processing your question. Please try again.',
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setQuestionLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmitQuestion(e as any);
    }
  };

  const handleQuickQuestion = (question: string) => {
    setCurrentQuestion(question);
    chatInputRef.current?.focus();
  };

  const loadSentimentData = async () => {
    if (!document || sentimentData) return;
    
    setSentimentLoading(true);
    try {
      // Try enhanced sentiment first, fallback to basic sentiment
      try {
        const enhanced = await apiService.getEnhancedDocumentSentiment(document.id);
        setSentimentData(enhanced);
      } catch (enhancedError) {
        console.warn('Enhanced sentiment failed, trying basic sentiment:', enhancedError);
        const basic = await apiService.getDocumentSentiment(document.id);
        setSentimentData(basic);
      }
    } catch (error) {
      console.error('Error loading sentiment data:', error);
    } finally {
      setSentimentLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="flex items-center gap-3">
          <Loader2 className="animate-spin" size={24} />
          <span>Loading document...</span>
        </div>
      </div>
    );
  }

  if (error || !document) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <div className="container mx-auto px-6 py-8">
          <div className="text-center">
            <h1 className="text-2xl font-bold text-red-600 mb-4">Error</h1>
            <p className="text-gray-600 mb-4">{error || 'Document not found'}</p>
            <Link to="/dashboard" className="btn btn-primary">
              <ArrowLeft size={16} />
              Back to Dashboard
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const quickQuestions = [
    "What is the company's revenue for this period?",
    "What are the main business risks? How is the company's cash flow?",
    "What are the key business segments?"
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => navigate('/dashboard')}
                className="btn btn-outline hover-lift"
              >
                <ArrowLeft size={16} />
                Back
              </button>
              <div>
                <h1 className="text-2xl font-bold">{document.company}</h1>
                <div className="flex items-center gap-4 text-sm text-gray-600">
                  <span className="badge badge-primary">{document.doc_type}</span>
                  <span>Period: {document.period}</span>
                  <span>Sector: {document.sector}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        {/* Tab Navigation */}
        <div className="bg-white rounded-t-xl shadow-sm border border-gray-200 border-b-0">
          <div className="flex">
            <button
              onClick={() => setActiveTab('chat')}
              className={`flex items-center gap-3 px-8 py-4 font-medium text-base border-b-3 transition-all duration-200 hover-lift ${
                activeTab === 'chat'
                  ? 'border-blue-500 text-blue-600 bg-gradient-to-r from-blue-50 to-indigo-50'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-50'
              }`}
            >
              <MessageSquare size={20} />
              Q&A Chat
            </button>
            <button
              onClick={() => setActiveTab('document')}
              className={`flex items-center gap-3 px-8 py-4 font-medium text-base border-b-3 transition-all duration-200 hover-lift ${
                activeTab === 'document'
                  ? 'border-blue-500 text-blue-600 bg-gradient-to-r from-blue-50 to-indigo-50'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-50'
              }`}
            >
              <FileText size={20} />
              PDF View
            </button>
            <button
              onClick={() => {
                setActiveTab('sentiment');
                loadSentimentData();
              }}
              className={`flex items-center gap-3 px-8 py-4 font-medium text-base border-b-3 transition-all duration-200 hover-lift ${
                activeTab === 'sentiment'
                  ? 'border-blue-500 text-blue-600 bg-gradient-to-r from-blue-50 to-indigo-50'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-50'
              }`}
            >
              <TrendingUp size={20} />
              Sentiment Analysis
            </button>
          </div>
        </div>

        {/* Tab Content */}
        <div className="bg-white rounded-b-xl shadow-lg border border-gray-200 min-h-[700px]">
          {/* Q&A Chat Tab */}
          {activeTab === 'chat' && (
            <div className="p-8">
              <div className="grid lg:grid-cols-4 gap-8 h-full">
                {/* Sidebar with Quick Questions and Metrics */}
                <div className="lg:col-span-1 space-y-6">
                  {/* Quick Questions */}
                  <div className="card fade-in">
                    <div className="card-header">
                      <h3 className="font-semibold text-lg">Quick Questions</h3>
                    </div>
                    <div className="card-body space-y-3">
                      {quickQuestions.map((question, index) => (
                        <button
                          key={index}
                          onClick={() => handleQuickQuestion(question)}
                          className="w-full text-left p-4 rounded-lg border border-gray-200 hover:border-blue-300 hover:bg-blue-50 transition-all duration-200 text-sm hover-lift"
                          disabled={questionLoading}
                        >
                          {question}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Financial Metrics */}
                  {document.financial_metrics && (
                    <div className="card fade-in">
                      <div className="card-header">
                        <h3 className="font-semibold text-lg flex items-center gap-2">
                          <BarChart3 size={20} />
                          Key Metrics
                        </h3>
                      </div>
                      <div className="card-body space-y-4">
                        {document.financial_metrics.revenue && (
                          <div className="flex justify-between items-center py-2 border-b border-gray-100">
                            <span className="text-gray-600 font-medium">Revenue:</span>
                            <span className="font-semibold text-green-600">
                              {formatNumber(document.financial_metrics.revenue, document.financial_metrics.currency || 'USD')}
                            </span>
                          </div>
                        )}
                        
                        {document.financial_metrics.net_income && (
                          <div className="flex justify-between items-center py-2 border-b border-gray-100">
                            <span className="text-gray-600 font-medium">Net Income:</span>
                            <span className="font-semibold text-green-600">
                              {formatNumber(document.financial_metrics.net_income, document.financial_metrics.currency || 'USD')}
                            </span>
                          </div>
                        )}
                        
                        {document.financial_metrics.eps_diluted && (
                          <div className="flex justify-between items-center py-2">
                            <span className="text-gray-600 font-medium">EPS (Diluted):</span>
                            <span className="font-semibold text-green-600">
                              ${document.financial_metrics.eps_diluted.toFixed(2)}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {/* Chat Interface */}
                <div className="lg:col-span-3">
                  <div className="bg-gray-50 rounded-xl border border-gray-200 h-[600px] flex flex-col">
                    {/* Messages Area */}
                    <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scroll">
                      {messages.length === 0 && (
                        <div className="text-center text-gray-500 mt-20">
                          <MessageSquare size={48} className="mx-auto mb-4 text-gray-400" />
                          <p className="text-lg font-medium mb-2">Start a conversation</p>
                          <p className="text-sm">Ask questions about this financial document or use the quick questions to get started.</p>
                        </div>
                      )}
                      
                      {messages.map((message) => (
                        <div
                          key={message.id}
                          className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                          <div
                            className={`max-w-[70%] rounded-xl px-4 py-3 ${
                              message.type === 'user'
                                ? 'bg-blue-500 text-white'
                                : 'bg-white border border-gray-200 text-gray-900'
                            }`}
                          >
                            <p className="text-sm leading-relaxed">{message.content}</p>
                          </div>
                        </div>
                      ))}
                      
                      {questionLoading && (
                        <div className="flex justify-start">
                          <div className="bg-white border border-gray-200 rounded-xl px-4 py-3 max-w-[70%]">
                            <div className="flex items-center gap-2">
                              <Loader2 className="animate-spin" size={16} />
                              <span className="text-sm text-gray-600">Analyzing document...</span>
                            </div>
                          </div>
                        </div>
                      )}
                      
                      <div ref={messagesEndRef} />
                    </div>

                    {/* Input Area */}
                    <div className="border-t border-gray-200 p-4 bg-white rounded-b-xl">
                      <form onSubmit={handleSubmitQuestion} className="flex gap-3">
                        <textarea
                          ref={chatInputRef}
                          value={currentQuestion}
                          onChange={(e) => setCurrentQuestion(e.target.value)}
                          onKeyPress={handleKeyPress}
                          placeholder="Ask a question about this document..."
                          className="flex-1 form-input resize-none min-h-[44px] max-h-[120px]"
                          rows={1}
                          disabled={questionLoading}
                        />
                        <button
                          type="submit"
                          disabled={!currentQuestion.trim() || questionLoading}
                          className="btn btn-primary px-4 py-2 hover-lift"
                        >
                          {questionLoading ? (
                            <Loader2 className="animate-spin" size={18} />
                          ) : (
                            <Send size={18} />
                          )}
                        </button>
                      </form>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* PDF View Tab */}
          {activeTab === 'document' && (
            <div className="p-8">
              <div className="bg-gray-50 rounded-xl border border-gray-200 h-[800px]">
                {document ? (
                  <iframe
                    title="PDF Document"
                    src={`http://localhost:5001/api/documents/${document.id}/pdf#view=FitH&zoom=page-width`}
                    className="w-full h-full rounded-xl"
                    loading="eager"
                    allow="fullscreen"
                  />
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center">
                      <FileText size={48} className="text-gray-400 mx-auto mb-4" />
                      <p className="text-gray-500">Loading PDF document...</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Sentiment Analysis Tab */}
          {activeTab === 'sentiment' && (
            <div className="p-8">
              {sentimentLoading ? (
                <div className="flex items-center justify-center h-64">
                  <div className="flex items-center gap-3">
                    <Loader2 className="animate-spin" size={24} />
                    <span className="text-lg">Analyzing document sentiment...</span>
                  </div>
                </div>
              ) : sentimentData ? (
                <div className="space-y-6">
                  {/* Overall Sentiment */}
                  <div className="card">
                    <div className="card-header">
                      <h3 className="text-xl font-semibold">Overall Document Sentiment</h3>
                    </div>
                    <div className="card-body">
                      <div className="grid md:grid-cols-3 gap-4">
                        <div className="text-center p-4 bg-green-50 rounded-lg">
                          <div className="text-2xl font-bold text-green-600">
                            {(sentimentData.overall.positive * 100).toFixed(1)}%
                          </div>
                          <div className="text-sm text-green-700 font-medium">Positive</div>
                        </div>
                        <div className="text-center p-4 bg-gray-50 rounded-lg">
                          <div className="text-2xl font-bold text-gray-600">
                            {(sentimentData.overall.neutral * 100).toFixed(1)}%
                          </div>
                          <div className="text-sm text-gray-700 font-medium">Neutral</div>
                        </div>
                        <div className="text-center p-4 bg-red-50 rounded-lg">
                          <div className="text-2xl font-bold text-red-600">
                            {(sentimentData.overall.negative * 100).toFixed(1)}%
                          </div>
                          <div className="text-sm text-red-700 font-medium">Negative</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Section Sentiment */}
                  {sentimentData.sections && Object.keys(sentimentData.sections).length > 0 && (
                    <div className="card">
                      <div className="card-header">
                        <h3 className="text-xl font-semibold">Section-wise Sentiment Analysis</h3>
                      </div>
                      <div className="card-body">
                        <div className="space-y-4">
                          {Object.entries(sentimentData.sections).map(([sectionName, sentiment], index) => (
                            <div key={index} className="border border-gray-200 rounded-lg p-4">
                              <h4 className="font-medium text-gray-900 mb-3">{sectionName}</h4>
                              <div className="grid grid-cols-3 gap-3 text-sm">
                                <div className="text-center">
                                  <div className="font-semibold text-green-600">
                                    {(sentiment.positive * 100).toFixed(1)}%
                                  </div>
                                  <div className="text-gray-600">Positive</div>
                                </div>
                                <div className="text-center">
                                  <div className="font-semibold text-gray-600">
                                    {(sentiment.neutral * 100).toFixed(1)}%
                                  </div>
                                  <div className="text-gray-600">Neutral</div>
                                </div>
                                <div className="text-center">
                                  <div className="font-semibold text-red-600">
                                    {(sentiment.negative * 100).toFixed(1)}%
                                  </div>
                                  <div className="text-gray-600">Negative</div>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* News Sentiment */}
                  {(sentimentData as any).news_sentiment && (
                    <div className="card">
                      <div className="card-header">
                        <h3 className="text-xl font-semibold">Recent News Sentiment</h3>
                      </div>
                      <div className="card-body space-y-4">
                        {/* Overall News Sentiment */}
                        <div className="grid md:grid-cols-3 gap-4 mb-6">
                          <div className="text-center p-4 bg-green-50 rounded-lg">
                            <div className="text-xl font-bold text-green-600">
                              {((sentimentData as any).news_sentiment.overall_sentiment.positive * 100).toFixed(1)}%
                            </div>
                            <div className="text-sm text-green-700 font-medium">Positive</div>
                          </div>
                          <div className="text-center p-4 bg-gray-50 rounded-lg">
                            <div className="text-xl font-bold text-gray-600">
                              {((sentimentData as any).news_sentiment.overall_sentiment.neutral * 100).toFixed(1)}%
                            </div>
                            <div className="text-sm text-gray-700 font-medium">Neutral</div>
                          </div>
                          <div className="text-center p-4 bg-red-50 rounded-lg">
                            <div className="text-xl font-bold text-red-600">
                              {((sentimentData as any).news_sentiment.overall_sentiment.negative * 100).toFixed(1)}%
                            </div>
                            <div className="text-sm text-red-700 font-medium">Negative</div>
                          </div>
                        </div>

                        {/* News Summary */}
                        {(sentimentData as any).news_sentiment.summary && (
                          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                            <h4 className="font-medium text-blue-900 mb-2">News Summary</h4>
                            <p className="text-blue-800 text-sm leading-relaxed">
                              {(sentimentData as any).news_sentiment.summary}
                            </p>
                          </div>
                        )}

                        {/* Recent Headlines */}
                        {(sentimentData as any).news_sentiment.recent_headlines && (
                          <div>
                            <h4 className="font-medium text-gray-900 mb-3">Recent Headlines</h4>
                            <div className="space-y-3">
                              {(sentimentData as any).news_sentiment.recent_headlines.map((headline: any, index: number) => (
                                <div key={index} className="border border-gray-200 rounded-lg p-3">
                                  <div className="flex justify-between items-start gap-3">
                                    <p className="text-sm text-gray-900 leading-relaxed flex-1">
                                      {headline.title}
                                    </p>
                                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                                      headline.sentiment_label === 'positive' 
                                        ? 'bg-green-100 text-green-800'
                                        : headline.sentiment_label === 'negative'
                                        ? 'bg-red-100 text-red-800'
                                        : 'bg-gray-100 text-gray-800'
                                    }`}>
                                      {headline.sentiment_label}
                                    </span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex items-center justify-center h-64">
                  <div className="text-center">
                    <TrendingUp size={48} className="text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-500 mb-4">No sentiment data available</p>
                    <button 
                      onClick={loadSentimentData}
                      className="btn btn-primary hover-lift"
                    >
                      Load Sentiment Analysis
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DocumentAnalysis;