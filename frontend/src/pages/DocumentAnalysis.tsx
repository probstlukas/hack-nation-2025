import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { 
  ArrowLeft, 
  MessageSquare, 
  FileText, 
  Send, 
  Loader2, 
  BarChart3,
  TrendingUp,
  ChevronRight
} from 'lucide-react';
import { apiService } from '../services/api';
import { Document, SentimentAnalysis } from '../types';
import PDFViewer from '../components/PDFViewer';

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
      // Prefer enhanced endpoint for robust document sentiment, but ignore news sentiment
      try {
        const enhanced = await apiService.getEnhancedDocumentSentiment(document.id);
        const docSent = (enhanced && (enhanced.document_sentiment || enhanced.sentiment)) ?? enhanced;
        const normalized = docSent as SentimentAnalysis;
        setSentimentData(normalized);
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
        <div className="container mx-auto px-6 py-8">
          <div className="text-center">
            <h1 className="text-2xl font-bold text-red-600 mb-4">Error</h1>
            <p className="text-gray-600 mb-4">{error || 'Document not found'}</p>
            <Link to="/" className="btn btn-primary">
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
    <div className="min-h-screen bg-gray-50 flex flex-col">
      
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="w-full max-w-screen-2xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => navigate('/')}
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

      <div className="w-full max-w-screen-2xl mx-auto px-6 py-8 flex-1 flex flex-col min-h-0">
        {/* Tab Navigation (consistent styling) */}
        <div className="bg-white rounded-t-xl shadow-sm border border-gray-200 border-b-0 w-full block">
          <div className="tabs">
            <button
              onClick={() => setActiveTab('chat')}
              className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
            >
              <MessageSquare size={18} />
              <span>Q&A Chat</span>
            </button>
            <button
              onClick={() => setActiveTab('document')}
              className={`tab ${activeTab === 'document' ? 'active' : ''}`}
            >
              <FileText size={18} />
              <span>PDF View</span>
            </button>
            <button
              onClick={() => {
                setActiveTab('sentiment');
                loadSentimentData();
              }}
              className={`tab ${activeTab === 'sentiment' ? 'active' : ''}`}
            >
              <TrendingUp size={18} />
              <span>Sentiment Analysis</span>
            </button>
          </div>
        </div>

        {/* Tab Content */}
        <div className="bg-white rounded-b-xl shadow-lg border border-gray-200 flex-1 flex flex-col min-h-0 w-full">
          {/* Q&A Chat Tab */}
          {activeTab === 'chat' && (
            <div className="p-8 flex-1 flex flex-col min-h-0">
              <div className="grid lg:grid-cols-4 gap-8 flex-1 min-h-0">
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
                          className="group w-full text-left btn btn-outline hover-lift"
                          disabled={questionLoading}
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div className="flex items-start gap-3">
                              <div className="w-8 h-8 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center shrink-0">
                                <MessageSquare size={16} />
                              </div>
                              <div className="text-slate-800 leading-relaxed">{question}</div>
                            </div>
                            <ChevronRight size={16} className="text-gray-400 group-hover:text-blue-600 transition-transform group-hover:translate-x-0.5" />
                          </div>
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
                <div className="lg:col-span-3 min-h-0 flex flex-col">
                  <div className="bg-gray-50 rounded-xl border border-gray-200 flex-1 flex flex-col min-h-[50vh]">
                    {/* Messages Area */}
                    <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scroll min-h-0">
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
                          className="flex-1 form-input resize-y min-h-24 md:min-h-28 max-h-[40vh]"
                          rows={3}
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
            <div className="p-8 grid lg:grid-cols-4 gap-8 flex-1 min-h-0">
              <div className="flex-1 min-h-0 lg:col-span-4">
                <div className="card full-height-card flex flex-col overflow-hidden w-full">
                  {document ? (
                    <PDFViewer
                      pdfUrl={`http://localhost:5001/api/documents/${document.id}/pdf`}
                      documentName={document.name}
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
            </div>
          )}

          {/* Sentiment Analysis Tab */}
          {activeTab === 'sentiment' && (
            <div className="p-8 flex-1 flex flex-col min-h-0">
              <div className="flex-1 min-h-0 w-full">
                {sentimentLoading ? (
                  <div className="flex items-center justify-center h-64">
                    <div className="flex items-center gap-3">
                      <Loader2 className="animate-spin" size={24} />
                      <span className="text-lg">Analyzing document sentiment...</span>
                    </div>
                  </div>
                ) : sentimentData && sentimentData.overall ? (
                  <div className="space-y-6 w-full">
                    {/* Overall Sentiment */}
                    <div className="card">
                      <div className="card-header">
                        <h3 className="text-xl font-semibold">Overall Document Sentiment</h3>
                      </div>
                      <div className="card-body">
                        <div className="grid md:grid-cols-3 gap-4">
                          <div className="text-center p-4 bg-green-50 rounded-lg">
                            <div className="text-2xl font-bold text-green-600">
                              {((sentimentData.overall?.positive ?? 0) * 100).toFixed(1)}%
                            </div>
                            <div className="text-sm text-green-700 font-medium">Positive</div>
                          </div>
                          <div className="text-center p-4 bg-gray-50 rounded-lg">
                            <div className="text-2xl font-bold text-gray-600">
                              {((sentimentData.overall?.neutral ?? 0) * 100).toFixed(1)}%
                            </div>
                            <div className="text-sm text-gray-700 font-medium">Neutral</div>
                          </div>
                          <div className="text-center p-4 bg-red-50 rounded-lg">
                            <div className="text-2xl font-bold text-red-600">
                              {((sentimentData.overall?.negative ?? 0) * 100).toFixed(1)}%
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
                                      {(((sentiment?.positive ?? 0) as number) * 100).toFixed(1)}%
                                    </div>
                                    <div className="text-gray-600">Positive</div>
                                  </div>
                                  <div className="text-center">
                                    <div className="font-semibold text-gray-600">
                                      {(((sentiment?.neutral ?? 0) as number) * 100).toFixed(1)}%
                                    </div>
                                    <div className="text-gray-600">Neutral</div>
                                  </div>
                                  <div className="text-center">
                                    <div className="font-semibold text-red-600">
                                      {(((sentiment?.negative ?? 0) as number) * 100).toFixed(1)}%
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
                  </div>
                ) : (
                  <div className="text-gray-500">No sentiment data available.</div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DocumentAnalysis;