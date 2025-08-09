import React, { useState, useEffect, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { 
  ArrowLeft, 
  MessageSquare, 
  Send, 
  FileText, 
  BarChart3,
  ExternalLink,
  Loader2
} from 'lucide-react';
import { Document, ChatMessage, QARequest } from '../types';
import { apiService } from '../services/api';
import PDFViewer from '../components/PDFViewer';

const DocumentAnalysis: React.FC = () => {
  const { documentId } = useParams<{ documentId: string }>();
  const [document, setDocument] = useState<Document | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [loading, setLoading] = useState(true);
  const [questionLoading, setQuestionLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'chat' | 'document'>('chat');
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatInputRef = useRef<HTMLTextAreaElement>(null);

  const loadDocument = React.useCallback(async () => {
    if (!documentId) return;

    try {
      setLoading(true);
      
      // Load document metadata only (PDF will be loaded by PDFViewer component)
      const docData = await apiService.getDocument(documentId);
      
      setDocument(docData);
      setError(null);
      
      // Add welcome message
      const welcomeMessage: ChatMessage = {
        id: 'welcome',
        type: 'assistant',
        content: `Hello! I'm ready to answer questions about ${docData.company}'s ${docData.doc_type} filing from ${docData.period}. What would you like to know?`,
        timestamp: new Date(),
      };
      setMessages([welcomeMessage]);
      
    } catch (err) {
      setError('Failed to load document');
      console.error('Error loading document:', err);
    } finally {
      setLoading(false);
    }
  }, [documentId]);

  useEffect(() => {
    loadDocument();
  }, [loadDocument]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);



  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmitQuestion = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentQuestion.trim() || !documentId || questionLoading) return;

    const question = currentQuestion.trim();
    setCurrentQuestion('');

    // Add user message
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: question,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMessage]);

    try {
      setQuestionLoading(true);
      
      const request: QARequest = {
        document_id: documentId,
        question: question,
      };
      
      const response = await apiService.askQuestion(request);
      
      // Add assistant response
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: response.answer,
        timestamp: new Date(),
        confidence: response.confidence,
        sources: response.sources,
        processing_time: response.processing_time,
      };
      
      setMessages(prev => [...prev, assistantMessage]);
      
    } catch (err) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: 'Sorry, I encountered an error while processing your question. Please try again.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
      console.error('Error asking question:', err);
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

  const formatNumber = (value: number | undefined, currency = 'USD') => {
    if (value === undefined || value === null) return 'N/A';
    
    if (Math.abs(value) >= 1e9) {
      return `${currency} ${(value / 1e9).toFixed(1)}B`;
    } else if (Math.abs(value) >= 1e6) {
      return `${currency} ${(value / 1e6).toFixed(1)}M`;
    } else {
      return `${currency} ${value.toLocaleString()}`;
    }
  };

  const getConfidenceBadge = (confidence?: number) => {
    if (!confidence) return null;
    
    const percentage = Math.round(confidence * 100);
    let colorClass = 'badge-secondary';
    
    if (percentage >= 90) colorClass = 'badge-success';
    else if (percentage >= 75) colorClass = 'badge-warning';
    else if (percentage >= 60) colorClass = 'badge-primary';
    
    return (
      <span className={`badge ${colorClass} text-xs`}>
        {percentage}% confident
      </span>
    );
  };

  if (loading) {
    return (
      <div className="container mx-auto px-6 py-8">
        <div className="flex items-center justify-center h-64">
          <div className="flex items-center gap-3">
            <Loader2 className="animate-spin" size={24} />
            <span>Loading document...</span>
          </div>
        </div>
      </div>
    );
  }

  if (error || !document) {
    return (
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
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link to="/" className="btn btn-outline">
                <ArrowLeft size={16} />
                Back
              </Link>
              <div>
                <h1 className="text-2xl font-bold">{document.company}</h1>
                <div className="flex items-center gap-4 text-sm text-gray-600">
                  <span className="badge badge-primary">{document.doc_type}</span>
                  <span>Period: {document.period}</span>
                  <span>Sector: {document.sector}</span>
                </div>
              </div>
            </div>
            
            {document.doc_link && (
              <a
                href={document.doc_link}
                target="_blank"
                rel="noopener noreferrer"
                className="btn btn-outline"
              >
                <ExternalLink size={16} />
                Original Document
              </a>
            )}
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Financial Metrics Sidebar */}
          <div className="space-y-6">
            {/* Key Metrics */}
            {document.financial_metrics && (
              <div className="card">
                <div className="card-header">
                  <h3 className="font-semibold flex items-center gap-2">
                    <BarChart3 size={20} />
                    Key Financial Metrics
                  </h3>
                </div>
                <div className="card-body space-y-4">
                  {document.financial_metrics.revenue && (
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Revenue:</span>
                      <span className="font-semibold">
                        {formatNumber(document.financial_metrics.revenue, document.financial_metrics.currency)}
                      </span>
                    </div>
                  )}
                  
                  {document.financial_metrics.net_income && (
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Net Income:</span>
                      <span className="font-semibold">
                        {formatNumber(document.financial_metrics.net_income, document.financial_metrics.currency)}
                      </span>
                    </div>
                  )}
                  
                  {document.financial_metrics.eps_diluted && (
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">EPS (Diluted):</span>
                      <span className="font-semibold">
                        ${document.financial_metrics.eps_diluted.toFixed(2)}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Quick Questions */}
            <div className="card">
              <div className="card-header">
                <h3 className="font-semibold">Quick Questions</h3>
              </div>
              <div className="card-body space-y-2">
                {[
                  "What is the company's revenue for this period?",
                  "What are the main business risks?",
                  "How is the company's cash flow?",
                  "What are the key business segments?"
                ].map((question, index) => (
                  <button
                    key={index}
                    onClick={() => setCurrentQuestion(question)}
                    className="w-full text-left p-3 rounded-lg border border-gray-200 hover:border-blue-300 hover:bg-blue-50 transition-colors text-sm"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-2">
            <div className="card h-[600px] flex flex-col">
              {/* Tabs */}
              <div className="card-header border-b-0 pb-0">
                <div className="flex gap-4">
                  <button
                    onClick={() => setActiveTab('chat')}
                    className={`pb-3 px-1 border-b-2 font-medium transition-colors ${
                      activeTab === 'chat'
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    <MessageSquare size={18} className="inline mr-2" />
                    Q&A Chat
                  </button>
                  <button
                    onClick={() => setActiveTab('document')}
                    className={`pb-3 px-1 border-b-2 font-medium transition-colors ${
                      activeTab === 'document'
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    <FileText size={18} className="inline mr-2" />
                    PDF View
                  </button>
                </div>
              </div>

              {/* Tab Content */}
              <div className="flex-1 overflow-hidden">
                {activeTab === 'chat' ? (
                  /* Chat Interface */
                  <div className="flex flex-col h-full">
                    {/* Messages */}
                    <div className="flex-1 overflow-y-auto p-6 space-y-4">
                      {messages.map((message) => (
                        <div
                          key={message.id}
                          className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                          <div
                            className={`max-w-[80%] rounded-lg p-4 ${
                              message.type === 'user'
                                ? 'bg-blue-600 text-white'
                                : 'bg-gray-100 text-gray-900'
                            }`}
                          >
                            <div className="text-sm mb-2">
                              {message.content}
                            </div>
                            
                            {message.type === 'assistant' && (
                              <div className="flex items-center gap-2 text-xs opacity-75">
                                {getConfidenceBadge(message.confidence)}
                                {message.processing_time && (
                                  <span>⏱ {message.processing_time}s</span>
                                )}
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                      
                      {questionLoading && (
                        <div className="flex justify-start">
                          <div className="bg-gray-100 rounded-lg p-4">
                            <div className="flex items-center gap-2">
                              <Loader2 className="animate-spin" size={16} />
                              <span className="text-sm">Thinking...</span>
                            </div>
                          </div>
                        </div>
                      )}
                      
                      <div ref={messagesEndRef} />
                    </div>

                    {/* Input */}
                    <div className="border-t border-gray-200 p-4">
                      <form onSubmit={handleSubmitQuestion} className="flex gap-3">
                        <textarea
                          ref={chatInputRef}
                          value={currentQuestion}
                          onChange={(e) => setCurrentQuestion(e.target.value)}
                          onKeyPress={handleKeyPress}
                          placeholder="Ask a question about this document..."
                          className="flex-1 form-input resize-none"
                          rows={1}
                          disabled={questionLoading}
                        />
                        <button
                          type="submit"
                          disabled={!currentQuestion.trim() || questionLoading}
                          className="btn btn-primary"
                        >
                          {questionLoading ? (
                            <Loader2 className="animate-spin" size={16} />
                          ) : (
                            <Send size={16} />
                          )}
                        </button>
                      </form>
                    </div>
                  </div>
                ) : (
                  /* PDF View */
                  <div className="h-full">
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
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocumentAnalysis;
