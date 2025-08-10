import React, { useState } from 'react';
import { BrainCircuit, Target, TrendingUp, Shield, Loader2 } from 'lucide-react';
import { apiService } from '../services/api';
import { InvestmentRecommendation } from '../types';

const InvestmentStrategy: React.FC = () => {
  const [docId, setDocId] = useState('');
  const [ticker, setTicker] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rec, setRec] = useState<InvestmentRecommendation | null>(null);

  const run = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!docId.trim()) return;
    try {
      setLoading(true);
      setError(null);
      setRec(null);
      const data = await apiService.getInvestmentRecommendation({ document_id: docId.trim(), ticker: ticker.trim() || undefined });
      setRec(data);
    } catch (err: any) {
      setError(err?.message || 'Failed to get recommendation');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white/60 backdrop-blur border-b">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center gap-3">
            <BrainCircuit className="text-green-600" size={24} />
            <h1 className="text-2xl font-bold">Investment Strategy</h1>
          </div>
          <p className="text-sm text-slate-600 mt-2">Combine document sentiment, news, and forecasts to recommend Buy/Sell/Hold.</p>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        <form onSubmit={run} className="card mb-6">
          <div className="card-body grid md:grid-cols-3 gap-4">
            <div>
              <label className="form-label">Document ID</label>
              <input className="form-input" placeholder="e.g. 3M_2015_10K.pdf" value={docId} onChange={(e)=>setDocId(e.target.value)} />
              <div className="text-xs text-slate-500 mt-1">Use an ID from the Dashboard list.</div>
            </div>
            <div>
              <label className="form-label">Ticker (optional)</label>
              <input className="form-input" placeholder="e.g. MMM" value={ticker} onChange={(e)=>setTicker(e.target.value)} />
              <div className="text-xs text-slate-500 mt-1">If omitted, backend will try to infer.</div>
            </div>
            <div className="flex items-end">
              <button type="submit" className="btn btn-primary w-full" disabled={loading}>
                {loading ? <Loader2 className="animate-spin" size={16} /> : 'Get Recommendation'}
              </button>
            </div>
          </div>
        </form>

        {error && <div className="alert alert-danger mb-6">{error}</div>}

        {rec && (
          <div className="grid lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 card">
              <div className="card-header flex items-center gap-2">
                <Target size={18} className="text-green-600" />
                <h3 className="font-semibold">Recommendation</h3>
              </div>
              <div className="card-body">
                <div className="flex items-center gap-3">
                  <span className={`px-3 py-1 rounded-full text-sm font-semibold ${rec.action === 'BUY' ? 'bg-green-100 text-green-700' : rec.action === 'SELL' ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-700'}`}>{rec.action}</span>
                  <span className="text-sm text-slate-600">Confidence: {(rec.confidence * 100).toFixed(0)}%</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${rec.risk_level === 'HIGH' ? 'bg-red-100 text-red-700' : rec.risk_level === 'LOW' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>Risk: {rec.risk_level}</span>
                </div>
                {rec.target_price && (
                  <div className="mt-3 inline-flex items-center gap-2 text-slate-700">
                    <TrendingUp size={16} /> Target price: ${rec.target_price.toFixed(2)}
                  </div>
                )}
                <div className="mt-4 whitespace-pre-line text-slate-800 text-sm">{rec.reasoning}</div>
              </div>
            </div>

            <div className="card">
              <div className="card-header flex items-center gap-2">
                <Shield size={18} />
                <h3 className="font-semibold">Details</h3>
              </div>
              <div className="card-body text-sm text-slate-700 space-y-1">
                <div>Document score: {((rec as any).components?.doc_score ?? 0).toFixed(2)}</div>
                {typeof (rec as any).components?.news_score === 'number' && (
                  <div>News score: {((rec as any).components?.news_score ?? 0).toFixed(2)}</div>
                )}
                {typeof (rec as any).components?.forecast_score === 'number' && (
                  <div>Forecast score: {((rec as any).components?.forecast_score ?? 0).toFixed(2)}</div>
                )}
                {((rec as any).components?.forecast?.ticker) && (
                  <div className="text-xs text-slate-500">Ticker: {(rec as any).components?.forecast?.ticker}</div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default InvestmentStrategy;
