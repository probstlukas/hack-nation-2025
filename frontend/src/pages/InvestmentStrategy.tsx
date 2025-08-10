import React, { useState, useEffect } from 'react';
import { BrainCircuit, Target, TrendingUp, Shield, Loader2, Info, ExternalLink } from 'lucide-react';
import { apiService } from '../services/api';
import { InvestmentRecommendation, Document as FinDoc } from '../types';
import { Link } from 'react-router-dom';

// lazy load Recharts to avoid bundle errors if missing
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

const InvestmentStrategy: React.FC = () => {
  const [query, setQuery] = useState(''); // ticker or company
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rec, setRec] = useState<InvestmentRecommendation | null>(null);
  const [allDocs, setAllDocs] = useState<FinDoc[]>([]);
  const [usedDocs, setUsedDocs] = useState<FinDoc[]>([]);
  
  useEffect(() => {
    (async () => {
      try {
        const docs = await apiService.getDocuments();
        setAllDocs(docs);
      } catch {
        // ignore
      }
    })();
  }, []);

  const normalizeName = (s: string) => (s || '')
    .toLowerCase()
    .replace(/[.,'’&()\-]/g, '')
    .replace(/\b(incorporated|inc|corp|corporation|company|co|ltd|plc|llc)\b/g, '')
    .replace(/\s+/g, ' ')
    .trim();

  const matchDocsForName = (nameOrQuery: string): FinDoc[] => {
    const needle = normalizeName(nameOrQuery);
    return allDocs.filter(d => {
      const dc = normalizeName(d.company || '');
      return !!dc && (dc.includes(needle) || needle.includes(dc));
    });
  };

  const run = async (e: React.FormEvent) => {
    e.preventDefault();
    const q = (query || '').trim();
    if (!q) return;
    try {
      setLoading(true);
      setError(null);
      setRec(null);
      setUsedDocs([]);

      // Resolve ticker/company via backend lookup (Yahoo Finance/yfinance)
      let t: string | undefined = undefined;
      let displayName: string = '';
      try {
        const lookup = await apiService.lookupSymbol(q);
        if (lookup?.best?.symbol) t = String(lookup.best.symbol).toUpperCase();
        if (lookup?.best?.name) displayName = String(lookup.best.name);
      } catch {
        // ignore lookup errors; fallback below
      }
      if (!t && /^[A-Za-z.]{1,5}$/.test(q)) {
        t = q.toUpperCase();
      }

      // Always include all matching documents by company name (best name or raw query)
      const docs = matchDocsForName(displayName || q);

      const payload: any = { model: 'lstm', period: '5y', horizon: 5, include_news: true };
      if (t) payload.ticker = t;
      if (docs.length > 0) payload.document_ids = docs.map(d => d.id);

      if (docs.length) setUsedDocs(docs);

      const data = await apiService.getInvestmentRecommendation(payload);
      setRec(data);
    } catch (err: any) {
      setError(err?.message || 'Failed to get recommendation');
    } finally {
      setLoading(false);
    }
  };

  const docBreakdown = React.useMemo(() => {
    const items: any[] = (rec as any)?.components?.doc_breakdown || [];
    // Map to chart points; prefer year, fallback to parsed date/year from id
    return items
      .map((d: any) => ({
        id: d.id,
        year: d.year ?? (() => {
          const m = String(d.id || '').match(/(19|20)\d{2}/);
          return m ? Number(m[0]) : undefined;
        })(),
        score: typeof d.score === 'number' ? d.score : 0,
      }))
      .filter((d: any) => typeof d.year === 'number')
      .sort((a: any, b: any) => a.year - b.year);
  }, [rec]);

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white/60 backdrop-blur border-b">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center gap-3">
            <BrainCircuit className="text-green-600" size={24} />
            <h1 className="text-2xl font-bold">Investment Strategy</h1>
          </div>
          <p className="text-sm text-slate-600 mt-2">Enter a ticker or company name; we automatically include matching FinanceBench documents when available, otherwise use market data only.</p>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        {/* Form */}
        <form onSubmit={run} className="card mb-6">
          <div className="card-body grid md:grid-cols-3 gap-4">
            <div>
              <label className="form-label">Ticker or Company</label>
              <input className="form-input" placeholder="e.g. NVDA or NVIDIA" value={query} onChange={(e)=>setQuery(e.target.value)} />
              <div className="text-xs text-slate-500 mt-1">You can input either a market ticker or a company name.</div>
            </div>
            <div className="flex items-end gap-4 md:col-span-2">
              <button type="submit" className="btn btn-primary flex-1" disabled={loading}>
                {loading ? <Loader2 className="animate-spin" size={16} /> : 'Get Recommendation'}
              </button>
            </div>
          </div>
        </form>

        {/* Show which documents were actually used by the backend, to avoid contradictions */}
        {rec && Array.isArray((rec as any).components?.used_documents) && (rec as any).components?.used_documents.length > 0 && (
          <div className="alert alert-info mb-6">
            <div className="flex items-start gap-2">
              <Info size={16} className="mt-0.5" />
              <div className="space-y-1">
                <div>
                  Using documents: {((rec as any).components?.used_documents as string[]).map((id: string, i: number) => (
                    <span key={id}>
                      <Link to={`/document/${id}`} className="underline inline-flex items-center gap-1">
                        {id} <ExternalLink size={12} />
                      </Link>
                      {i < (((rec as any).components?.used_documents as string[]).length - 1) ? ', ' : ''}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {error && <div className="alert alert-danger mb-6">{error}</div>}

        {rec && Array.isArray((rec as any).components?.used_documents) && ((rec as any).components?.used_documents.length === 0) && (
          <div className="alert alert-info mb-6">
            <div className="flex items-start gap-2">
              <Info size={16} className="mt-0.5" />
              <div>
                No FinanceBench documents found for {query || (rec as any).components?.forecast?.ticker || 'this ticker'}. Recommendation based on {
                  [
                    typeof (rec as any).components?.news_score === 'number' ? 'news sentiment' : null,
                    typeof (rec as any).components?.forecast_score === 'number' ? 'price forecast' : null,
                  ].filter(Boolean).join(' and ') || 'available data'
                }.
              </div>
            </div>
          </div>
        )}

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
                {typeof (rec as any).components?.forecast?.last_price === 'number' && (
                  <div className="mt-3 inline-flex items-center gap-2 text-slate-700">
                    <TrendingUp size={16} /> Current price: ${((rec as any).components?.forecast?.last_price as number).toFixed(2)}
                  </div>
                )}
                {rec.target_price && (
                  <div className="mt-3 inline-flex items-center gap-2 text-slate-700">
                    <TrendingUp size={16} /> Target price: ${rec.target_price.toFixed(2)}
                  </div>
                )}
                <div className="mt-4 whitespace-pre-line text-slate-800 text-sm">{rec.reasoning}</div>

                {/* Document sentiment over time */}
                {docBreakdown && docBreakdown.length > 0 && (
                  <div className="mt-6">
                    <div className="text-sm font-semibold mb-2">Document sentiment over years</div>
                    <div style={{ width: '100%', height: 220 }}>
                      <ResponsiveContainer>
                        <LineChart data={docBreakdown} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="year" tick={{ fontSize: 12 }} />
                          <YAxis domain={[-1, 1]} tick={{ fontSize: 12 }} />
                          <Tooltip formatter={(v: any) => (typeof v === 'number' ? v.toFixed(2) : v)} labelFormatter={(l: any) => `Year: ${l}`} />
                          <Line type="monotone" dataKey="score" stroke="#16a34a" dot={{ r: 2 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="text-xs text-slate-500 mt-1">Higher is more positive; values range -1..1 (TextBlob polarity).</div>
                  </div>
                )}
              </div>
            </div>

            <div className="card">
              <div className="card-header flex items-center gap-2">
                <Shield size={18} />
                <h3 className="font-semibold">Details</h3>
              </div>
              <div className="card-body text-sm text-slate-700 space-y-1">
                {Array.isArray((rec as any).components?.used_documents) && (rec as any).components?.used_documents.length > 0 && typeof (rec as any).components?.doc_score === 'number' && (
                  <div>Document score: {((rec as any).components?.doc_score).toFixed(2)}</div>
                )}
                {/* If doc_breakdown present but doc_score is null, still show 0.00 */}
                {Array.isArray((rec as any).components?.used_documents) && (rec as any).components?.used_documents.length > 0 && typeof (rec as any).components?.doc_score !== 'number' && (rec as any).components?.doc_breakdown && (
                  <div>Document score: {(0).toFixed(2)}</div>
                )}
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
