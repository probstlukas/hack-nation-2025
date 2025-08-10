import React, { useState } from 'react';
import { TrendingUp, Search, LineChart as LineChartIcon, Loader2, Newspaper, ExternalLink, FileText } from 'lucide-react';
import { apiService } from '../services/api';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip as RechartsTooltip, CartesianGrid, ReferenceLine, Legend, Brush, AreaChart, Area, BarChart, Bar, Cell } from 'recharts';
import { Document as FinDoc, SentimentAnalysis } from '../types';

const periods = [
  { label: '1y', value: '1y' },
  { label: '2y', value: '2y' },
  { label: '5y', value: '5y' },
  { label: '10y', value: '10y' },
  { label: 'Max', value: 'max' },
];

const modelOptions = [
  { label: 'LSTM', value: 'lstm' },
  { label: 'RandomForest', value: 'rf' },
  { label: 'Prophet', value: 'prophet' },
  
];

const Forecasting: React.FC = () => {
  const [ticker, setTicker] = useState('AAPL');
  const [period, setPeriod] = useState('5y');
  const [horizon, setHorizon] = useState(5);
  const [model, setModel] = useState('lstm');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any | null>(null);
  const [news, setNews] = useState<any | null>(null);
  const [newsLoading, setNewsLoading] = useState(false);
  const [allDocs, setAllDocs] = useState<FinDoc[]>([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [docSentiments, setDocSentiments] = useState<Array<{ date: string; sentiment: number; id: string; company: string; label?: string }>>([]);
  const [docSentimentCache, setDocSentimentCache] = useState<Record<string, number>>({});
  const [visibleRange, setVisibleRange] = useState<{ start: string; end: string } | null>(null);

  const runForecast = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!ticker.trim()) return;
    try {
      setLoading(true);
      setError(null);
      setResult(null);
      const data = await apiService.createForecast({ ticker: ticker.trim().toUpperCase(), period, horizon, model } as any);
      if (!data.success) throw new Error(data.error || 'Forecast failed');
      setResult(data.data);
      // Load news sentiment alongside forecast
      setNewsLoading(true);
      try {
        const sentiment = await apiService.getCompanyNewsSentiment(ticker.trim().toUpperCase(), 30);
        setNews(sentiment);
      } catch {
        setNews(null);
      } finally {
        setNewsLoading(false);
      }
    } catch (err: any) {
      setError(err?.message || 'Failed to run forecast');
    } finally {
      setLoading(false);
    }
  };

  // Load documents once
  React.useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        setDocsLoading(true);
        const docs = await apiService.getDocuments();
        if (mounted) setAllDocs(docs);
      } catch (e) {
        // no-op
      } finally {
        if (mounted) setDocsLoading(false);
      }
    })();
    return () => { mounted = false; };
  }, []);

  const combinedData = React.useMemo((): { hist: any[]; preds: any[]; lastDate: string | undefined; merged: any[] } => {
    if (!result) return { hist: [], preds: [], lastDate: undefined, merged: [] };
    const hist = (result.history || []).map((d: any) => ({ date: d.date, close: d.close }));
    const lastDate = hist.length ? hist[hist.length - 1].date : undefined;
    const preds = (result.predictions || []).map((d: any) => ({ date: d.date, pred: d.pred }));

    // Merge by date for a single data source in LineChart (required for Brush)
    const map = new Map<string, any>();
    for (const h of hist) {
      map.set(h.date, { date: h.date, close: h.close });
    }
    for (const p of preds) {
      const existing = map.get(p.date) || { date: p.date };
      existing.pred = p.pred;
      map.set(p.date, existing);
    }
    const merged = Array.from(map.values()).sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));

    return { hist, preds, lastDate, merged };
  }, [result]);

  // Determine chart date range
  const chartRange = React.useMemo(() => {
    const merged = (combinedData && combinedData.merged) || [];
    if (!merged.length) return null;
    const start = new Date(merged[0].date);
    const end = new Date(merged[merged.length - 1].date);
    return { start, end };
  }, [combinedData]);

  // Map ticker to possible company names
  const mapTickerToCandidates = React.useCallback((t: string): string[] => {
    const T = (t || '').toUpperCase();
    const m: Record<string, string[]> = {
      AAPL: ['Apple'],
      NVDA: ['NVIDIA'],
      MSFT: ['Microsoft'],
      AMZN: ['Amazon'],
      META: ['Meta', 'Facebook'],
      TSLA: ['Tesla'],
      GOOGL: ['Alphabet', 'Google'],
      GOOG: ['Alphabet', 'Google'],
      MMM: ['3M'],
      ADBE: ['Adobe'],
      AES: ['AES'],
      ATVI: ['Activision', 'Activision Blizzard', 'ActivisionBlizzard'],
    };
    return m[T] ? m[T] : [T];
  }, []);

  // Helper to get an ISO date for a document period/year
  const docToDate = React.useCallback((doc: FinDoc): string | null => {
    const yAny = (doc.year as any);
    const yearNum = typeof yAny === 'number' ? yAny : parseInt(String(yAny || ''), 10);
    const p = String(doc.period || '');
    if (!isNaN(yearNum)) {
      const q = /Q([1-4])/i.exec(p || '');
      if (q) {
        const qn = parseInt(q[1], 10);
        const monthDay = qn === 1 ? '03-31' : qn === 2 ? '06-30' : qn === 3 ? '09-30' : '12-31';
        return `${yearNum}-${monthDay}`;
      }
      return `${yearNum}-12-31`;
    }
    // fallback: try to extract a 4-digit year from period
    const yMatch = /(20\d{2}|19\d{2})/.exec(p);
    if (yMatch) return `${yMatch[1]}-12-31`;
    return null;
  }, []);

  // Build daily sentiment series from news.sentiment_trend for bar visualization
  const sentimentSeries = React.useMemo(() => {
    const trend = (news && Array.isArray(news.sentiment_trend)) ? news.sentiment_trend : [];
    return trend.map((d: any) => ({
      date: d.date,
      sentiment: typeof d.sentiment === 'number' ? Math.max(-1, Math.min(1, d.sentiment)) : 0,
      label: d.label,
      count: d.article_count || d.count || 0,
    }));
  }, [news]);

  // Compute and fetch document sentiments when forecast/news are available
  React.useEffect(() => {
    let cancelled = false;
    const run = async () => {
      if (!result || !chartRange || !allDocs.length) return;
      const candidates = mapTickerToCandidates(result.ticker || ticker);
      const start = chartRange.start.getTime();
      const end = chartRange.end.getTime();
      // Filter docs by company match and within chart range
      const matched = allDocs.filter(d => {
        const comp = (d.company || '').toLowerCase();
        const hit = candidates.some(name => comp.includes(name.toLowerCase()));
        if (!hit) return false;
        const iso = docToDate(d);
        if (!iso) return false;
        const t = new Date(iso).getTime();
        return t >= start && t <= end;
      });

      // Limit to avoid many requests
      const limited = matched.slice(0, 12);

      // Fetch sentiments (use cache when available)
      const results: Array<{ date: string; sentiment: number; id: string; company: string; label?: string }> = [];
      for (const doc of limited) {
        const iso = docToDate(doc);
        if (!iso) continue;
        let sent = docSentimentCache[doc.id];
        if (typeof sent !== 'number') {
          try {
            // Prefer enhanced endpoint; exclude news to keep doc-only sentiment
            const enhanced = await apiService.getEnhancedDocumentSentiment(doc.id, false);
            const docSent = (enhanced && (enhanced.document_sentiment || enhanced.sentiment || enhanced)) as any;
            let compound: number | undefined = docSent?.overall?.compound;
            if (typeof compound !== 'number') {
              const pos = Number(docSent?.overall?.positive || 0);
              const neg = Number(docSent?.overall?.negative || 0);
              compound = pos - neg;
            }
            if (typeof compound === 'number') {
              sent = Math.max(-1, Math.min(1, compound));
              setDocSentimentCache(prev => ({ ...prev, [doc.id]: sent! }));
            } else {
              continue;
            }
          } catch {
            try {
              const basic = await apiService.getDocumentSentiment(doc.id);
              const pos = Number(basic?.overall?.positive || 0);
              const neg = Number(basic?.overall?.negative || 0);
              sent = Math.max(-1, Math.min(1, pos - neg));
              setDocSentimentCache(prev => ({ ...prev, [doc.id]: sent! }));
            } catch {
              continue;
            }
          }
        }
        results.push({ date: iso, sentiment: sent as number, id: doc.id, company: doc.company, label: doc.doc_type });
        if (cancelled) break;
      }
      if (!cancelled) setDocSentiments(results.sort((a, b) => a.date.localeCompare(b.date)));
    };
    run();
    return () => { cancelled = true; };
  }, [result, chartRange, allDocs, mapTickerToCandidates, docToDate, docSentimentCache, ticker]);

  // Initialize/reset visible range when merged series changes
  React.useEffect(() => {
    const merged = (combinedData && combinedData.merged) || [];
    if (merged.length) {
      const start = merged[0].date;
      const end = merged[merged.length - 1].date;
      setVisibleRange({ start, end });
    } else {
      setVisibleRange(null);
    }
  }, [combinedData]);

  const isInVisibleRange = React.useCallback((d: string) => {
    if (!visibleRange) return true;
    const t = new Date(d).getTime();
    const s = new Date(visibleRange.start).getTime();
    const e = new Date(visibleRange.end).getTime();
    return t >= s && t <= e;
  }, [visibleRange]);

  // Filtered series for rendering under the chart
  const filteredSentimentSeries = React.useMemo(() => (sentimentSeries || []).filter((d: any) => isInVisibleRange(d.date)), [sentimentSeries, isInVisibleRange]);
  const filteredDocSentiments = React.useMemo(() => (docSentiments || []).filter((d: any) => isInVisibleRange(d.date)), [docSentiments, isInVisibleRange]);

  const overallNews = news?.overall_sentiment || { positive: 0, neutral: 0, negative: 0 };
  const posPct = Math.round((overallNews.positive || 0) * 100);
  const neuPct = Math.round((overallNews.neutral || 0) * 100);
  const negPct = Math.round((overallNews.negative || 0) * 100);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero */}
      <div className="bg-white/60 backdrop-blur border-b border-white/60">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center gap-3">
            <TrendingUp size={28} className="text-blue-600" />
            <h1 className="text-2xl font-bold">Financial Forecasting</h1>
          </div>
          <p className="text-sm text-slate-600 mt-2">Visualize recent prices, overlay predictions, and factor in news sentiment.</p>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        {/* Controls */}
        <form onSubmit={runForecast} className="card mb-6">
          <div className="card-body">
            <div className="grid md:grid-cols-5 gap-4">
              <div>
                <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                <label className="form-label"> Ticker</label>
                <div className="relative">
                  
                  <input className="form-input pl-9" placeholder="AAPL, NVDA, MSFT" value={ticker} onChange={(e)=>setTicker(e.target.value)} />
                </div>
                <div style={{fontSize: 12, color: 'grey'}} className="text-xs text-slate-500 mt-1">Public market symbol (case-insensitive).</div>
              </div>

              <div>
                <label className="form-label inline-flex items-center gap-1">
                  Period
                </label>
                <div className="select-wrap">
                  <select className="form-input" value={period} onChange={(e)=>setPeriod(e.target.value)}>
                    {periods.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
                  </select>
                </div>
                <div style={{fontSize: 12, color: 'grey'}} className="text-xs text-slate-500 mt-1">History window used for training and chart context.</div>
              </div>

              <div>
                <label className="form-label inline-flex items-center gap-1">
                  Horizon (days)
                </label>
                <input className="form-input" type="number" min={1} max={30} value={horizon} onChange={(e)=>setHorizon(parseInt(e.target.value||'5'))} />
                <div style={{fontSize: 12, color: 'grey'}} className="text-xs text-slate-500 mt-1">Forecast length (future trading days).</div>
              </div>

              <div>
                <label className="form-label inline-flex items-center gap-1">
                  Model
                </label>
                <div className="select-wrap">
                  <select className="form-input" value={model} onChange={(e)=>setModel(e.target.value)}>
                    {modelOptions.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
                  </select>
                </div>
                <div style={{fontSize: 12, color: 'grey'}} className="text-xs text-slate-500 mt-1">Choose RF, Prophet, or LSTM.</div>
              </div>

              <div className="flex items-end">
                <button type="submit" className="btn btn-primary w-full" disabled={loading}>
                  {loading ? <Loader2 className="animate-spin" size={16} /> : <LineChartIcon size={16} />}
                  Run Forecast
                </button>
              </div>
            </div>
          </div>
        </form>

        {error && (
          <div className="alert alert-danger mb-6">{error}</div>
        )}

        {/* Main Content */}
        <div className="grid lg:grid-cols-4 gap-6">
          {/* Left: Chart */}
          <div className="lg:col-span-3">
            <div className="card">
              <div className="card-header flex items-center justify-between">
                <h3 className="font-semibold flex items-center gap-2">
                  <LineChartIcon size={18} /> Price & Forecast
                </h3>
                {result && <div className="text-sm text-slate-600">{result.ticker} • {result.model} • MAE: {result.mae.toFixed(2)}</div>}
              </div>
              <div className="card-body" style={{height: 480}}>
                {result ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={combinedData.merged} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                      <YAxis tick={{ fontSize: 12 }} domain={["auto", "auto"]} />
                      <RechartsTooltip />
                      <Legend />
                      <Line dataKey="close" name="History" stroke="#0ea5e9" strokeWidth={2} dot={false} connectNulls />
                      {combinedData.lastDate && (
                        <ReferenceLine x={combinedData.lastDate} stroke="#94a3b8" strokeDasharray="3 3" />
                      )}
                      <Line dataKey="pred" name="Forecast" stroke="#2563eb" strokeWidth={2} dot={false} connectNulls />
                      <Brush
                        dataKey="date"
                        height={40}
                        travellerWidth={10}
                        stroke="#cbd5e1"
                        fill="#f8fafc"
                        className="mt-2"
                        onChange={(range: any) => {
                          try {
                            const startIdx = Math.max(0, Math.min(range?.startIndex ?? 0, combinedData.merged.length - 1));
                            const endIdx = Math.max(0, Math.min(range?.endIndex ?? combinedData.merged.length - 1, combinedData.merged.length - 1));
                            const start = combinedData.merged[startIdx]?.date;
                            const end = combinedData.merged[endIdx]?.date;
                            if (start && end) setVisibleRange({ start, end });
                          } catch {
                            // ignore
                          }
                        }}
                      >
                        <AreaChart data={combinedData.merged}>
                          <defs>
                            <linearGradient id="brushColor" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor="#94a3b8" stopOpacity={0.8} />
                              <stop offset="100%" stopColor="#94a3b8" stopOpacity={0.2} />
                            </linearGradient>
                          </defs>
                          <Area type="monotone" dataKey="close" stroke="#94a3b8" fill="url(#brushColor)" fillOpacity={1} dot={false} />
                        </AreaChart>
                      </Brush>
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-slate-500">
                    Run a forecast to see the chart.
                  </div>
                )}
              </div>

              {/* Daily news sentiment bars (filtered by Brush range) */}
              {filteredSentimentSeries.length > 0 && (
                <div className="px-4 pb-4">
                  <div className="text-sm text-slate-600 mb-1 flex items-center gap-2">
                    <Newspaper size={16} /> Daily news sentiment
                  </div>
                  <div style={{ height: 120 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={filteredSentimentSeries} margin={{ top: 4, right: 20, left: 0, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                        <YAxis domain={[-1, 1]} tick={{ fontSize: 10 }} />
                        <RechartsTooltip formatter={(v: any) => [typeof v === 'number' ? v.toFixed(2) : v, 'Sentiment']} />
                        <ReferenceLine y={0} stroke="#94a3b8" />
                        <Bar dataKey="sentiment" radius={[4, 4, 0, 0]}>
                          {filteredSentimentSeries.map((entry: any, index: number) => {
                            const s = entry.sentiment as number;
                            const color = s > 0.1 ? '#16a34a' : s < -0.1 ? '#dc2626' : '#64748b';
                            return <Cell key={`cell-${index}`} fill={color} />;
                          })}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Document sentiment bars (filtered by Brush range) */}
              {filteredDocSentiments.length > 0 && (
                <div className="px-4 pb-4">
                  <div className="text-sm text-slate-600 mb-1 flex items-center gap-2">
                    <FileText size={16} /> Document sentiment (FinanceBench)
                  </div>
                  <div style={{ height: 100 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={filteredDocSentiments} margin={{ top: 4, right: 20, left: 0, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                        <YAxis domain={[-1, 1]} tick={{ fontSize: 10 }} />
                        <RechartsTooltip formatter={(v: any, _n: any, p: any) => [typeof v === 'number' ? v.toFixed(2) : v, p && p.payload ? `${p.payload.company} • ${p.payload.label || ''}` : 'Document']} />
                        <ReferenceLine y={0} stroke="#94a3b8" />
                        <Bar dataKey="sentiment" radius={[4, 4, 0, 0]}>
                          {filteredDocSentiments.map((entry, index) => {
                            const s = entry.sentiment as number;
                            const color = s > 0.1 ? '#16a34a' : s < -0.1 ? '#dc2626' : '#64748b';
                            return <Cell key={`doc-cell-${index}`} fill={color} />;
                          })}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Recent News Sentiment: moved below and matched width to chart (col-span-3) */}
          <div className="lg:col-span-3">
            <div className="card h-full">
              <div className="card-header flex items-center gap-2">
                <Newspaper size={18} />
                <h3 className="font-semibold">Recent News Sentiment</h3>
              </div>
              <div className="card-body space-y-4">
                {newsLoading && (
                  <div className="flex items-center gap-2 text-slate-600">
                    <Loader2 className="animate-spin" size={16} /> Loading news sentiment...
                  </div>
                )}

                {news && (
                  <>
                    {/* Colored bars */}
                    <div className="space-y-2">
                      <div>
                        <div className="flex justify-between text-xs mb-1"><span className="text-green-700">Positive</span><span className="text-green-700">{posPct}%</span></div>
                        <div className="w-full h-2 bg-green-100 rounded">
                          <div className="h-2 bg-green-500 rounded" style={{ width: `${posPct}%` }} />
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-xs mb-1"><span className="text-gray-700">Neutral</span><span className="text-gray-700">{neuPct}%</span></div>
                        <div className="w-full h-2 bg-gray-100 rounded">
                          <div className="h-2 bg-gray-500 rounded" style={{ width: `${neuPct}%` }} />
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-xs mb-1"><span className="text-red-700">Negative</span><span className="text-red-700">{negPct}%</span></div>
                        <div className="w-full h-2 bg-red-100 rounded">
                          <div className="h-2 bg-red-500 rounded" style={{ width: `${negPct}%` }} />
                        </div>
                      </div>
                    </div>

                    {news.summary && (
                      <div className="bg-blue-50 border border-blue-200 rounded p-3 text-sm text-blue-900">
                        {news.summary}
                      </div>
                    )}

                    <div>
                      <h4 className="font-medium text-slate-900 mb-2">Recent Headlines</h4>
                      <div className="space-y-2">
                        {(news.recent_headlines || news.articles || []).slice(0,5).map((h: any, i: number) => {
                          const label = h.sentiment_label;
                          const border = label === 'positive' ? 'border-green-200' : label === 'negative' ? 'border-red-200' : 'border-gray-200';
                          const badge = label === 'positive' ? 'bg-green-100 text-green-700' : label === 'negative' ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-700';
                          const title = h.title || 'Untitled';
                          const url = h.url;
                          const source = h.source || '';
                          return (
                            <div key={i} className={`border ${border} rounded p-2`}>
                              <div className="flex items-start justify-between gap-3">
                                {url ? (
                                  <a href={url} target="_blank" rel="noopener noreferrer" className="text-sm text-slate-900 hover:text-blue-600 underline flex-1 inline-flex items-center gap-1">
                                    {title}
                                    <ExternalLink size={14} className="text-blue-600" />
                                  </a>
                                ) : (
                                  <div className="text-sm text-slate-900 flex-1">{title}</div>
                                )}
                                {label && <div className={`inline-flex px-2 py-0.5 rounded-full text-xs ${badge}`}>{label}</div>}
                              </div>
                              {(source || h.published_at) && (
                                <div className="mt-1 text-xs text-slate-500">{[source, h.published_at]?.filter(Boolean).join(' • ')}</div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </>
                )}

                {!newsLoading && !news && (
                  <div className="text-sm text-slate-500">Run a forecast to load news sentiment for the ticker.</div>
                )}
              </div>
            </div>
          </div>

          {/* remove previous right-column news panel */}
        </div>
      </div>
    </div>
  );
};

export default Forecasting;
