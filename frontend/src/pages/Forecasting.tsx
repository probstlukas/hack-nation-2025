import React, { useState } from 'react';
import { TrendingUp, Search, LineChart as LineChartIcon, Loader2, Newspaper, ExternalLink } from 'lucide-react';
import { apiService } from '../services/api';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip as RechartsTooltip, CartesianGrid, ReferenceLine, Legend, Brush, AreaChart, Area } from 'recharts';

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
                <select className="form-input" value={period} onChange={(e)=>setPeriod(e.target.value)}>
                  {periods.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
                </select>
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
                <select className="form-input" value={model} onChange={(e)=>setModel(e.target.value)}>
                  {modelOptions.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
                </select>
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
            </div>
          </div>

          {/* Right: News Sentiment */}
          <div className="lg:col-span-1">
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
        </div>
      </div>
    </div>
  );
};

export default Forecasting;
