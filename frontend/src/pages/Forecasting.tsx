import React, { useState } from 'react';
import { TrendingUp, Search, LineChart as LineChartIcon, Loader2, Newspaper, BrainCircuit } from 'lucide-react';
import { apiService } from '../services/api';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine, Legend, Area, AreaChart } from 'recharts';

const periods = [
  { label: '1y', value: '1y' },
  { label: '2y', value: '2y' },
  { label: '5y', value: '5y' },
  { label: '10y', value: '10y' },
  { label: 'Max', value: 'max' },
];

const modelOptions = [
  { label: 'RandomForest', value: 'rf' },
  { label: 'Prophet', value: 'prophet' },
  { label: 'LSTM', value: 'lstm' },
];

const Forecasting: React.FC = () => {
  const [ticker, setTicker] = useState('AAPL');
  const [period, setPeriod] = useState('5y');
  const [horizon, setHorizon] = useState(5);
  const [model, setModel] = useState('rf');
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

  const combinedData = React.useMemo((): { hist: any[]; preds: any[]; lastDate: string | undefined } => {
    if (!result) return { hist: [], preds: [], lastDate: undefined };
    const hist = (result.history || []).map((d: any) => ({ date: d.date, close: d.close }));
    const lastDate = hist.length ? hist[hist.length - 1].date : undefined;
    const preds = (result.predictions || []).map((d: any) => ({ date: d.date, pred: d.pred }));
    return { hist, preds, lastDate };
  }, [result]);

  const overallNews = news?.overall_sentiment || { positive: 0, neutral: 0, negative: 0 };

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
                <label className="form-label">Ticker</label>
                <div className="relative">
                  <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                  <input className="form-input pl-9" placeholder="AAPL, NVDA, MSFT" value={ticker} onChange={(e)=>setTicker(e.target.value)} />
                </div>
              </div>

              <div>
                <label className="form-label">Period</label>
                <select className="form-input" value={period} onChange={(e)=>setPeriod(e.target.value)}>
                  {periods.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
                </select>
              </div>

              <div>
                <label className="form-label">Horizon (days)</label>
                <input className="form-input" type="number" min={1} max={30} value={horizon} onChange={(e)=>setHorizon(parseInt(e.target.value||'5'))} />
              </div>

              <div>
                <label className="form-label">Model</label>
                <select className="form-input" value={model} onChange={(e)=>setModel(e.target.value)}>
                  {modelOptions.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
                </select>
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
              <div className="card-body" style={{height: 440}}>
                {result ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" tick={{ fontSize: 12 }} allowDuplicatedCategory={false} />
                      <YAxis tick={{ fontSize: 12 }} domain={["auto", "auto"]} />
                      <Tooltip />
                      <Legend />
                      <Line dataKey="close" name="History" stroke="#0ea5e9" strokeWidth={2} dot={false} data={combinedData.hist} />
                      {combinedData.lastDate && (
                        <ReferenceLine x={combinedData.lastDate} stroke="#94a3b8" strokeDasharray="3 3" />
                      )}
                      <Line dataKey="pred" name="Forecast" stroke="#2563eb" strokeWidth={2} dot={false} data={combinedData.preds} />
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
                    <div className="grid grid-cols-3 gap-3 text-center">
                      <div className="bg-green-50 rounded p-3">
                        <div className="text-lg font-bold text-green-600">{(overallNews.positive * 100).toFixed(0)}%</div>
                        <div className="text-xs text-green-700">Positive</div>
                      </div>
                      <div className="bg-gray-50 rounded p-3">
                        <div className="text-lg font-bold text-gray-600">{(overallNews.neutral * 100).toFixed(0)}%</div>
                        <div className="text-xs text-gray-700">Neutral</div>
                      </div>
                      <div className="bg-red-50 rounded p-3">
                        <div className="text-lg font-bold text-red-600">{(overallNews.negative * 100).toFixed(0)}%</div>
                        <div className="text-xs text-red-700">Negative</div>
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
                        {(news.recent_headlines || news.articles || []).slice(0,5).map((h: any, i: number) => (
                          <div key={i} className="border border-gray-200 rounded p-2">
                            <div className="text-sm text-slate-900">{h.title}</div>
                            {h.sentiment_label && (
                              <div className={`inline-flex px-2 py-0.5 rounded-full text-xs mt-1 ${
                                h.sentiment_label === 'positive' ? 'bg-green-100 text-green-700' :
                                h.sentiment_label === 'negative' ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-700'
                              }`}>
                                {h.sentiment_label}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                )}

                {!newsLoading && !news && (
                  <div className="text-sm text-slate-500">Run a forecast to load news sentiment for the ticker.</div>
                )}
              </div>
            </div>

            {/* Models blurb */}
            <div className="card mt-6">
              <div className="card-header flex items-center gap-2">
                <BrainCircuit size={18} />
                <h3 className="font-semibold">Forecasting Models</h3>
              </div>
              <div className="card-body text-sm text-slate-700 space-y-2">
                <p>Choose between RandomForest, Prophet, or LSTM. Prophet may take longer on first run as it compiles CmdStan; LSTM trains a small neural net.</p>
                <ul className="list-disc pl-4 space-y-1">
                  <li>Predict near‑term movements (1–30 days)</li>
                  <li>Evaluate with MAE against holdout</li>
                  <li>Overlay predictions on recent price chart</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Forecasting;
