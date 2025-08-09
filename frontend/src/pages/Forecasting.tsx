import React, { useState } from 'react';
import { TrendingUp, Search, LineChart as LineChartIcon, Loader2 } from 'lucide-react';
import { apiService } from '../services/api';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

const periods = [
  { label: '1y', value: '1y' },
  { label: '2y', value: '2y' },
  { label: '5y', value: '5y' },
  { label: '10y', value: '10y' },
  { label: 'Max', value: 'max' },
];

const Forecasting: React.FC = () => {
  const [ticker, setTicker] = useState('AAPL');
  const [period, setPeriod] = useState('5y');
  const [horizon, setHorizon] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any | null>(null);

  const runForecast = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!ticker.trim()) return;
    try {
      setLoading(true);
      setError(null);
      const data = await apiService.createForecast({ ticker: ticker.trim().toUpperCase(), period, forecast_periods: horizon as any, metrics: [] } as any);
      // api returns { success, data }
      if (!data.success) throw new Error(data.error || 'Forecast failed');
      setResult(data.data);
    } catch (err: any) {
      setError(err?.message || 'Failed to run forecast');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero */}
      <div className="bg-white/60 backdrop-blur border-b border-white/60">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center gap-3">
            <TrendingUp size={28} className="text-blue-600" />
            <h1 className="text-2xl font-bold">Financial Forecasting</h1>
          </div>
          <p className="text-sm text-slate-600 mt-2">Predict near‑term price levels using a lightweight RandomForest model trained on engineered features (returns, moving averages, volatility) from Yahoo Finance.</p>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        {/* Controls */}
        <form onSubmit={runForecast} className="card mb-6">
          <div className="card-body">
            <div className="grid md:grid-cols-4 gap-4">
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

        {/* Results */}
        {result && (
          <div className="card">
            <div className="card-header flex items-center justify-between">
              <h3 className="font-semibold">{result.ticker} • {result.model} • MAE: {result.mae.toFixed(2)}</h3>
              <div className="text-sm text-slate-600">Last price: ${result.last_price.toFixed(2)}</div>
            </div>
            <div className="card-body" style={{height: 380}}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={result.predictions} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} domain={["auto", "auto"]} />
                  <Tooltip />
                  <Line type="monotone" dataKey="pred" stroke="#2563eb" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Forecasting;
