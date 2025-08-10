import React from 'react';
import { FileText, Target, TrendingUp, HelpCircle } from 'lucide-react';

const HowToUse: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="bg-white/60 backdrop-blur border-b">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center gap-3">
            <HelpCircle size={24} className="text-indigo-600" />
            <h1 className="text-2xl font-bold">How to Use</h1>
          </div>
          <p className="text-sm text-slate-600 mt-2">Quick guide to explore documents, run forecasts, and get investment recommendations.</p>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        <div className="grid lg:grid-cols-3 gap-6">
          <div className="card">
            <div className="card-header flex items-center gap-2">
              <FileText size={18} />
              <h3 className="font-semibold">1. Browse Documents</h3>
            </div>
            <div className="card-body text-sm text-slate-700 space-y-2">
              <p>Go to the Dashboard and use search and filters to find a company filing.</p>
              <p>Open a document to view it and ask questions about the content.</p>
            </div>
          </div>

          <div className="card">
            <div className="card-header flex items-center gap-2">
              <TrendingUp size={18} />
              <h3 className="font-semibold">2. Run Forecasts</h3>
            </div>
            <div className="card-body text-sm text-slate-700 space-y-2">
              <p>Use the Forecasting page to select a ticker, history period, horizon, and model.</p>
              <p>View historical prices and overlaid predictions. News sentiment loads automatically.</p>
            </div>
          </div>

          <div className="card">
            <div className="card-header flex items-center gap-2">
              <Target size={18} />
              <h3 className="font-semibold">3. Investment Strategy</h3>
            </div>
            <div className="card-body text-sm text-slate-700 space-y-2">
              <p>On the Investment Strategy page, enter a document ID and optional ticker.</p>
              <p>Get a Buy/Sell/Hold recommendation with confidence, risk, and rationale.</p>
            </div>
          </div>
        </div>

        <div className="card mt-6">
          <div className="card-header">
            <h3 className="font-semibold">Tips</h3>
          </div>
          <div className="card-body text-sm text-slate-700 space-y-2">
            <ul className="list-disc pl-4 space-y-1">
              <li>The default forecast model is LSTM; try others if you prefer.</li>
              <li>News sentiment needs a valid API key configured in the backend environment.</li>
              <li>Use the Dashboard to copy the exact document ID for recommendations.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HowToUse;
