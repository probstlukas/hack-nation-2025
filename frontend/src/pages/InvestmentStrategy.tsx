import React from 'react';
import { Target, TrendingUp, Shield, DollarSign, ArrowRight, Briefcase } from 'lucide-react';

const InvestmentStrategy: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-green-600 to-green-700 text-white">
        <div className="container mx-auto px-6 py-16">
          <div className="text-center">
            <Target size={64} className="mx-auto mb-4" />
            <h1 className="text-4xl md:text-5xl font-bold mb-4">
              Investment Strategy & Decision-Making
            </h1>
            <p className="text-xl text-green-100 mb-8 max-w-2xl mx-auto">
              Stage 3: Generate actionable buy/sell recommendations based on comprehensive AI analysis
            </p>
            <div className="inline-flex items-center gap-2 bg-green-500 text-green-900 px-4 py-2 rounded-full font-semibold">
              <Briefcase size={20} />
              Coming Soon
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-12">
        {/* Features Preview */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-12">
          <div className="card">
            <div className="card-body text-center">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <TrendingUp className="text-blue-600" size={32} />
              </div>
              <h3 className="text-xl font-semibold mb-3">Investment Decision-Making</h3>
              <p className="text-gray-600">
                Based on predictions, recommend whether to Buy, Sell, or Hold a stock with confidence scores.
              </p>
            </div>
          </div>

          <div className="card">
            <div className="card-body text-center">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Shield className="text-green-600" size={32} />
              </div>
              <h3 className="text-xl font-semibold mb-3">Strategic Decision Support</h3>
              <p className="text-gray-600">
                Use financial data, market sentiment, and forecasts to build clear investment recommendations.
              </p>
            </div>
          </div>

          <div className="card">
            <div className="card-body text-center">
              <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <DollarSign className="text-purple-600" size={32} />
              </div>
              <h3 className="text-xl font-semibold mb-3">Portfolio Optimization</h3>
              <p className="text-gray-600">
                Generate optimal portfolio allocations based on risk tolerance and investment goals.
              </p>
            </div>
          </div>
        </div>

        {/* Investment Decision Matrix */}
        <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
          <h2 className="text-3xl font-bold text-center mb-8">Investment Decision Framework</h2>
          
          <div className="grid md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <TrendingUp className="text-green-600" size={32} />
              </div>
              <h4 className="font-semibold text-lg mb-2 text-green-700">BUY</h4>
              <p className="text-sm text-gray-600">
                Strong fundamentals, positive forecast, undervalued metrics
              </p>
            </div>

            <div className="text-center">
              <div className="w-20 h-20 bg-yellow-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Shield className="text-yellow-600" size={32} />
              </div>
              <h4 className="font-semibold text-lg mb-2 text-yellow-700">HOLD</h4>
              <p className="text-sm text-gray-600">
                Stable performance, fair valuation, neutral outlook
              </p>
            </div>

            <div className="text-center">
              <div className="w-20 h-20 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Target className="text-red-600" size={32} />
              </div>
              <h4 className="font-semibold text-lg mb-2 text-red-700">SELL</h4>
              <p className="text-sm text-gray-600">
                Weak fundamentals, negative forecast, overvalued
              </p>
            </div>
          </div>
        </div>

        {/* Planned Features */}
        <div className="bg-white rounded-xl shadow-lg p-8">
          <h2 className="text-3xl font-bold text-center mb-8">Planned Features</h2>
          
          <div className="space-y-6">
            <div className="flex items-start gap-4">
              <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                <ArrowRight className="text-green-600" size={16} />
              </div>
              <div>
                <h4 className="font-semibold text-lg mb-2">Multi-Factor Analysis</h4>
                <p className="text-gray-600">
                  Combine financial metrics, market sentiment, technical indicators, and macroeconomic factors.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                <ArrowRight className="text-green-600" size={16} />
              </div>
              <div>
                <h4 className="font-semibold text-lg mb-2">Risk-Adjusted Recommendations</h4>
                <p className="text-gray-600">
                  Calculate Sharpe ratios, beta coefficients, and risk-adjusted returns for each recommendation.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                <ArrowRight className="text-green-600" size={16} />
              </div>
              <div>
                <h4 className="font-semibold text-lg mb-2">Portfolio Construction</h4>
                <p className="text-gray-600">
                  Build diversified portfolios with optimal asset allocation based on modern portfolio theory.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                <ArrowRight className="text-green-600" size={16} />
              </div>
              <div>
                <h4 className="font-semibold text-lg mb-2">Backtesting & Performance</h4>
                <p className="text-gray-600">
                  Test investment strategies against historical data and track recommendation performance.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                <ArrowRight className="text-green-600" size={16} />
              </div>
              <div>
                <h4 className="font-semibold text-lg mb-2">Real-Time Decision Support</h4>
                <p className="text-gray-600">
                  Provide live investment alerts and recommendations based on market changes and news events.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                <ArrowRight className="text-green-600" size={16} />
              </div>
              <div>
                <h4 className="font-semibold text-lg mb-2">ESG Integration</h4>
                <p className="text-gray-600">
                  Incorporate Environmental, Social, and Governance factors into investment decisions.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Value Proposition */}
        <div className="text-center mt-12">
          <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-xl p-8 border border-green-200">
            <h3 className="text-2xl font-bold mb-4">Complete Investment Workflow</h3>
            <p className="text-gray-600 mb-6 max-w-3xl mx-auto">
              Stage 3 will combine insights from document analysis (Stage 1) and financial forecasting (Stage 2) 
              to deliver comprehensive investment recommendations with clear reasoning and risk assessment.
            </p>
            <div className="flex flex-col md:flex-row items-center justify-center gap-6">
              <div className="flex items-center gap-2 text-blue-600 font-semibold">
                <Target size={20} />
                Data-Driven Decisions
              </div>
              <div className="flex items-center gap-2 text-green-600 font-semibold">
                <Shield size={20} />
                Risk Management
              </div>
              <div className="flex items-center gap-2 text-purple-600 font-semibold">
                <DollarSign size={20} />
                Optimized Returns
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InvestmentStrategy;
