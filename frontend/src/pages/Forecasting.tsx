import React from 'react';
import { TrendingUp, BarChart3, Calendar, Target, ArrowRight } from 'lucide-react';

const Forecasting: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-orange-500 to-orange-600 text-white">
        <div className="container mx-auto px-6 py-16">
          <div className="text-center">
            <TrendingUp size={64} className="mx-auto mb-4" />
            <h1 className="text-4xl md:text-5xl font-bold mb-4">
              Financial Forecasting
            </h1>
            <p className="text-xl text-orange-100 mb-8 max-w-2xl mx-auto">
              Stage 2: Predict future financial outcomes based on historical data and market trends
            </p>
            <div className="inline-flex items-center gap-2 bg-orange-400 text-orange-900 px-4 py-2 rounded-full font-semibold">
              <Calendar size={20} />
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
                <BarChart3 className="text-blue-600" size={32} />
              </div>
              <h3 className="text-xl font-semibold mb-3">Predict Financial Trends</h3>
              <p className="text-gray-600">
                Forecast stock prices, earnings growth, and market performance using advanced AI models.
              </p>
            </div>
          </div>

          <div className="card">
            <div className="card-body text-center">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Target className="text-green-600" size={32} />
              </div>
              <h3 className="text-xl font-semibold mb-3">External Data Integration</h3>
              <p className="text-gray-600">
                Enhance predictions with data from Yahoo Finance API, Quandl, and Alpha Vantage.
              </p>
            </div>
          </div>

          <div className="card">
            <div className="card-body text-center">
              <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <TrendingUp className="text-purple-600" size={32} />
              </div>
              <h3 className="text-xl font-semibold mb-3">Forecasting Models</h3>
              <p className="text-gray-600">
                Build sophisticated models to predict stock movements, earnings, and market risks.
              </p>
            </div>
          </div>
        </div>

        {/* Planned Features */}
        <div className="bg-white rounded-xl shadow-lg p-8">
          <h2 className="text-3xl font-bold text-center mb-8">Planned Features</h2>
          
          <div className="space-y-6">
            <div className="flex items-start gap-4">
              <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                <ArrowRight className="text-orange-600" size={16} />
              </div>
              <div>
                <h4 className="font-semibold text-lg mb-2">Time Series Forecasting</h4>
                <p className="text-gray-600">
                  Use historical financial data to predict future trends with LSTM, ARIMA, and other time series models.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                <ArrowRight className="text-orange-600" size={16} />
              </div>
              <div>
                <h4 className="font-semibold text-lg mb-2">Multi-Asset Correlation Analysis</h4>
                <p className="text-gray-600">
                  Analyze relationships between different financial instruments and market sectors.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                <ArrowRight className="text-orange-600" size={16} />
              </div>
              <div>
                <h4 className="font-semibold text-lg mb-2">Risk Assessment Models</h4>
                <p className="text-gray-600">
                  Calculate Value at Risk (VaR), volatility forecasts, and scenario analysis.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                <ArrowRight className="text-orange-600" size={16} />
              </div>
              <div>
                <h4 className="font-semibold text-lg mb-2">External Data Integration</h4>
                <p className="text-gray-600">
                  Connect with Yahoo Finance, Quandl, and Alpha Vantage APIs for real-time market data.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                <ArrowRight className="text-orange-600" size={16} />
              </div>
              <div>
                <h4 className="font-semibold text-lg mb-2">Interactive Forecasting Dashboard</h4>
                <p className="text-gray-600">
                  Visualize predictions with interactive charts, confidence intervals, and scenario modeling.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Call to Action */}
        <div className="text-center mt-12">
          <div className="bg-gradient-to-r from-orange-50 to-yellow-50 rounded-xl p-8 border border-orange-200">
            <h3 className="text-2xl font-bold mb-4">Stay Tuned for Stage 2!</h3>
            <p className="text-gray-600 mb-6 max-w-2xl mx-auto">
              We're working hard to bring you advanced financial forecasting capabilities. 
              This stage will integrate historical data analysis with machine learning to predict future market trends.
            </p>
            <div className="inline-flex items-center gap-2 text-orange-600 font-semibold">
              <TrendingUp size={20} />
              Expected Release: Next Phase of Development
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Forecasting;
