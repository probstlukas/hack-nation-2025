import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import DocumentAnalysis from './pages/DocumentAnalysis';
import Forecasting from './pages/Forecasting';
import InvestmentStrategy from './pages/InvestmentStrategy';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/document/:documentId" element={<DocumentAnalysis />} />
            <Route path="/forecasting" element={<Forecasting />} />
            <Route path="/investment-strategy" element={<InvestmentStrategy />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;