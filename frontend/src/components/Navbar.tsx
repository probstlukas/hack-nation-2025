import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { BarChart3, Home, TrendingUp, Target, HelpCircle } from 'lucide-react';

const Navbar: React.FC = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: Home },
    { path: '/forecasting', label: 'Forecasting', icon: TrendingUp },
    { path: '/investment-strategy', label: 'Investment Strategy', icon: Target },
    { path: '/how-to-use', label: 'How to Use', icon: HelpCircle },
  ];

  return (
    <nav className="backdrop-blur bg-white/40 border-b border-white/50 text-slate-800">
      <div className="container mx-auto px-6 py-3">
        <div className="flex items-center justify-between">
          {/* Logo and Brand */}
          <Link to="/" className="flex items-center gap-2 text-xl font-bold text-slate-900">
            <BarChart3 size={22} />
            <span>FinDocGPT</span>
          </Link>

          {/* Navigation Tabs */}
          <div className="flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`nav-tab ${isActive ? 'active' : ''}`}
                >
                  <Icon size={16} />
                  <span className="hidden md:inline text-sm font-semibold">{item.label}</span>
                </Link>
              );
            })}
          </div>

          {/* Sponsor */}
          <div className="hidden lg:flex items-center gap-2 text-sm text-slate-600">
            <span>Sponsored by</span>
            <span className="font-semibold text-slate-900">AkashX.ai</span>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
