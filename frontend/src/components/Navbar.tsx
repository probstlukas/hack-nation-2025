import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { BarChart3, Home, TrendingUp, Target } from 'lucide-react';

const Navbar: React.FC = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: Home },
    { path: '/forecasting', label: 'Forecasting', icon: TrendingUp },
    { path: '/investment-strategy', label: 'Investment Strategy', icon: Target },
  ];

  return (
    <nav className="bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo and Brand */}
          <Link to="/" className="flex items-center gap-3 text-xl font-bold">
            <BarChart3 size={28} />
            <span className="bg-gradient-to-r from-white to-blue-100 bg-clip-text text-transparent">
              FinDocGPT
            </span>
          </Link>

          {/* Navigation Links */}
          <div className="flex items-center gap-6">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                    isActive
                      ? 'bg-white/20 text-white font-medium'
                      : 'text-blue-100 hover:text-white hover:bg-white/10'
                  }`}
                >
                  <Icon size={18} />
                  <span className="hidden md:inline">{item.label}</span>
                </Link>
              );
            })}
          </div>

          {/* Sponsor Badge */}
          <div className="hidden lg:flex items-center gap-2 text-sm text-blue-100">
            <span>Sponsored by</span>
            <span className="font-semibold text-white">AkashX.ai</span>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
