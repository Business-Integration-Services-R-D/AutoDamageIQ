import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Car, History, Upload, Brain, ArrowLeftRight } from 'lucide-react';

const Header = () => {
  const location = useLocation();
  
  return (
    <header className="fixed top-0 left-0 right-0 bg-white/80 backdrop-blur-lg border-b border-apple-border z-50">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
          <div className="w-10 h-10 bg-black rounded-xl flex items-center justify-center">
            <Car className="w-6 h-6 text-white" />
          </div>
          <span className="font-semibold text-xl text-apple-text">AutoDamageID</span>
        </Link>
        
        {/* Navigation */}
        <nav className="flex items-center gap-2">
          <Link 
            to="/"
            className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${
              location.pathname === '/' 
                ? 'bg-black text-white' 
                : 'text-apple-secondary hover:text-apple-text hover:bg-gray-100'
            }`}
          >
            <span className="flex items-center gap-2">
              <Upload className="w-4 h-4" />
              Analiz
            </span>
          </Link>
          <Link 
            to="/compare"
            className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${
              location.pathname === '/compare' 
                ? 'bg-black text-white' 
                : 'text-apple-secondary hover:text-apple-text hover:bg-gray-100'
            }`}
          >
            <span className="flex items-center gap-2">
              <ArrowLeftRight className="w-4 h-4" />
              Karsilastir
            </span>
          </Link>
          <Link 
            to="/history"
            className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${
              location.pathname === '/history' 
                ? 'bg-black text-white' 
                : 'text-apple-secondary hover:text-apple-text hover:bg-gray-100'
            }`}
          >
            <span className="flex items-center gap-2">
              <History className="w-4 h-4" />
              Gecmis
            </span>
          </Link>
          <Link 
            to="/training"
            className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${
              location.pathname === '/training' 
                ? 'bg-black text-white' 
                : 'text-apple-secondary hover:text-apple-text hover:bg-gray-100'
            }`}
          >
            <span className="flex items-center gap-2">
              <Brain className="w-4 h-4" />
              Egitim
            </span>
          </Link>
        </nav>
      </div>
    </header>
  );
};

export default Header;
