import React, { useEffect, useState } from 'react';
import { ShieldCheck, Activity, Terminal, FileText, Cpu, Wifi, WifiOff } from 'lucide-react';
import { checkBackendHealth } from '../api/apiClient';

interface NavbarProps {
  currentTab: string;
  setCurrentTab: (tab: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({ currentTab, setCurrentTab }) => {
  const [health, setHealth] = useState<'healthy' | 'offline' | 'checking'>('checking');

  useEffect(() => {
    let isMounted = true;
    const fetchHealth = async () => {
      const res = await checkBackendHealth();
      if (isMounted) {
        setHealth(res.status === 'healthy' ? 'healthy' : 'offline');
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const navItems = [
    { id: 'playground', label: 'Playground', icon: Terminal },
    { id: 'metrics', label: 'Observability', icon: Activity },
    { id: 'documents', label: 'Documents', icon: FileText },
    { id: 'architecture', label: 'Architecture', icon: Cpu },
  ];

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-200/80 bg-white/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand */}
          <div 
            className="flex items-center gap-3 cursor-pointer group"
            onClick={() => setCurrentTab('landing')}
          >
            <div className="w-9 h-9 rounded-xl bg-indigo-600 text-white flex items-center justify-center shadow-md shadow-indigo-500/20 group-hover:scale-105 transition-transform">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg tracking-tight text-slate-900">SentinelRAG</span>
                <span className="px-2 py-0.5 text-[10px] font-mono font-semibold bg-indigo-50 text-indigo-700 rounded-full border border-indigo-200">v1.0 RC</span>
              </div>
              <p className="text-[11px] text-slate-500 hidden sm:block">Self-Correcting RAG Transparency Platform</p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="flex items-center gap-1 sm:gap-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setCurrentTab(item.id)}
                  className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-500/30'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden md:inline">{item.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Health Status Indicator */}
          <div className="flex items-center gap-2">
            <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono border ${
              health === 'healthy'
                ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                : health === 'offline'
                ? 'bg-amber-50 text-amber-700 border-amber-200'
                : 'bg-slate-50 text-slate-600 border-slate-200'
            }`}>
              {health === 'healthy' ? (
                <>
                  <Wifi className="w-3.5 h-3.5 text-emerald-600 animate-pulse" />
                  <span className="hidden sm:inline">API Live</span>
                </>
              ) : health === 'offline' ? (
                <>
                  <WifiOff className="w-3.5 h-3.5 text-amber-600" />
                  <span className="hidden sm:inline">Offline (Mock)</span>
                </>
              ) : (
                <span className="animate-pulse">Checking...</span>
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
