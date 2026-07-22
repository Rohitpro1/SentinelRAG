import React from 'react';
import { ShieldCheck, Github, ExternalLink } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-slate-200 bg-white py-8 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2 text-slate-600 text-sm">
          <ShieldCheck className="w-4 h-4 text-indigo-600" />
          <span className="font-semibold text-slate-900">SentinelRAG v1.0 RC</span>
          <span>— Production Self-Correcting RAG Platform</span>
        </div>
        <div className="flex items-center gap-6 text-xs text-slate-500 font-mono">
          <span>LangGraph + FastAPI + Qdrant</span>
          <span className="text-slate-300">•</span>
          <span className="flex items-center gap-1 hover:text-indigo-600 transition-colors">
            <Github className="w-3.5 h-3.5" />
            GitHub Repo
          </span>
          <span className="text-slate-300">•</span>
          <span className="flex items-center gap-1 hover:text-indigo-600 transition-colors">
            <ExternalLink className="w-3.5 h-3.5" />
            Docs & Specifications
          </span>
        </div>
      </div>
    </footer>
  );
};
