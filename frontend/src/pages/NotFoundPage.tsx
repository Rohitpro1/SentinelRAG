import React from 'react';
import { ShieldAlert, ArrowLeft } from 'lucide-react';

export const NotFoundPage: React.FC<{ onHome: () => void }> = ({ onHome }) => {
  return (
    <div className="min-h-[500px] flex items-center justify-center p-6 text-center">
      <div className="bg-white rounded-2xl p-8 border border-slate-200 shadow-lg max-w-md space-y-4">
        <div className="w-12 h-12 rounded-2xl bg-amber-50 text-amber-600 flex items-center justify-center mx-auto">
          <ShieldAlert className="w-6 h-6" />
        </div>
        <h1 className="text-xl font-bold text-slate-900">404 — Page Not Found</h1>
        <p className="text-xs text-slate-500">
          The page or route you requested does not exist in SentinelRAG.
        </p>
        <button
          onClick={onHome}
          className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white text-xs font-semibold rounded-xl hover:bg-indigo-700 transition-all shadow-md shadow-indigo-500/20"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Return to Playground
        </button>
      </div>
    </div>
  );
};
