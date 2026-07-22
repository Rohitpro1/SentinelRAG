import React from 'react';
import { QueryResponseBody } from '../types';
import { ShieldCheck, AlertCircle, BarChart3, Clock, Database } from 'lucide-react';

interface DiagnosticsPanelProps {
  result: QueryResponseBody;
}

export const DiagnosticsPanel: React.FC<DiagnosticsPanelProps> = ({ result }) => {
  return (
    <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm space-y-4">
      <div className="flex items-center justify-between border-b border-slate-100 pb-3">
        <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-indigo-600" />
          Verification & Observability Diagnostics
        </h3>
        <span className="text-[11px] font-mono bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded border border-indigo-200">
          Unit 2.9 & 3.9 Diagnostics
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {/* Contradiction Flag */}
        <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/80">
          <div className="flex items-center gap-1.5 text-xs text-slate-500 mb-1">
            <AlertCircle className="w-3.5 h-3.5 text-amber-500" />
            <span>Contradiction</span>
          </div>
          <p className={`text-sm font-bold font-mono ${
            result.contradiction_detected ? 'text-rose-600' : 'text-emerald-600'
          }`}>
            {result.contradiction_detected ? 'FLAGGED' : 'NONE'}
          </p>
        </div>

        {/* Evidence Coverage */}
        <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/80">
          <div className="flex items-center gap-1.5 text-xs text-slate-500 mb-1">
            <ShieldCheck className="w-3.5 h-3.5 text-indigo-500" />
            <span>Coverage</span>
          </div>
          <p className="text-sm font-bold font-mono text-slate-900">
            {Math.round(result.evidence_coverage * 100)}%
          </p>
        </div>

        {/* Retry Count */}
        <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/80">
          <div className="flex items-center gap-1.5 text-xs text-slate-500 mb-1">
            <Database className="w-3.5 h-3.5 text-sky-500" />
            <span>Retry Passes</span>
          </div>
          <p className="text-sm font-bold font-mono text-slate-900">
            {result.retry_count}
          </p>
        </div>

        {/* Latency / Execution Mode */}
        <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/80">
          <div className="flex items-center gap-1.5 text-xs text-slate-500 mb-1">
            <Clock className="w-3.5 h-3.5 text-emerald-500" />
            <span>Execution</span>
          </div>
          <p className="text-xs font-semibold text-slate-800">
            LangGraph Sub-10ms
          </p>
        </div>
      </div>
    </div>
  );
};
