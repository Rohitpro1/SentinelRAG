import React from 'react';
import { Activity, ShieldCheck, AlertCircle, BarChart3, Zap, Clock } from 'lucide-react';

export const MetricsPage: React.FC = () => {
  const metrics = [
    { label: 'Total Queries Processed', value: '1,420', change: '+12%', icon: Activity, color: 'text-indigo-600', bg: 'bg-indigo-50' },
    { label: 'Grounding Pass Rate', value: '94.2%', change: '+3.1%', icon: ShieldCheck, color: 'text-emerald-600', bg: 'bg-emerald-50' },
    { label: 'Contradictions Blocked', value: '48', change: '100% Caught', icon: AlertCircle, color: 'text-rose-600', bg: 'bg-rose-50' },
    { label: 'Average Pipeline Latency', value: '4.8 ms', change: 'Sub-10ms', icon: Clock, color: 'text-purple-600', bg: 'bg-purple-50' },
  ];

  const decisionBreakdown = [
    { action: 'PROCEED', count: 1140, pct: '80.3%', color: 'bg-emerald-500' },
    { action: 'LOW_CONFIDENCE_RESPONSE', count: 180, pct: '12.7%', color: 'bg-amber-500' },
    { action: 'CLARIFY', count: 62, pct: '4.4%', color: 'bg-sky-500' },
    { action: 'HUMAN_REVIEW', count: 38, pct: '2.6%', color: 'bg-rose-500' },
  ];

  return (
    <div className="max-w-7xl mx-auto space-y-8 pb-12">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Observability & Diagnostics Dashboard</h1>
        <p className="text-xs text-slate-500">
          Telemetry metrics, NLI verification scores, and decision distribution across all queries.
        </p>
      </div>

      {/* Top Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((m, idx) => {
          const Icon = m.icon;
          return (
            <div key={idx} className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm space-y-3">
              <div className="flex items-center justify-between">
                <div className={`w-9 h-9 rounded-xl ${m.bg} ${m.color} flex items-center justify-center`}>
                  <Icon className="w-5 h-5" />
                </div>
                <span className="text-[11px] font-mono font-medium text-slate-400">{m.change}</span>
              </div>
              <div>
                <p className="text-xs font-medium text-slate-500">{m.label}</p>
                <p className="text-2xl font-extrabold font-mono text-slate-900 mt-1">{m.value}</p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Action Breakdown Chart Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-indigo-600" />
              Decision Action Distribution
            </h3>
            <span className="text-xs font-mono text-slate-400">Past 30 Days</span>
          </div>

          <div className="space-y-4">
            {decisionBreakdown.map((item, i) => (
              <div key={i} className="space-y-1.5">
                <div className="flex items-center justify-between text-xs font-medium">
                  <span className="text-slate-700 font-mono">{item.action}</span>
                  <span className="text-slate-500 font-mono">{item.count} ({item.pct})</span>
                </div>
                <div className="w-full h-3 bg-slate-100 rounded-full overflow-hidden">
                  <div className={`h-full ${item.color} rounded-full transition-all duration-500`} style={{ width: item.pct }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* System Health Overview */}
        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-4">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <Zap className="w-4 h-4 text-indigo-600" />
            Verification Engine Health
          </h3>

          <div className="space-y-3 text-xs">
            <div className="flex items-center justify-between p-3 rounded-xl bg-slate-50 border border-slate-200">
              <span className="text-slate-600">Deterministic NLI Verifier</span>
              <span className="font-mono text-emerald-600 font-bold">READY</span>
            </div>
            <div className="flex items-center justify-between p-3 rounded-xl bg-slate-50 border border-slate-200">
              <span className="text-slate-600">Cross-Encoder Reranker</span>
              <span className="font-mono text-emerald-600 font-bold">READY</span>
            </div>
            <div className="flex items-center justify-between p-3 rounded-xl bg-slate-50 border border-slate-200">
              <span className="text-slate-600">Qdrant Vector DB</span>
              <span className="font-mono text-emerald-600 font-bold">CONNECTED</span>
            </div>
            <div className="flex items-center justify-between p-3 rounded-xl bg-slate-50 border border-slate-200">
              <span className="text-slate-600">LangGraph Engine</span>
              <span className="font-mono text-emerald-600 font-bold">ACTIVE</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
