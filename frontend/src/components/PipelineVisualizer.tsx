import React from 'react';
import { Cpu, Search, CheckCircle2, Scale, MessageSquareCode, ArrowRight, RefreshCw } from 'lucide-react';
import { DecisionAction } from '../types';

interface PipelineVisualizerProps {
  isExecuting: boolean;
  action?: DecisionAction;
  retryCount?: number;
}

export const PipelineVisualizer: React.FC<PipelineVisualizerProps> = ({
  isExecuting,
  action,
  retryCount = 0,
}) => {
  const steps = [
    { id: 'planner', label: 'Planner Node', icon: Cpu, desc: 'Query normalization & intent classification' },
    { id: 'retrieval', label: 'Retrieval Node', icon: Search, desc: 'Hybrid search, fusion & reranking' },
    { id: 'verification', label: 'Verification Node', icon: CheckCircle2, desc: 'NLI contradiction & coverage check' },
    { id: 'decision', label: 'Decision Node', icon: Scale, desc: 'Confidence thresholding & routing' },
    { id: 'response', label: 'Response Gen', icon: MessageSquareCode, desc: 'Grounded answer generation' },
  ];

  return (
    <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-indigo-600 animate-ping" />
          <h3 className="text-sm font-semibold text-slate-900">LangGraph Pipeline Execution</h3>
        </div>
        {retryCount > 0 && (
          <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono bg-amber-50 text-amber-700 border border-amber-200">
            <RefreshCw className="w-3 h-3 animate-spin" />
            <span>Retry Loop Pass: {retryCount}</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
        {steps.map((step, idx) => {
          const Icon = step.icon;
          const isDone = !isExecuting && action !== undefined;
          const isCurrentExecuting = isExecuting && idx < 4;

          return (
            <div key={step.id} className="relative flex flex-col justify-between p-3.5 rounded-xl border bg-slate-50/50 border-slate-200/80 transition-all hover:bg-slate-50">
              <div className="flex items-center justify-between mb-2">
                <div className={`w-7 h-7 rounded-lg flex items-center justify-center ${
                  isCurrentExecuting
                    ? 'bg-indigo-600 text-white animate-pulse'
                    : isDone
                    ? 'bg-emerald-100 text-emerald-700 border border-emerald-300'
                    : 'bg-slate-200 text-slate-600'
                }`}>
                  <Icon className="w-4 h-4" />
                </div>
                <span className="text-[10px] font-mono text-slate-400">0{idx + 1}</span>
              </div>

              <div>
                <p className="text-xs font-semibold text-slate-800">{step.label}</p>
                <p className="text-[11px] text-slate-500 mt-0.5 leading-tight">{step.desc}</p>
              </div>

              {idx < steps.length - 1 && (
                <div className="hidden sm:block absolute -right-2.5 top-1/2 -translate-y-1/2 z-10 text-slate-300">
                  <ArrowRight className="w-4 h-4" />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
