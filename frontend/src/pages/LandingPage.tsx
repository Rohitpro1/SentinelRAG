import React from 'react';
import { ShieldCheck, Cpu, Terminal, Sparkles, ArrowRight, CheckCircle2, Eye } from 'lucide-react';

interface LandingPageProps {
  onNavigate: (tab: string) => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onNavigate }) => {
  return (
    <div className="space-y-16 pb-16">
      {/* Hero Section */}
      <section className="relative pt-12 pb-16 text-center max-w-4xl mx-auto space-y-6">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200 shadow-sm">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Production-Grade Self-Correcting RAG Framework</span>
        </div>

        <h1 className="text-4xl sm:text-6xl font-extrabold text-slate-900 tracking-tight leading-tight">
          Transparent, Verifiable AI <br className="hidden sm:inline" />
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600">
            Powered by LangGraph
          </span>
        </h1>

        <p className="text-lg text-slate-600 max-w-2xl mx-auto leading-relaxed">
          SentinelRAG eliminates LLM hallucinations using adaptive state-graph execution, NLI-based contradiction detection, evidence grounding thresholds, and deterministic reasoning.
        </p>

        <div className="flex flex-wrap items-center justify-center gap-4 pt-4">
          <button
            onClick={() => onNavigate('playground')}
            className="flex items-center gap-2 px-6 py-3.5 rounded-xl bg-indigo-600 text-white font-semibold shadow-lg shadow-indigo-500/25 hover:bg-indigo-700 hover:scale-[1.02] active:scale-[0.98] transition-all"
          >
            <Terminal className="w-5 h-5" />
            <span>Launch Live Playground</span>
            <ArrowRight className="w-4 h-4" />
          </button>

          <button
            onClick={() => onNavigate('architecture')}
            className="flex items-center gap-2 px-6 py-3.5 rounded-xl bg-white text-slate-700 font-semibold border border-slate-200 shadow-sm hover:bg-slate-50 transition-all"
          >
            <Cpu className="w-5 h-5 text-indigo-600" />
            <span>Explore Architecture</span>
          </button>
        </div>
      </section>

      {/* Highlights / Features Grid */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-3">
          <div className="w-10 h-10 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center">
            <Cpu className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-slate-900">LangGraph Orchestration</h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            State-machine based query execution. Automatically triggers query rewriting and retrieval retries when initial evidence confidence falls below safety thresholds.
          </p>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-3">
          <div className="w-10 h-10 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center">
            <CheckCircle2 className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-slate-900">NLI Verification & Diagnostics</h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            Pairwise Natural Language Inference (NLI) flags semantic contradictions and calculates evidence coverage before an answer is formulated.
          </p>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-3">
          <div className="w-10 h-10 rounded-xl bg-purple-50 text-purple-600 flex items-center justify-center">
            <Eye className="w-5 h-5" />
          </div>
          <h3 className="text-base font-bold text-slate-900">Glass-Box Explainability</h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            Full observability into decision paths, confidence scores, evidence reliability ratings, latency breakdowns, and routing actions.
          </p>
        </div>
      </section>

      {/* Production Readiness Metrics Banner */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="bg-gradient-to-r from-slate-900 to-indigo-950 rounded-3xl p-8 text-white shadow-xl flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-indigo-400 text-xs font-mono font-semibold uppercase tracking-wider">
              <ShieldCheck className="w-4 h-4" />
              <span>Production Verification Metrics</span>
            </div>
            <h2 className="text-2xl font-bold">Tested & Hardened Engine</h2>
            <p className="text-xs text-slate-300 max-w-xl">
              SentinelRAG has passed rigorous automated testing: 353 pytest unit/integration tests, 97% code coverage, 100% mypy type checking, and zero Ruff violations.
            </p>
          </div>

          <div className="grid grid-cols-3 gap-4 text-center">
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-4 border border-white/10">
              <p className="text-2xl font-extrabold font-mono text-emerald-400">353</p>
              <p className="text-[11px] text-slate-300">Passing Tests</p>
            </div>
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-4 border border-white/10">
              <p className="text-2xl font-extrabold font-mono text-indigo-300">97%</p>
              <p className="text-[11px] text-slate-300">Code Coverage</p>
            </div>
            <div className="bg-white/10 backdrop-blur-md rounded-2xl p-4 border border-white/10">
              <p className="text-2xl font-extrabold font-mono text-purple-300">100%</p>
              <p className="text-[11px] text-slate-300">Type Safe</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
