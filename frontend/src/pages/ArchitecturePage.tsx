import React from 'react';
import { Cpu, ArrowDown, ShieldCheck, Scale, MessageSquareCode } from 'lucide-react';

export const ArchitecturePage: React.FC = () => {
  const nodes = [
    {
      title: '1. Planner Node',
      desc: 'Normalizes incoming query text and performs deterministic query classification. On retry passes (retry_count > 0), generates stopword-stripped rewritten queries to improve recall.',
      badge: 'Retry Aware',
      icon: Cpu,
      color: 'text-indigo-600',
      bg: 'bg-indigo-50',
    },
    {
      title: '2. Retrieval Node',
      desc: 'Invokes RetrieverAgent to execute dense vector search, hybrid retrieval, fusion, and cross-encoder reranking over vector repository chunks.',
      badge: 'Hybrid Search & Rerank',
      icon: Cpu,
      color: 'text-sky-600',
      bg: 'bg-sky-50',
    },
    {
      title: '3. Verification Node',
      desc: 'Invokes VerificationAgent to run pairwise NLI contradiction detection, evaluate evidence coverage, and build verification diagnostics.',
      badge: 'NLI Contradiction Check',
      icon: ShieldCheck,
      color: 'text-emerald-600',
      bg: 'bg-emerald-50',
    },
    {
      title: '4. Decision Node & Routing',
      desc: 'Evaluates DecisionEngine thresholds. If confidence is insufficient and retry ceiling is not reached, conditional edge routes back to PlannerNode via RetryIncrementNode. Otherwise routes to ResponseGenerationNode.',
      badge: 'Conditional Graph Routing',
      icon: Scale,
      color: 'text-purple-600',
      bg: 'bg-purple-50',
    },
    {
      title: '5. Response Generation Node',
      desc: 'Invokes BaseResponseGenerator service to generate grounded natural-language answer (QueryResult.answer) matching the decision action (PROCEED, LOW_CONFIDENCE_RESPONSE, CLARIFY, HUMAN_REVIEW).',
      badge: 'Grounded Answer Generation',
      icon: MessageSquareCode,
      color: 'text-indigo-600',
      bg: 'bg-indigo-50',
    },
  ];

  return (
    <div className="max-w-4xl mx-auto space-y-8 pb-12">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">SentinelRAG Architecture & Graph Specification</h1>
        <p className="text-xs text-slate-500">
          Complete breakdown of LangGraph state machine execution, node contracts, and self-correction loops.
        </p>
      </div>

      <div className="space-y-4">
        {nodes.map((node, i) => {
          const Icon = node.icon;
          return (
            <div key={i} className="relative">
              <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`w-10 h-10 rounded-xl ${node.bg} ${node.color} flex items-center justify-center font-bold`}>
                      <Icon className="w-5 h-5" />
                    </div>
                    <h3 className="text-base font-bold text-slate-900">{node.title}</h3>
                  </div>

                  <span className="text-xs font-mono font-semibold px-2.5 py-1 rounded-full bg-slate-100 text-slate-700 border border-slate-200">
                    {node.badge}
                  </span>
                </div>

                <p className="text-xs text-slate-600 leading-relaxed font-sans pl-13">
                  {node.desc}
                </p>
              </div>

              {i < nodes.length - 1 && (
                <div className="flex justify-center my-2 text-slate-400">
                  <ArrowDown className="w-5 h-5 animate-bounce" />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
