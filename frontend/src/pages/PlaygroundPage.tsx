import React, { useState } from 'react';
import { executeQuery } from '../api/apiClient';
import { QueryResponseBody } from '../types';
import { PipelineVisualizer } from '../components/PipelineVisualizer';
import { AnswerCard } from '../components/AnswerCard';
import { DiagnosticsPanel } from '../components/DiagnosticsPanel';
import { AnswerSkeleton } from '../components/Skeleton';
import { Toast } from '../components/Toast';
import { Send, Sliders, Sparkles, RefreshCw } from 'lucide-react';

export const PlaygroundPage: React.FC = () => {
  const [queryText, setQueryText] = useState('refund policy detail number 2');
  const [topK, setTopK] = useState(20);
  const [rerankTopN, setRerankTopN] = useState(5);

  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<QueryResponseBody | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleExecute = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!queryText.trim()) return;

    setIsLoading(true);
    setErrorMessage(null);

    try {
      const res = await executeQuery({
        query: queryText.trim(),
        top_k: topK,
        rerank_top_n: rerankTopN,
      });
      setResult(res);
    } catch (err: any) {
      setErrorMessage(err?.message || 'Execution failed.');
      // Fallback mock response for standalone frontend testing if API is unreachable
      setResult({
        action: 'PROCEED',
        confidence: 0.95,
        reasons: ['Evidence confidence satisfied', 'No contradiction detected'],
        retry_count: 0,
        contradiction_detected: false,
        evidence_coverage: 1.0,
        answer: `Based on verified evidence: Refunds allowed within 30 days with original receipt. Full refund will be processed to original payment method.`,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const presetQueries = [
    'refund policy detail number 2',
    'what is the cancellation fee for subscriptions?',
    'how to request a replacement item [X]',
    'unanswerable policy inquiry',
  ];

  return (
    <div className="max-w-7xl mx-auto space-y-6 pb-12">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Query Playground</h1>
        <p className="text-xs text-slate-500">
          Execute queries against SentinelRAG's self-correcting state graph and inspect real-time verification outputs.
        </p>
      </div>

      {/* Main Grid: Query Controls + Visualizer */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Query Form & Preset Options */}
        <div className="lg:col-span-1 space-y-4">
          <form onSubmit={handleExecute} className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                Query Prompt
              </label>
              <textarea
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                rows={3}
                placeholder="Enter query text..."
                className="w-full p-3 rounded-xl border border-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-600 transition-all resize-none"
              />
            </div>

            {/* Hyperparameter Controls */}
            <div className="pt-2 border-t border-slate-100 space-y-3">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-700">
                <Sliders className="w-3.5 h-3.5 text-indigo-600" />
                <span>Search Hyperparameters</span>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-[11px] text-slate-500 mb-1">Top-K Candidates ({topK})</label>
                  <input
                    type="range"
                    min={1}
                    max={50}
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                    className="w-full accent-indigo-600 cursor-pointer"
                  />
                </div>

                <div>
                  <label className="block text-[11px] text-slate-500 mb-1">Rerank Top-N ({rerankTopN})</label>
                  <input
                    type="range"
                    min={1}
                    max={20}
                    value={rerankTopN}
                    onChange={(e) => setRerankTopN(Number(e.target.value))}
                    className="w-full accent-indigo-600 cursor-pointer"
                  />
                </div>
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading || !queryText.trim()}
              className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-indigo-600 text-white font-semibold text-xs hover:bg-indigo-700 disabled:opacity-50 transition-all shadow-md shadow-indigo-500/20"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Executing Pipeline...</span>
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  <span>Execute Query</span>
                </>
              )}
            </button>
          </form>

          {/* Preset Prompts */}
          <div className="bg-white rounded-2xl p-4 border border-slate-200 shadow-sm space-y-2">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">Sample Test Scenarios</span>
            <div className="space-y-1.5">
              {presetQueries.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => setQueryText(q)}
                  className="w-full text-left text-xs p-2 rounded-lg bg-slate-50 hover:bg-indigo-50 hover:text-indigo-700 text-slate-700 transition-colors truncate block"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Execution Visualization & Results */}
        <div className="lg:col-span-2 space-y-6">
          <PipelineVisualizer
            isExecuting={isLoading}
            action={result?.action}
            retryCount={result?.retry_count}
          />

          {isLoading ? (
            <AnswerSkeleton />
          ) : result ? (
            <div className="space-y-6">
              <AnswerCard result={result} />
              <DiagnosticsPanel result={result} />
            </div>
          ) : (
            <div className="bg-white rounded-2xl p-12 border border-slate-200 text-center space-y-3">
              <div className="w-12 h-12 rounded-2xl bg-indigo-50 text-indigo-600 flex items-center justify-center mx-auto">
                <Sparkles className="w-6 h-6" />
              </div>
              <h3 className="text-base font-bold text-slate-900">Ready to Query</h3>
              <p className="text-xs text-slate-500 max-w-sm mx-auto">
                Select a sample scenario or enter your own query on the left to see SentinelRAG evaluate grounded evidence.
              </p>
            </div>
          )}
        </div>
      </div>

      {errorMessage && (
        <Toast
          message={`Backend API note: ${errorMessage} (Showing fallback result)`}
          type="info"
          onClose={() => setErrorMessage(null)}
        />
      )}
    </div>
  );
};
