import React from 'react';
import { DecisionAction, QueryResponseBody } from '../types';
import { CheckCircle, AlertTriangle, HelpCircle, ShieldAlert, Sparkles } from 'lucide-react';

interface AnswerCardProps {
  result: QueryResponseBody;
}

export const AnswerCard: React.FC<AnswerCardProps> = ({ result }) => {
  const getBadgeDetails = (act: DecisionAction) => {
    switch (act) {
      case 'PROCEED':
        return {
          label: 'PROCEED (High Grounding)',
          bg: 'bg-emerald-50 text-emerald-800 border-emerald-300',
          icon: CheckCircle,
        };
      case 'LOW_CONFIDENCE_RESPONSE':
        return {
          label: 'LOW CONFIDENCE (Caveat)',
          bg: 'bg-amber-50 text-amber-800 border-amber-300',
          icon: AlertTriangle,
        };
      case 'CLARIFY':
        return {
          label: 'CLARIFY (Insufficient Evidence)',
          bg: 'bg-sky-50 text-sky-800 border-sky-300',
          icon: HelpCircle,
        };
      case 'HUMAN_REVIEW':
        return {
          label: 'HUMAN REVIEW (Contradiction/Policy)',
          bg: 'bg-rose-50 text-rose-800 border-rose-300',
          icon: ShieldAlert,
        };
      default:
        return {
          label: act,
          bg: 'bg-slate-100 text-slate-800 border-slate-300',
          icon: Sparkles,
        };
    }
  };

  const badge = getBadgeDetails(result.action);
  const BadgeIcon = badge.icon;
  const confidencePercent = Math.round((result.confidence || 0) * 100);

  return (
    <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-md space-y-4">
      {/* Header with Action Badge and Confidence */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div className="flex items-center gap-2">
          <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${badge.bg}`}>
            <BadgeIcon className="w-3.5 h-3.5" />
            {badge.label}
          </span>
          {result.retry_count > 0 && (
            <span className="text-xs font-mono text-amber-600 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
              Retried {result.retry_count}x
            </span>
          )}
        </div>

        <div className="flex items-center gap-4 text-xs font-mono">
          <div>
            <span className="text-slate-400">Confidence: </span>
            <span className="font-bold text-slate-800">{confidencePercent}%</span>
          </div>
          <div>
            <span className="text-slate-400">Coverage: </span>
            <span className="font-bold text-slate-800">{Math.round(result.evidence_coverage * 100)}%</span>
          </div>
        </div>
      </div>

      {/* Answer Body */}
      <div>
        <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Generated Natural Language Response</h4>
        <div className="bg-slate-50 p-4 rounded-xl border border-slate-200/80 text-slate-900 font-sans text-sm leading-relaxed whitespace-pre-wrap">
          {result.answer || "No natural language answer was generated."}
        </div>
      </div>

      {/* Decision Reasons */}
      {result.reasons && result.reasons.length > 0 && (
        <div className="pt-2">
          <h5 className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5">Decision Engine Rationale</h5>
          <ul className="space-y-1">
            {result.reasons.map((reason, i) => (
              <li key={i} className="text-xs text-slate-600 flex items-start gap-1.5">
                <span className="text-indigo-500 font-bold">•</span>
                <span>{reason}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
