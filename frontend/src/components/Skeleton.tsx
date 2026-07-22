import React from 'react';

export const Skeleton: React.FC<{ className?: string }> = ({ className = 'h-4 bg-slate-200 rounded' }) => {
  return <div className={`animate-pulse ${className}`} />;
};

export const AnswerSkeleton: React.FC = () => {
  return (
    <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-100">
        <Skeleton className="h-6 w-36 bg-slate-200 rounded-full" />
        <Skeleton className="h-4 w-28 bg-slate-200 rounded" />
      </div>
      <div className="space-y-2">
        <Skeleton className="h-4 w-1/4 bg-slate-200" />
        <Skeleton className="h-16 w-full bg-slate-100 rounded-xl" />
      </div>
      <div className="space-y-2 pt-2">
        <Skeleton className="h-3 w-1/3 bg-slate-200" />
        <Skeleton className="h-3 w-3/4 bg-slate-200" />
      </div>
    </div>
  );
};
