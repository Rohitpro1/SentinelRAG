import React, { useState } from 'react';
import { Search, Shield, Hash } from 'lucide-react';
import { DocumentChunk } from '../types';

export const DocumentsPage: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');

  const sampleChunks: DocumentChunk[] = [
    { chunk_id: 'c0', document_id: 'doc-1', text: 'refund policy detail number 0: Customers are eligible for full refund within 30 days of purchase.', token_count: 18, source_reliability_score: 0.95 },
    { chunk_id: 'c1', document_id: 'doc-1', text: 'refund policy detail number 1: Items must be returned in original packaging with proof of purchase.', token_count: 16, source_reliability_score: 0.95 },
    { chunk_id: 'c2', document_id: 'doc-1', text: 'refund policy detail number 2: Digital goods and downloadable content are non-refundable after initial download.', token_count: 17, source_reliability_score: 0.90 },
    { chunk_id: 'c3', document_id: 'doc-2', text: 'subscription cancellation detail: Recurring billing can be cancelled anytime via the user settings panel.', token_count: 15, source_reliability_score: 0.88 },
    { chunk_id: 'c4', document_id: 'doc-3', text: 'replacement policy detail: Damaged shipments are replaced free of charge upon receipt of photo verification.', token_count: 16, source_reliability_score: 0.92 },
  ];

  const filteredChunks = sampleChunks.filter(c =>
    c.text.toLowerCase().includes(searchTerm.toLowerCase()) ||
    c.document_id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="max-w-7xl mx-auto space-y-6 pb-12">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Document Index & Chunk Inspector</h1>
          <p className="text-xs text-slate-500">
            Browse ingested document chunks, vector embeddings, token counts, and source reliability ratings.
          </p>
        </div>

        {/* Search Input */}
        <div className="relative w-full sm:w-72">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search index chunks..."
            className="w-full pl-9 pr-4 py-2 rounded-xl border border-slate-200 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-600 transition-all"
          />
        </div>
      </div>

      {/* Chunks List */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
        <div className="p-4 bg-slate-50 border-b border-slate-200 flex items-center justify-between text-xs font-semibold text-slate-600">
          <span>Vector Index Document Chunks ({filteredChunks.length})</span>
          <span className="font-mono text-indigo-600">InMemory / Qdrant Hybrid Repo</span>
        </div>

        <div className="divide-y divide-slate-100">
          {filteredChunks.map((chunk) => (
            <div key={chunk.chunk_id} className="p-4 hover:bg-slate-50/80 transition-colors space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-indigo-50 text-indigo-700 border border-indigo-200">
                    ID: {chunk.chunk_id}
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-100 text-slate-600 border border-slate-200">
                    Doc: {chunk.document_id}
                  </span>
                </div>

                <div className="flex items-center gap-4 text-xs font-mono">
                  <div className="flex items-center gap-1 text-slate-500">
                    <Hash className="w-3.5 h-3.5" />
                    <span>{chunk.token_count} tokens</span>
                  </div>
                  <div className="flex items-center gap-1 text-emerald-700 font-bold">
                    <Shield className="w-3.5 h-3.5 text-emerald-600" />
                    <span>Reliability: {Math.round(chunk.source_reliability_score * 100)}%</span>
                  </div>
                </div>
              </div>

              <p className="text-xs text-slate-800 font-sans leading-relaxed bg-slate-50/50 p-2.5 rounded-lg border border-slate-200/60">
                {chunk.text}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
