export type DecisionAction =
  | 'PROCEED'
  | 'LOW_CONFIDENCE_RESPONSE'
  | 'CLARIFY'
  | 'HUMAN_REVIEW'
  | 'RETRY_RETRIEVAL';

export interface QueryRequestBody {
  query: string;
  top_k?: number;
  rerank_top_n?: number;
  document_filter?: Record<string, unknown> | null;
}

export interface QueryResponseBody {
  action: DecisionAction;
  confidence: number;
  reasons: string[];
  retry_count: number;
  contradiction_detected: boolean;
  evidence_coverage: number;
  answer?: string | null;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}

export interface PipelineNodeStatus {
  id: 'planner' | 'retrieval' | 'verification' | 'decision' | 'response_generation';
  label: string;
  status: 'idle' | 'running' | 'completed' | 'failed' | 'bypassed';
  description: string;
}

export interface DocumentChunk {
  chunk_id: string;
  document_id: string;
  text: string;
  token_count: number;
  source_reliability_score: number;
}
