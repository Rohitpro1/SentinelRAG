import { HealthResponse, QueryRequestBody, QueryResponseBody } from '../types';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';
const ROOT_URL = BASE_URL.replace(/\/api\/v1\/?$/, '');

export async function checkBackendHealth(): Promise<HealthResponse> {
  try {
    const response = await fetch(`${ROOT_URL}/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) {
      throw new Error(`Health check failed with status ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.warn('Backend health check failed:', error);
    return {
      status: 'offline',
      service: 'SentinelRAG',
      version: '0.1.0',
    };
  }
}

export async function executeQuery(request: QueryRequestBody): Promise<QueryResponseBody> {
  const url = `${BASE_URL}/query`;
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer dev-token',
    },
    body: JSON.stringify({
      query: request.query,
      top_k: request.top_k ?? 20,
      rerank_top_n: request.rerank_top_n ?? 5,
      document_filter: request.document_filter ?? null,
    }),
  });

  if (!response.ok) {
    let errorDetail = 'Query execution failed.';
    try {
      const errorData = await response.json();
      errorDetail = errorData.detail || errorDetail;
    } catch {
      // Fallback if response is not JSON
    }
    throw new Error(errorDetail);
  }

  return await response.json();
}
