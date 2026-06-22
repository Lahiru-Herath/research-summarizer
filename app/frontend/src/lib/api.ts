const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export interface SummarizeResponse {
  title: string;
  domain: "General" | "Medical";
  confidence: number;
  low_confidence: boolean;
  probabilities: { General: number; Medical: number };
  model_used: string;
  summary: string;
  key_findings: string[];
  methodology: string;
  limitations: string[];
  future_work: string[];
}

export interface EvaluationRequest {
  title: string;
  domain: string;
  fluency: number;
  relevance: number;
  conciseness: number;
}

export async function summarizePDF(
  file: File,
  title: string
): Promise<SummarizeResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("title", title);

  const res = await fetch(`${API_BASE}/summarize/pdf`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

export async function summarizeText(
  text: string,
  title: string
): Promise<SummarizeResponse> {
  const res = await fetch(`${API_BASE}/summarize/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, title }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

export async function submitEvaluation(
  payload: EvaluationRequest
): Promise<void> {
  const res = await fetch(`${API_BASE}/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
}