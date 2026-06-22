"use client";

import { useState } from "react";
import UploadForm from "@/components/shared/UploadForm";
import ResultCard from "@/components/shared/ResultCard";
import LoadingState from "@/components/shared/LoadingState";
import { SummarizeResponse } from "@/lib/api";

export default function Home() {
  const [result, setResult] = useState<SummarizeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleResult(r: SummarizeResponse) {
    setResult(r);
    setTimeout(() => {
      document.getElementById("result-section")?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }, 100);
  }

  return (
    <main className="min-h-screen">
      <div className="max-w-3xl mx-auto px-4 py-12 space-y-10">

        <div className="space-y-2">
          <h1 className="text-2xl font-semibold tracking-tight">
            Research Paper Summarizer
          </h1>
          <p className="text-sm text-muted-foreground">
            Domain-adaptive summarization using fine-tuned BART (General) and
            PEGASUS (Medical) with BERT-based routing.
          </p>
        </div>

        <UploadForm
          onResult={handleResult}
          onLoading={setLoading}
          onError={setError}
          loading={loading}
        />

        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            <span className="font-medium">Error:</span> {error}
          </div>
        )}

        {loading && <LoadingState />}

        {result && !loading && (
          <div id="result-section">
            <ResultCard result={result} />
          </div>
        )}

      </div>
    </main>
  );
}