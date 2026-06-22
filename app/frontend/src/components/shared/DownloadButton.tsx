"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { SummarizeResponse } from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

interface DownloadButtonProps {
  result: SummarizeResponse;
}

export default function DownloadButton({ result }: DownloadButtonProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleDownload() {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/summary/pdf`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(result),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const safeTitle = result.title.slice(0, 60).replace(/\s+/g, "_");
      a.href = url;
      a.download = `${safeTitle}_summary.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-1">
      <Button
        variant="outline"
        onClick={handleDownload}
        disabled={loading}
        className="gap-2"
      >
        {loading ? (
          <>
            <span className="w-3.5 h-3.5 border border-current border-t-transparent rounded-full animate-spin" />
            Generating PDF...
          </>
        ) : (
          <>
            ↓ Download Summary PDF
          </>
        )}
      </Button>
      {error && (
        <p className="text-xs text-red-600">Error: {error}</p>
      )}
    </div>
  );
}