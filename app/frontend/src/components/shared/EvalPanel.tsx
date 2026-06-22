"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { submitEvaluation } from "@/lib/api";

interface EvalPanelProps {
  title: string;
  domain: string;
}

interface Scores {
  fluency: number;
  relevance: number;
  conciseness: number;
}

function StarRating({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  const [hovered, setHovered] = useState(0);

  return (
    <div className="flex items-center justify-between py-2">
      <span className="text-sm text-muted-foreground w-28">{label}</span>
      <div className="flex gap-1">
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            type="button"
            onClick={() => onChange(star)}
            onMouseEnter={() => setHovered(star)}
            onMouseLeave={() => setHovered(0)}
            className="text-2xl leading-none transition-colors"
            aria-label={`${star} star`}
          >
            <span
              className={
                star <= (hovered || value)
                  ? "text-amber-400"
                  : "text-muted-foreground/30"
              }
            >
              ★
            </span>
          </button>
        ))}
      </div>
      <span className="text-sm font-medium w-6 text-right">
        {value > 0 ? value : "—"}
      </span>
    </div>
  );
}

export default function EvalPanel({ title, domain }: EvalPanelProps) {
  const [scores, setScores] = useState<Scores>({
    fluency: 0,
    relevance: 0,
    conciseness: 0,
  });
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const allRated = Object.values(scores).every((v) => v > 0);

  async function handleSubmit() {
    if (!allRated) return;
    setLoading(true);
    setError(null);
    try {
      await submitEvaluation({ title, domain, ...scores });
      setSubmitted(true);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  if (submitted) {
    return (
      <Card className="border-green-200 bg-green-50">
        <CardContent className="pt-6 text-center text-green-800">
          <p className="font-medium">Evaluation submitted — thank you!</p>
          <p className="text-sm mt-1 text-green-700">
            Scores saved to results/human_eval_scores.json
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Human Evaluation</CardTitle>
        <p className="text-sm text-muted-foreground">
          Rate the quality of this summary (1 = poor, 5 = excellent)
        </p>
      </CardHeader>
      <CardContent className="space-y-1">
        <div className="divide-y divide-border">
          <StarRating
            label="Fluency"
            value={scores.fluency}
            onChange={(v) => setScores((s) => ({ ...s, fluency: v }))}
          />
          <StarRating
            label="Relevance"
            value={scores.relevance}
            onChange={(v) => setScores((s) => ({ ...s, relevance: v }))}
          />
          <StarRating
            label="Conciseness"
            value={scores.conciseness}
            onChange={(v) => setScores((s) => ({ ...s, conciseness: v }))}
          />
        </div>

        {error && (
          <p className="text-sm text-red-600 pt-2">Error: {error}</p>
        )}

        <div className="pt-4">
          <Button
            onClick={handleSubmit}
            disabled={!allRated || loading}
            className="w-full"
          >
            {loading ? "Submitting..." : "Submit Evaluation"}
          </Button>
          {!allRated && (
            <p className="text-xs text-muted-foreground text-center mt-2">
              Rate all three criteria to submit
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}