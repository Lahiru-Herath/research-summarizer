interface DomainBadgeProps {
  domain: "General" | "Medical";
  confidence: number;
  lowConfidence: boolean;
}

export default function DomainBadge({
  domain,
  confidence,
  lowConfidence,
}: DomainBadgeProps) {
  const isGeneral = domain === "General";

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span
        className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
          isGeneral
            ? "bg-blue-100 text-blue-800"
            : "bg-teal-100 text-teal-800"
        }`}
      >
        {isGeneral ? "⚙ General" : "⚕ Medical"}
      </span>

      <span className="text-sm text-muted-foreground">
        {(confidence * 100).toFixed(1)}% confidence
      </span>

      {lowConfidence && (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-amber-100 text-amber-800">
          ⚠ Low confidence — verify domain manually
        </span>
      )}
    </div>
  );
}