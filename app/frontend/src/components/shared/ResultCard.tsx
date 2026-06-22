import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import DomainBadge from "./DomainBadge";
import EvalPanel from "./EvalPanel";
import DownloadButton from "./DownloadButton";
import { SummarizeResponse } from "@/lib/api";

interface ResultCardProps {
  result: SummarizeResponse;
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
        {title}
      </h3>
      {children}
    </div>
  );
}

function BulletList({ items }: { items: string[] }) {
  return (
    <ul className="space-y-1.5">
      {items.map((item, i) => (
        <li key={i} className="flex gap-2 text-sm">
          <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-foreground/40 shrink-0" />
          <span>{item}</span>
        </li>
      ))}
    </ul>
  );
}

export default function ResultCard({ result }: ResultCardProps) {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <CardTitle className="text-lg leading-snug">{result.title}</CardTitle>
            <DomainBadge
              domain={result.domain}
              confidence={result.confidence}
              lowConfidence={result.low_confidence}
            />
          </div>
          <p className="text-xs text-muted-foreground pt-1">
            Model: {result.model_used}
          </p>
          <div className="pt-3">
            <DownloadButton result={result} />
          </div>
        </CardHeader>

        <CardContent className="space-y-6">
          <Section title="Summary">
            <p className="text-sm leading-relaxed">{result.summary}</p>
          </Section>

          <Separator />

          <Section title="Key Findings">
            <BulletList items={result.key_findings} />
          </Section>

          <Separator />

          <Section title="Methodology">
            <p className="text-sm leading-relaxed">{result.methodology}</p>
          </Section>

          <Separator />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Section title="Limitations">
              <BulletList items={result.limitations} />
            </Section>
            <Section title="Future Work">
              <BulletList items={result.future_work} />
            </Section>
          </div>

          <Separator />

          <Section title="Domain Probabilities">
            <div className="space-y-2">
              {(["General", "Medical"] as const).map((d) => (
                <div key={d} className="space-y-1">
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>{d}</span>
                    <span>
                      {(result.probabilities[d] * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        d === "General" ? "bg-blue-500" : "bg-teal-500"
                      }`}
                      style={{
                        width: `${(result.probabilities[d] * 100).toFixed(1)}%`,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </Section>
        </CardContent>
      </Card>

      <EvalPanel title={result.title} domain={result.domain} />
    </div>
  );
}