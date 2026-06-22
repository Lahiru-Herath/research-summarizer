"use client";

import { useState, useRef } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { summarizePDF, summarizeText, SummarizeResponse } from "@/lib/api";

interface UploadFormProps {
  onResult: (result: SummarizeResponse) => void;
  onLoading: (loading: boolean) => void;
  onError: (error: string | null) => void;
  loading: boolean;
}

export default function UploadForm({
  onResult,
  onLoading,
  onError,
  loading,
}: UploadFormProps) {
  const [pdfTitle, setPdfTitle] = useState("");
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [textTitle, setTextTitle] = useState("");
  const [bodyText, setBodyText] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  async function handlePDFSubmit() {
    if (!pdfFile || !pdfTitle.trim()) return;
    onLoading(true);
    onError(null);
    try {
      const result = await summarizePDF(pdfFile, pdfTitle.trim());
      onResult(result);
    } catch (e: any) {
      onError(e.message);
    } finally {
      onLoading(false);
    }
  }

  async function handleTextSubmit() {
    if (!bodyText.trim() || !textTitle.trim()) return;
    onLoading(true);
    onError(null);
    try {
      const result = await summarizeText(bodyText.trim(), textTitle.trim());
      onResult(result);
    } catch (e: any) {
      onError(e.message);
    } finally {
      onLoading(false);
    }
  }

  const inputClass =
    "w-full px-3 py-2 text-sm rounded-md border border-input bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring disabled:opacity-50";

  return (
    <Card>
      <CardContent className="pt-6">
        <Tabs defaultValue="pdf">
          <TabsList className="w-full mb-6">
            <TabsTrigger value="pdf" className="flex-1">
              Upload PDF
            </TabsTrigger>
            <TabsTrigger value="text" className="flex-1">
              Paste Text
            </TabsTrigger>
          </TabsList>

          <TabsContent value="pdf" className="space-y-4">
            <div className="space-y-1.5">
              <label className="text-sm font-medium">Paper title</label>
              <input
                className={inputClass}
                placeholder="e.g. Attention Is All You Need"
                value={pdfTitle}
                onChange={(e) => setPdfTitle(e.target.value)}
                disabled={loading}
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-sm font-medium">PDF file</label>
              <div
                onClick={() => fileRef.current?.click()}
                className={`flex flex-col items-center justify-center gap-2 border-2 border-dashed border-border rounded-lg p-8 cursor-pointer transition-colors hover:bg-muted/50 ${
                  loading ? "opacity-50 pointer-events-none" : ""
                }`}
              >
                <span className="text-2xl">📄</span>
                {pdfFile ? (
                  <p className="text-sm font-medium">{pdfFile.name}</p>
                ) : (
                  <>
                    <p className="text-sm font-medium">Click to upload PDF</p>
                    <p className="text-xs text-muted-foreground">
                      PDF files only
                    </p>
                  </>
                )}
              </div>
              <input
                ref={fileRef}
                type="file"
                accept=".pdf"
                className="hidden"
                onChange={(e) => setPdfFile(e.target.files?.[0] ?? null)}
                disabled={loading}
              />
            </div>

            <Button
              onClick={handlePDFSubmit}
              disabled={!pdfFile || !pdfTitle.trim() || loading}
              className="w-full"
            >
              {loading ? "Processing..." : "Summarize PDF"}
            </Button>
          </TabsContent>

          <TabsContent value="text" className="space-y-4">
            <div className="space-y-1.5">
              <label className="text-sm font-medium">Paper title</label>
              <input
                className={inputClass}
                placeholder="e.g. BERT: Pre-training of Deep Bidirectional Transformers"
                value={textTitle}
                onChange={(e) => setTextTitle(e.target.value)}
                disabled={loading}
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-sm font-medium">Paper text</label>
              <textarea
                className={`${inputClass} min-h-48 resize-y`}
                placeholder="Paste the full paper text or abstract here..."
                value={bodyText}
                onChange={(e) => setBodyText(e.target.value)}
                disabled={loading}
              />
            </div>

            <Button
              onClick={handleTextSubmit}
              disabled={!bodyText.trim() || !textTitle.trim() || loading}
              className="w-full"
            >
              {loading ? "Processing..." : "Summarize Text"}
            </Button>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}