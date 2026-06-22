export default function LoadingState() {
  return (
    <div className="flex flex-col items-center justify-center py-24 gap-4 text-muted-foreground">
      <div className="w-10 h-10 border-2 border-muted border-t-foreground rounded-full animate-spin" />
      <div className="text-sm text-center space-y-1">
        <p className="font-medium text-foreground">Processing your paper</p>
        <p>Classifying domain → routing to summarizer → extracting structure</p>
        <p className="text-xs">This takes 20–60 seconds depending on paper length.</p>
      </div>
    </div>
  );
}