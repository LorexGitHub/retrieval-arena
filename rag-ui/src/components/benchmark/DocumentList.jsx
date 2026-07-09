export function DocumentList({ documents, scores }) {
  if (documents.length === 0) {
    return <div className="text-text-faint text-xs">—</div>
  }

  return (
    <div className="space-y-0">
      {documents.map((doc, i) => (
        <div key={i} className="flex items-center gap-2.5 py-1.5 border-b border-border last:border-b-0">
          <div className="w-5 h-5 rounded-[6px] bg-surface flex items-center justify-center text-[0.6rem] font-semibold text-text-faint shrink-0">
            {i + 1}
          </div>
          <div className="flex-1 text-xs text-text truncate">{doc}</div>
          <div className="text-[0.68rem] font-medium text-text-sec tabular-nums shrink-0">{scores[i]?.toFixed(4)}</div>
        </div>
      ))}
    </div>
  )
}
