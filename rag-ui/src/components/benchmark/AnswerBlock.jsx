export function AnswerBlock({ answer }) {
  return (
    <div className="bg-surface rounded-[8px] p-3 text-sm text-text leading-relaxed">
      {answer || <span className="text-text-faint">—</span>}
    </div>
  )
}
