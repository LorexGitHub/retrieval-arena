function metricColor(v) {
  if (v === null) return "text-text-faint"
  if (v >= 0.8) return "text-green"
  if (v >= 0.5) return "text-amber"
  return "text-red"
}

function MetricBox({ label, value }) {
  return (
    <div className="bg-metric-bg rounded-[8px] px-2.5 py-1.5 min-w-[60px] flex-1 text-center">
      <div className="text-[0.55rem] font-semibold uppercase tracking-wider text-text-faint">{label}</div>
      <div className="text-sm font-semibold tabular-nums">{value}</div>
    </div>
  )
}

export function MetricsGrid({ evaluation, group }) {
  const { hit_rate, mrr, precision, ndcg, exact_match, rouge_l_f1, semantic_similarity, faithfulness, answer_relevancy, llm_quality_score } = evaluation

  if (group === "retrieval") {
    return (
      <div className="flex gap-1.5 flex-wrap">
        <MetricBox label="Hit Rate" value={<span className={metricColor(hit_rate)}>{hit_rate.toFixed(3)}</span>} />
        <MetricBox label="MRR" value={<span className={metricColor(mrr)}>{mrr.toFixed(3)}</span>} />
        <MetricBox label="Precision" value={<span className={metricColor(precision)}>{precision.toFixed(3)}</span>} />
        <MetricBox label="NDCG" value={<span className={metricColor(ndcg)}>{ndcg.toFixed(3)}</span>} />
      </div>
    )
  }

  return (
    <div className="flex gap-1.5 flex-wrap">
      <MetricBox label="Exact" value={
        <span className={exact_match ? "text-green" : "text-red"}>{exact_match ? "✓" : "✗"}</span>
      } />
      <MetricBox label="ROUGE-L" value={<span className={metricColor(rouge_l_f1)}>{rouge_l_f1.toFixed(3)}</span>} />
      <MetricBox label="Semantic" value={<span className={metricColor(semantic_similarity)}>{semantic_similarity.toFixed(3)}</span>} />
      <MetricBox label="Faithful" value={
        <span className={metricColor(faithfulness)}>{faithfulness !== null ? faithfulness.toFixed(3) : "—"}</span>
      } />
      <MetricBox label="Relevancy" value={
        <span className={metricColor(answer_relevancy)}>{answer_relevancy !== null ? answer_relevancy.toFixed(3) : "—"}</span>
      } />
      <MetricBox label="LLM" value={
        <span className={metricColor(llm_quality_score)}>{llm_quality_score !== null ? llm_quality_score.toFixed(1) : "—"}</span>
      } />
    </div>
  )
}
