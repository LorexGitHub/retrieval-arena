import { Accordion, AccordionItem, AccordionTrigger, AccordionContent } from "@/components/ui/accordion"
import { Badge } from "@/components/ui/badge"
import { MetricsGrid } from "./MetricsGrid"
import { DocumentList } from "./DocumentList"
import { AnswerBlock } from "./AnswerBlock"
import type { RAGResult } from "@/types"
import { EMBEDDING_MODELS } from "@/types"

interface ResultCardProps {
  modelName: string
  result: RAGResult
  isBest: boolean
  topK: number
}

export function ResultCard({ modelName, result, isBest, topK }: ResultCardProps) {
  const ev = result.evaluation
  const ret = result.retrieval
  const gen = result.generation

  const rScore = (ev.hit_rate + ev.mrr + ev.precision + ev.ndcg) / 4
  const gVals = [
    ev.exact_match ? 1 : 0,
    ev.rouge_l_f1,
    ev.semantic_similarity,
    ev.answer_relevancy,
    ev.faithfulness ?? undefined,
    ev.llm_quality_score !== null ? ev.llm_quality_score / 5 : undefined,
  ].filter((v): v is number => v !== undefined)
  const gScore = gVals.reduce((a, b) => a + b, 0) / gVals.length

  const oScore = (rScore + gScore) / 2
  const sc = (v: number) => v >= 0.8 ? "#50D68A" : v >= 0.5 ? "#FFCF95" : "#D65050"

  return (
    <Accordion type="single" collapsible className="animate-fade-up">
      <AccordionItem value={modelName}>
        <AccordionTrigger>
          <span className="font-semibold text-sm text-text min-w-[120px]">{modelName}</span>
          {EMBEDDING_MODELS[modelName] && <span className="text-text-faint text-[0.55rem] ml-1">({EMBEDDING_MODELS[modelName].size})</span>}
          {isBest && <Badge variant="accent" className="text-[0.55rem] ml-1">Best</Badge>}
          <span className="flex-1" />
          <span className="flex items-center gap-3">
            <span className="flex items-center gap-1">
              <span className="text-[0.55rem] text-text-faint font-medium w-3">R</span>
              <span className="text-xs font-semibold tabular-nums w-[4ch] text-right" style={{ color: sc(rScore) }}>{rScore.toFixed(3)}</span>
            </span>
            <span className="flex items-center gap-1">
              <span className="text-[0.55rem] text-text-faint font-medium w-3">G</span>
              <span className="text-xs font-semibold tabular-nums w-[4ch] text-right" style={{ color: sc(gScore) }}>{gScore.toFixed(3)}</span>
            </span>
            <span className="flex items-center gap-1">
              <span className="text-[0.55rem] text-text-faint font-medium w-3">O</span>
              <span className="text-xs font-semibold tabular-nums w-[4ch] text-right" style={{ color: sc(oScore) }}>{oScore.toFixed(3)}</span>
            </span>
          </span>
        </AccordionTrigger>
        <AccordionContent>
          <h3 className="text-[0.7rem] font-semibold uppercase tracking-wider text-text-faint mt-2 mb-1.5">Retrieval</h3>
          <MetricsGrid evaluation={ev} group="retrieval" />
          <div className="h-3" />
          <h3 className="text-[0.7rem] font-semibold uppercase tracking-wider text-text-faint mb-1.5">Generation</h3>
          <MetricsGrid evaluation={ev} group="generation" />
          <div className="h-3" />
          <h3 className="text-[0.7rem] font-semibold uppercase tracking-wider text-text-faint mb-1.5">Documents ({Math.min(topK, ret.documents.length)})</h3>
          <DocumentList documents={ret.documents.slice(0, topK)} scores={ret.scores.slice(0, topK)} />
          <div className="h-3" />
          <h3 className="text-[0.7rem] font-semibold uppercase tracking-wider text-text-faint mb-1.5">Answer</h3>
          <AnswerBlock answer={gen.answer} />
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  )
}
