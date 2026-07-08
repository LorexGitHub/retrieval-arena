import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select"
import type { QueryItem } from "@/types"
import { LLM_MODELS, EMBEDDING_MODELS } from "@/types"

interface RunPanelProps {
  queries: QueryItem[]
  selectedDatasets: string[]
  onRun: (queries: { query: string; ground_truth: string }[], llmModel: string, models: string[]) => void
  onClear: () => void
  progress: string | null
  progressPercent: number
  loading: boolean
  onCancel: () => void
}

export function RunPanel({ queries, selectedDatasets, onRun, onClear, progress, progressPercent, loading, onCancel }: RunPanelProps) {
  const [queryInput, setQueryInput] = useState("")
  const [gtInput, setGtInput] = useState("")
  const [selectedModels, setSelectedModels] = useState(["minilm-l12", "bge-small", "gte-small"])
  const [selectedSampleQueries, setSelectedSampleQueries] = useState<Set<number>>(new Set([0, 1, 2]))
  const [topK, setTopK] = useState(5)
  const [llmModel, setLlmModel] = useState("")

  const llmKeys = Object.keys(LLM_MODELS)

  const toggleSample = (idx: number) => {
    setSelectedSampleQueries((prev) => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }

  const handleEvaluate = () => {
    const sampleItems: { query: string; ground_truth: string }[] = []
    for (const idx of selectedSampleQueries) {
      if (queries[idx]) {
        sampleItems.push({ query: queries[idx].query, ground_truth: queries[idx].ground_truth })
      }
    }
    if (queryInput.trim()) {
      sampleItems.push({ query: queryInput.trim(), ground_truth: gtInput.trim() })
    }
    if (sampleItems.length === 0 || selectedModels.length === 0 || selectedDatasets.length === 0) return
    onRun(sampleItems, llmModel, selectedModels)
  }

  const handleClear = () => {
    setQueryInput("")
    setGtInput("")
    setSelectedSampleQueries(new Set())
    onClear()
  }

  return (
    <div className="bg-card border border-border rounded-[12px] p-5 shadow-[0_1px_8px_rgba(114,62,195,0.15),0_1px_3px_rgba(0,0,0,0.2)]">
      <div className="flex gap-3 mb-3">
        <div className="w-24">
          <label className="text-xs text-text-faint font-medium">Top-K</label>
          <select
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="mt-1 w-full h-[38px] rounded-[10px] border border-border bg-surface px-3 text-sm text-text caret-accent focus:outline-none focus:border-accent focus:ring-[3px] focus:ring-accent/15"
          >
            {[1, 3, 5, 10, 20].map((k) => (
              <option key={k} value={k}>{k}</option>
            ))}
          </select>
        </div>
        <div className="flex-1">
          <label className="text-xs text-text-faint font-medium">LLM (optional)</label>
          <Select value={llmModel} onValueChange={setLlmModel}>
            <SelectTrigger className="mt-1">
              <SelectValue placeholder="Template (no LLM)" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="">Template (no LLM)</SelectItem>
              {llmKeys.map((k) => (
                <SelectItem key={k} value={k}>
                  <span>{k}</span>
                  <span className="text-text-faint ml-2 text-[0.6rem]">({LLM_MODELS[k].size})</span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Sample queries — multi-select */}
      <div className="text-text-faint text-[0.68rem] font-medium mt-3 mb-1.5">Samples</div>
      <div className="grid grid-cols-3 gap-2">
        {queries.slice(0, 9).map((q, i) => {
          const checked = selectedSampleQueries.has(i)
          return (
            <Label
              key={i}
              className={`flex items-start gap-1.5 p-2 rounded-[8px] cursor-pointer border text-xs transition-all ${
                checked
                  ? "bg-accent/10 border-accent/30"
                  : "bg-surface border-border hover:border-border-h"
              }`}
            >
              <Checkbox
                checked={checked}
                onCheckedChange={() => toggleSample(i)}
                className="mt-0.5 shrink-0"
              />
              <div className="min-w-0">
                <div className={`truncate ${checked ? "text-accent" : "text-text-sec"}`}>{q.query.slice(0, 38)}</div>
                {q.relevant_dataset && (
                  <span className="text-[0.55rem] text-text-faint">{q.relevant_dataset}</span>
                )}
              </div>
            </Label>
          )
        })}
      </div>

      {/* Custom query */}
      <div className="mt-4 space-y-2">
        <input
          value={queryInput}
          onChange={(e) => setQueryInput(e.target.value)}
          placeholder="Custom query (runs alongside selected samples)"
          className="w-full h-[38px] rounded-[10px] border border-border bg-surface px-3 text-sm text-text placeholder:text-text-faint caret-accent focus:outline-none focus:border-accent focus:ring-[3px] focus:ring-accent/15"
        />
        <input
          value={gtInput}
          onChange={(e) => setGtInput(e.target.value)}
          placeholder="Expected answer (optional)"
          className="w-full h-[38px] rounded-[10px] border border-border bg-surface px-3 text-sm text-text placeholder:text-text-faint caret-accent focus:outline-none focus:border-accent focus:ring-[3px] focus:ring-accent/15"
        />
      </div>

      {/* Models */}
      <div className="mt-3">
        <label className="text-xs text-text-faint font-medium">Models</label>
        <div className="flex flex-wrap gap-2 mt-1">
          {["minilm-l12", "bge-small", "gte-small", "bge-base", "mpnet", "qwen3", "jina", "bge-large", "granite", "harrier"].map((key) => {
            const isChecked = selectedModels.includes(key)
            const m = EMBEDDING_MODELS[key]
            return (
              <Label
                key={key}
                className={`flex items-center gap-1 px-2.5 py-1 rounded-[8px] text-xs cursor-pointer border transition-all ${
                  isChecked
                    ? "bg-accent/10 border-accent/30 text-accent"
                    : "bg-surface border-border text-text-sec hover:border-border-h"
                }`}
              >
                <Checkbox
                  checked={isChecked}
                  onCheckedChange={(c) => {
                    if (c) setSelectedModels([...selectedModels, key])
                    else setSelectedModels(selectedModels.filter((k) => k !== key))
                  }}
                  className="data-[state=checked]:bg-accent data-[state=checked]:border-accent"
                />
                <span>{key}</span>
                {m && <span className="text-text-faint text-[0.55rem]">({m.size})</span>}
              </Label>
            )
          })}
        </div>
      </div>

      <div className="flex gap-3 mt-4">
        {loading ? (
          <Button variant="secondary" onClick={onCancel} className="flex-1">Cancel</Button>
        ) : (
          <>
            <Button variant="primary" onClick={handleEvaluate} className="flex-1">Evaluate</Button>
            <Button variant="secondary" onClick={handleClear} className="flex-1">Clear</Button>
          </>
        )}
      </div>

      {loading && progress && (
        <div className="mt-4 space-y-2">
          <Progress value={progressPercent} />
          <p className="text-xs text-text-faint text-center animate-pulse">{progress}</p>
        </div>
      )}
    </div>
  )
}
