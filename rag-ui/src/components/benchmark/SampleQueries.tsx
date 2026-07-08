import { Button } from "@/components/ui/button"
import type { QueryItem } from "@/types"

interface SampleQueriesProps {
  queries: QueryItem[]
  onSelect: (query: string, groundTruth: string) => void
}

export function SampleQueries({ queries, onSelect }: SampleQueriesProps) {
  const samples = queries.slice(0, 9)
  return (
    <>
      <div className="text-text-faint text-[0.68rem] font-medium mt-3 mb-1.5">Samples</div>
      <div className="grid grid-cols-3 gap-2">
        {samples.map((q, i) => (
          <div key={i} className="flex flex-col gap-0">
            <Button
              variant="secondary"
              size="sm"
              className="w-full justify-start truncate text-xs"
              onClick={() => onSelect(q.query, q.ground_truth)}
            >
              {q.query.slice(0, 38)}
            </Button>
            {q.relevant_dataset && (
              <span className="text-[0.6rem] text-text-faint px-1 mt-0.5">
                <span className="inline-block text-[0.6rem] font-medium px-[7px] py-[1px] rounded-[6px] bg-metric-bg text-text-faint border border-border">
                  {q.relevant_dataset}
                </span>
              </span>
            )}
          </div>
        ))}
      </div>
    </>
  )
}
