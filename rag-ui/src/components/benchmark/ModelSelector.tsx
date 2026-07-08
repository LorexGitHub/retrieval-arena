import { MODEL_KEYS, EMBEDDING_MODELS } from "@/types"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"

interface ModelSelectorProps {
  selected: string[]
  onChange: (models: string[]) => void
}

export function ModelSelector({ selected, onChange }: ModelSelectorProps) {
  const handleToggle = (key: string, checked: boolean) => {
    if (checked) {
      onChange([...selected, key])
    } else {
      onChange(selected.filter((k) => k !== key))
    }
  }

  return (
    <div className="flex flex-wrap gap-2 mt-2">
      {MODEL_KEYS.map((key) => {
        const model = EMBEDDING_MODELS[key]
        const isChecked = selected.includes(key)
        return (
          <Label
            key={key}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-[8px] text-xs cursor-pointer transition-all border ${
              isChecked
                ? "bg-accent/10 border-accent/30 text-accent"
                : "bg-surface border-border text-text-sec hover:border-border-h"
            }`}
          >
            <Checkbox
              checked={isChecked}
              onCheckedChange={(c) => handleToggle(key, c === true)}
              className="data-[state=checked]:bg-accent data-[state=checked]:border-accent"
            />
            <span className="font-medium">{key}</span>
            <span className="text-text-faint text-[0.6rem]">({model.size})</span>
          </Label>
        )
      })}
    </div>
  )
}
