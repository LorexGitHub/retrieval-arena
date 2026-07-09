import * as React from "react"
import { cn } from "@/lib/utils"

const Textarea = React.forwardRef(({ className, ...props }, ref) => {
  return (
    <textarea
      className={cn(
        "flex min-h-[120px] w-full rounded-[10px] border border-border bg-surface px-3 py-2 text-sm text-text placeholder:text-text-faint caret-accent focus:outline-none focus:border-accent focus:ring-[3px] focus:ring-accent/15 disabled:cursor-not-allowed disabled:opacity-50",
        className
      )}
      ref={ref}
      {...props}
    />
  )
})
Textarea.displayName = "Textarea"

export { Textarea }
