import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full text-[0.6rem] font-semibold uppercase tracking-wider px-2 py-0.5 border",
  {
    variants: {
      variant: {
        default: "bg-surface text-text-sec border-border",
        accent: "bg-accent/10 text-accent border-accent/20",
        green: "bg-green/10 text-green border-green/20",
        red: "bg-red/10 text-red border-red/20",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />
}

export { Badge, badgeVariants }
