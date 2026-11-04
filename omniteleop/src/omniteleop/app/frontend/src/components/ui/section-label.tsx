import { cn } from '../../lib/utils'
import { type HTMLAttributes, forwardRef } from 'react'

const SectionLabel = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  ({ className, children, ...props }, ref) => (
    <div
      ref={ref}
      className={cn('flex items-center gap-2', className)}
      {...props}
    >
      <span className="text-[10px] font-bold tracking-tight whitespace-nowrap">{children}</span>
      <div className="h-px flex-1 bg-border" />
    </div>
  )
)
SectionLabel.displayName = 'SectionLabel'

export { SectionLabel }
