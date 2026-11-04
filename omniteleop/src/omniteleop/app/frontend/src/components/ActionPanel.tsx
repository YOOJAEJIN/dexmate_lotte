import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import {ActionMessage} from "../types.ts";
import {formatTimestamp} from "../lib/utils.ts";

interface JointStatusPanelProps {
  title?: string
  actions?: ActionMessage
  accentColor?: string
}

export function ActionPanel({
  title = "Status Joint", 
  actions = {},
  accentColor = "bg-sky-700"
}: JointStatusPanelProps) {
  const parts = ['left_arm', 'right_arm', 'head', 'torso', 'left_hand', 'right_hand', 'chassis']
  const components = actions.components ?? {}

  return (
    <Card className="overflow-hidden border-border shadow-md">
      <CardHeader className={`py-2.5 px-4 ${accentColor} text-white flex flex-row items-center justify-between space-y-0`}>
        <CardTitle className="text-xs font-bold tracking-tight normal-case text-white">
          {title}
        </CardTitle>
        {actions.timestamp_ns && (
          <span className="text-[10px] font-mono text-white/80">
            {formatTimestamp(actions.timestamp_ns)}
          </span>
        )}
      </CardHeader>
      <CardContent className="p-3 bg-muted/30 dark:bg-zinc-900/50">
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-[11px] border-collapse table-fixed min-w-full">
            <thead>
              <tr className="bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100">
                <th className="p-1 px-2 text-left border border-zinc-200 dark:border-zinc-700 font-semibold" style={{ width: '14.28%' }}>Part</th>
                <th className="p-1 px-2 text-left border border-zinc-200 dark:border-zinc-700 font-semibold" style={{ width: '85.72%' }}>Values</th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-zinc-900/40 text-zinc-700 dark:text-zinc-300">
              {parts.map((part) => (
                <tr key={part} className="hover:bg-zinc-50 dark:hover:bg-white/5 transition-colors">
                  <td className="p-1 px-2 border border-zinc-200 dark:border-zinc-700 font-medium truncate capitalize">
                    {part.replace('_', ' ')}
                  </td>
                  <td className="p-1 px-2 border border-zinc-200 dark:border-zinc-700 font-mono whitespace-nowrap overflow-hidden text-ellipsis">
                    {components[part]
                      ? `[${(components[part].pos || components[part].vel || []).map(v => v.toFixed(3)).join(', ')}]`
                      : ''}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  )
}
