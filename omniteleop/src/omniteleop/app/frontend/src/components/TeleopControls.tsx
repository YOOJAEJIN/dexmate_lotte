import { useState } from 'react'
import { Play, Square, Circle, Trash2, RotateCcw, RefreshCw } from 'lucide-react'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './ui/card'
import { cn } from '../lib/utils'

export interface TeleopLaunchConfig {
  record_mode: boolean
  env: {
    ROBOT_NAME: string
    ROBOT_CONFIG: string
    ZENOH_CONFIG: string
  }
}

interface Props {
  teleopRunning: boolean
  isRecording: boolean
  episodeId: string | null
  elapsedTime: number | null
  softwareEstop: boolean
  loading: boolean
  onTeleopStart: (cfg: TeleopLaunchConfig) => void
  onTeleopStop: () => void
  onRecordStart: () => void
  onRecordSave: () => void
  onRecordDiscard: () => void
  onReset: () => void
  onResetAll: () => void
  onEstop: () => void
  onClearEstop: () => void
}

export function TeleopControls({
  teleopRunning,
  isRecording,
  episodeId,
  elapsedTime,
  softwareEstop,
  loading,
  onTeleopStart,
  onTeleopStop,
  onRecordStart,
  onRecordSave,
  onRecordDiscard,
  onReset,
  onResetAll,
  onEstop,
  onClearEstop,
}: Props) {
  const [cfg] = useState<TeleopLaunchConfig>({
    record_mode: false,
    env: { ROBOT_NAME: '', ROBOT_CONFIG: '', ZENOH_CONFIG: '' },
  })

  const canStartTeleop = !teleopRunning && !softwareEstop

  return (
    <Card className="border-border shadow-md overflow-hidden">
      <CardHeader className="py-2.5 px-4 bg-sky-700 text-white space-y-0">
        <CardTitle className="text-sm font-bold tracking-tight text-white normal-case text-center">Controls</CardTitle>
      </CardHeader>
      <CardContent className="px-5 py-2">

        {/* ---- Teleop start/stop ---- */}
        <div className="space-y-2">
          <p className="text-sm text-muted-foreground font-medium">Teleoperation</p>

          {/* Start / Stop / Reset / E-Stop row */}
          <div className="grid grid-cols-[3fr_3fr_2fr_2fr_3fr] gap-1">
            <Button
              onClick={() => onTeleopStart(cfg)}
              disabled={loading || !canStartTeleop}
              className={cn('w-full px-2 gap-1', canStartTeleop ? 'bg-emerald-600 hover:bg-emerald-700 text-white' : '')}
              variant={canStartTeleop ? 'default' : 'outline'}
            >
              <Play className="h-3.5 w-3.5 fill-current" />
              Start
            </Button>
            <Button
              onClick={onTeleopStop}
              disabled={loading || !teleopRunning}
              variant="outline"
              className={cn('w-full px-2 gap-1', canStartTeleop ? '' : 'bg-red-600 hover:bg-red-700 text-white hover:text-white')}
            >
              <Square className="h-3.5 w-3.5 fill-current" />
              Stop
            </Button>
            <Button
              onClick={onReset}
              disabled={loading || !teleopRunning}
              variant="outline"
              className="w-full px-1 gap-1 text-[10px] sm:text-xs"
              title="Reset"
            >
              <RotateCcw className="h-3.5 w-3.5 shrink-0" /> Reset
            </Button>
            <Button
              onClick={onResetAll}
              disabled={loading || !teleopRunning}
              variant="outline"
              className="w-full px-1 gap-1 text-[10px] sm:text-xs"
              title="Reset All"
            >
              <RefreshCw className="h-3.5 w-3.5 shrink-0" /> Reset All
            </Button>
            {/* E-Stop */}
            {!softwareEstop ? (
              <Button
                onClick={onEstop}
                disabled={loading}
                className="w-full px-1 gap-1 bg-red-600 hover:bg-red-700 text-white hover:text-white text-[10px] sm:text-[11px] font-bold"
                title="Emergency Stop"
              >
                <svg className="h-3 w-3 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/></svg>
                Emergency Stop
              </Button>
            ) : (
              <Button
                onClick={onClearEstop}
                disabled={loading}
                variant="outline"
                className="w-full px-1 gap-1 border-2 border-emerald-500 text-emerald-600 dark:text-emerald-400 hover:bg-emerald-50 dark:hover:bg-emerald-950 hover:text-emerald-600 dark:hover:text-emerald-400 text-[10px] sm:text-[11px] font-bold"
                title="Clear E-Stop"
              >
                <svg className="h-3 w-3 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/></svg>
                Clear E-Stop
              </Button>
            )}
          </div>
        </div>

        {/* ---- Recording ---- */}
        <div className="space-y-1.5 pt-1 border-t">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground font-medium">Recording</p>
            {!teleopRunning && (
              <span className="text-xs text-muted-foreground/60 italic">not enabled</span>
            )}
          </div>

          {isRecording && (
            <div className="flex items-center gap-2 px-2.5 py-1.5 bg-red-50 border border-red-200 rounded-lg">
              <span className="relative flex h-2 w-2 shrink-0">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500" />
              </span>
              <span className="text-sm text-red-700 font-medium">Recording</span>
              <span className="ml-auto font-mono text-xs text-red-600 truncate max-w-[180px]">episode id: {episodeId}</span>
              <span className="ml-auto font-mono text-sm text-red-600 truncate max-w-[180px]">elapsed time: {elapsedTime}</span>
            </div>
          )}

          <div className={cn('flex flex-wrap gap-2', !teleopRunning && 'opacity-50 pointer-events-none select-none')}>
            <Button
              onClick={onRecordStart}
              disabled={loading || isRecording || !teleopRunning}
              variant={teleopRunning && !isRecording ? 'default' : 'outline'}
              className={cn('gap-1.5', teleopRunning && !isRecording ? 'bg-emerald-600 hover:bg-emerald-700 text-white' : '')}
              size="lg"
            >
              <Circle className="h-3 w-3 fill-current" /> Start
            </Button>
            <Button
              onClick={onRecordSave}
              disabled={loading || !isRecording}
              variant={teleopRunning && isRecording ? 'default' : 'outline'}
              size="lg"
              className="gap-1.5"
            >
              <Square className="h-3 w-3" /> Save
            </Button>
            <Button
              onClick={onRecordDiscard}
              disabled={loading || !isRecording}
              variant={teleopRunning && isRecording ? 'secondary' : 'outline'}
              size="lg"
              className="gap-1.5 text-orange-600 border-orange-200 hover:bg-orange-50"
            >
              <Trash2 className="h-3 w-3" /> Discard
            </Button>
          </div>
        </div>

      </CardContent>
    </Card>
  )
}
