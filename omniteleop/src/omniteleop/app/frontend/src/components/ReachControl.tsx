import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Crosshair, Loader2 } from 'lucide-react';

interface ReachControlProps {
  apiUrl: string;
  disabled?: boolean;
}

type ReachState =
  | { status: 'idle' }
  | { status: 'running' }
  | { status: 'done'; label: string; confidence: number; xyz: number[]; arm?: string; elapsed: number }
  | { status: 'error'; message: string; elapsed: number };

const formatXyz = (xyz: number[]): string =>
  xyz
    .map((v) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(2) : '—'))
    .join(', ');

export const ReachControl: React.FC<ReachControlProps> = ({ apiUrl, disabled }) => {
  const [state, setState] = useState<ReachState>({ status: 'idle' });

  const handleReach = async (detectOnly: boolean) => {
    if (disabled || state.status === 'running') return;
    setState({ status: 'running' });
    const t0 = performance.now();
    const elapsed = () => (performance.now() - t0) / 1000;
    try {
      const response = await fetch(`${apiUrl}/control/reach`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ detect_only: detectOnly }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setState({ status: 'error', message: data.detail || `Request failed (${response.status})`, elapsed: elapsed() });
        return;
      }
      if (!data.found) {
        setState({ status: 'error', message: 'No object detected in the camera center ROI', elapsed: elapsed() });
        return;
      }
      setState({
        status: 'done',
        label: data.label || 'object',
        confidence: Number(data.confidence) || 0,
        xyz: Array.isArray(data.xyz_base) ? data.xyz_base : [],
        arm: data.arm,
        elapsed: elapsed(),
      });
    } catch (error) {
      setState({ status: 'error', message: error instanceof Error ? error.message : 'Network error', elapsed: elapsed() });
    }
  };

  const busy = state.status === 'running';
  const buttonClass =
    'flex-1 flex items-center justify-center gap-2 p-2.5 rounded-lg text-white text-sm font-medium shadow-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed';

  return (
    <Card className="border-border shadow-md overflow-hidden">
      <CardHeader className="py-2.5 px-4 bg-orange-700 text-white space-y-0">
        <CardTitle className="text-xs font-bold tracking-tight text-white normal-case">Reach (Vision-Guided)</CardTitle>
      </CardHeader>
      <CardContent className="p-4 space-y-2">
        <div className="flex flex-row gap-2">
          <button
            onClick={() => handleReach(false)}
            disabled={disabled || busy}
            className={`${buttonClass} bg-primary hover:bg-primary/90`}
            title="Detect the object with the head camera and reach above it"
          >
            {busy ? <Loader2 size={18} className="animate-spin" /> : <Crosshair size={18} />}
            {busy ? 'Reaching…' : 'Reach'}
          </button>
          <button
            onClick={() => handleReach(true)}
            disabled={disabled || busy}
            className={`${buttonClass} bg-gray-500 hover:bg-gray-500/90`}
            title="Run detection only (no arm motion) — used to verify the camera transform"
          >
            Detect Only
          </button>
        </div>

        {state.status === 'done' && (
          <div className="text-xs text-emerald-700 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-950 border border-emerald-200 dark:border-emerald-900 rounded-lg px-3 py-2 leading-relaxed">
            <div className="font-semibold">
              Detected: {state.label} ({(state.confidence * 100).toFixed(0)}%)
              {state.arm ? ` → ${state.arm} arm` : ' (no motion)'}
              <span className="font-normal opacity-60"> · {state.elapsed.toFixed(1)}s</span>
            </div>
            <div className="font-mono opacity-80">base xyz: [{formatXyz(state.xyz)}] m</div>
          </div>
        )}
        {state.status === 'error' && (
          <div className="text-xs text-red-700 dark:text-red-400 bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-900 rounded-lg px-3 py-2">
            {state.message}
            <span className="opacity-60"> ({state.elapsed.toFixed(1)}s)</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
