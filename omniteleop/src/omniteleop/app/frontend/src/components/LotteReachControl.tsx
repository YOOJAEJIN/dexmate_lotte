import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Crosshair, Loader2 } from 'lucide-react';

interface LotteReachControlProps {
  apiUrl: string;
  disabled?: boolean;
}

type ReachState =
  | { status: 'idle' }
  | { status: 'running' }
  | {
      status: 'done';
      frame: string;
      grasp: number[];
      size?: number[];
      layers: number;
      grav: string;
      boxes: number;
      elapsed: number;
    }
  | { status: 'error'; message: string; elapsed: number };

const fmt = (xyz: number[]): string =>
  xyz.map((v) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(3) : '—')).join(', ');

// Lotte Reach: press after the head angle or torso height changed. The backend
// asks HEAD SEG (vision/seg_overlay.py) to detect + segment again with fresh
// frames and returns the grasp point. Nothing moves.
export const LotteReachControl: React.FC<LotteReachControlProps> = ({ apiUrl }) => {
  const [state, setState] = useState<ReachState>({ status: 'idle' });

  const handleReach = async () => {
    if (state.status === 'running') return;
    setState({ status: 'running' });
    const t0 = performance.now();
    const elapsed = () => (performance.now() - t0) / 1000;
    try {
      const response = await fetch(`${apiUrl}/control/lotte_reach`, { method: 'POST' });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setState({ status: 'error', message: data.detail || `Request failed (${response.status})`, elapsed: elapsed() });
        return;
      }
      const box = Array.isArray(data.boxes) ? data.boxes[0] : undefined;
      if (!box) {
        setState({ status: 'error', message: 'No box found — is the pile in the head camera view?', elapsed: elapsed() });
        return;
      }
      setState({
        status: 'done',
        frame: data.frame === 'base' ? 'B' : 'C',
        grasp: box.grasp?.xyz ?? box.xyz_base ?? box.xyz_cam ?? [],
        size: box.size_m ?? undefined,
        layers: Array.isArray(box.layers) ? box.layers.length : 0,
        grav: data.grav ?? '-',
        boxes: data.boxes.length,
        elapsed: elapsed(),
      });
    } catch (error) {
      setState({ status: 'error', message: error instanceof Error ? error.message : 'Network error', elapsed: elapsed() });
    }
  };

  const busy = state.status === 'running';

  return (
    <Card className="border-border shadow-md overflow-hidden">
      <CardHeader className="py-2.5 px-4 bg-orange-700 text-white space-y-0">
        <CardTitle className="text-xs font-bold tracking-tight text-white normal-case">Lotte Reach (Grasp Point)</CardTitle>
      </CardHeader>
      <CardContent className="p-4 space-y-2">
        <button
          onClick={() => void handleReach()}
          disabled={busy}
          className="w-full flex items-center justify-center gap-2 p-2.5 rounded-lg text-white text-sm font-medium shadow-sm transition-colors bg-primary hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
          title="Detect + segment the box pile again with fresh frames and compute the grasp point (no arm motion)"
        >
          {busy ? <Loader2 size={18} className="animate-spin" /> : <Crosshair size={18} />}
          {busy ? 'Scanning…' : 'Reach'}
        </button>

        {state.status === 'done' && (
          <div className="text-xs text-emerald-700 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-950 border border-emerald-200 dark:border-emerald-900 rounded-lg px-3 py-2 leading-relaxed">
            <div className="font-semibold">
              GRASP {state.frame}({fmt(state.grasp)}) m
              <span className="font-normal opacity-60"> · {state.elapsed.toFixed(1)}s</span>
            </div>
            <div className="opacity-80">
              {state.size ? `box ${(state.size[0] * 100).toFixed(1)} × ${(state.size[1] * 100).toFixed(1)} cm · ` : ''}
              {state.layers} layers of 6.1 cm · {state.boxes} box{state.boxes > 1 ? 'es' : ''} · gravity {state.grav}
            </div>
            {state.frame === 'C' && (
              <div className="text-amber-700 dark:text-amber-400">
                camera frame — no live joints; Start teleop for base coordinates
              </div>
            )}
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
