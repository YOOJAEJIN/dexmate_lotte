import React, { useCallback, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { ArrowUp, ArrowDown, RotateCcw, RotateCw, Hand, Grab, Handshake } from 'lucide-react';
import { SectionLabel } from './ui/section-label';

interface ChassisControlProps {
  apiUrl: string;
  disabled?: boolean;
}

type HandPose = 'open' | 'close';
type HandSide = 'left' | 'right' | 'both';

const HEARTBEAT_MS = 250;

const VELOCITY_DELTA_OPTIONS = [
  { label: '0.5×', value: 0.25 },
  { label: '0.75×', value: 0.375 },
  { label: '1×', value: 0.5 },
  { label: '1.5×', value: 0.75 },
  { label: '2×', value: 1.0 },
] as const;

export const ChassisControl: React.FC<ChassisControlProps> = ({ apiUrl, disabled }) => {
  const heartbeatRef = useRef<number | null>(null);
  const heldDirRef = useRef<string | null>(null);

  // Linear velocity (m/s) while a chassis button is held. 0.5 matches the
  // backend default (1x). Kept in a ref so heartbeats sent by an existing
  // interval closure always carry the latest selection.
  const [velocityDelta, setVelocityDelta] = React.useState(0.5);
  const velocityDeltaRef = useRef(velocityDelta);
  useEffect(() => {
    velocityDeltaRef.current = velocityDelta;
  }, [velocityDelta]);

  const sendCommand = useCallback(
    async (direction: string, pressed: boolean) => {
      try {
        const response = await fetch(`${apiUrl}/control/chassis`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ direction, pressed, speed: velocityDeltaRef.current }),
        });
        if (!response.ok) {
          console.error('Failed to control chassis');
        }
      } catch (error) {
        console.error('Error controlling chassis:', error);
      }
    },
    [apiUrl]
  );

  const stopHeartbeat = useCallback(() => {
    if (heartbeatRef.current !== null) {
      window.clearInterval(heartbeatRef.current);
      heartbeatRef.current = null;
    }
  }, []);

  const handleHandPose = async (pose: HandPose, side: HandSide) => {
    if (disabled) return;
    try {
      const response = await fetch(`${apiUrl}/control/hand`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ pose, side }),
      });
      if (!response.ok) {
        console.error(`Failed to ${pose} hand`);
      }
    } catch (error) {
      console.error(`Error ${pose}ing hand:`, error);
    }
  };

  const release = useCallback(() => {
    stopHeartbeat();
    const held = heldDirRef.current;
    heldDirRef.current = null;
    if (held) {
      void sendCommand(held, false);
    }
  }, [stopHeartbeat, sendCommand]);

  // Safety: stop heartbeat and release on unmount
  useEffect(() => {
    return () => release();
  }, [release]);

  const handlePointerDown = (direction: string) => (e: React.PointerEvent<HTMLButtonElement>) => {
    if (disabled) return;
    // Keep receiving pointerup even if the pointer leaves the button
    e.currentTarget.setPointerCapture(e.pointerId);
    release(); // in case a previous release event was missed
    heldDirRef.current = direction;
    void sendCommand(direction, true);
    heartbeatRef.current = window.setInterval(() => {
      void sendCommand(direction, true);
    }, HEARTBEAT_MS);
  };

  const buttonClass =
    'p-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 active:bg-primary/70 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm flex items-center justify-center select-none';
  const buttonStyle = { touchAction: 'none' } as const;

  return (
    <Card className="border-border shadow-md overflow-hidden">
      <CardHeader className="py-2.5 px-4 bg-orange-700 text-white">
        <CardTitle className="text-xs font-bold tracking-tight text-white normal-case">Chassis & Hand Control</CardTitle>
      </CardHeader>
      <CardContent className="p-4 flex flex-col items-center gap-3">
        <SectionLabel className="w-full self-stretch">Chassis Control</SectionLabel>

        {/* Speed selector: linear velocity (m/s) while held */}
        <select
          value={velocityDelta}
          onChange={(e) => setVelocityDelta(parseFloat(e.target.value))}
          disabled={disabled}
          className="w-1/2 self-start h-6 px-2 rounded border border-border bg-background text-[10px] font-medium focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50 disabled:cursor-not-allowed"
          title="Chassis speed while held"
        >
          {VELOCITY_DELTA_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              Δ {opt.label}
            </option>
          ))}
        </select>

        <div className="grid grid-cols-3 gap-2 w-fit">
          <div />
          <button
            onPointerDown={handlePointerDown('up')}
            onPointerUp={release}
            onPointerCancel={release}
            disabled={disabled}
            className={buttonClass}
            style={buttonStyle}
            title="Chassis Forward (hold)"
          >
            <ArrowUp size={24} />
          </button>
          <div />

          <button
            onPointerDown={handlePointerDown('rotate-left')}
            onPointerUp={release}
            onPointerCancel={release}
            disabled={disabled}
            className={buttonClass}
            style={buttonStyle}
            title="Rotate Left (hold)"
          >
            <RotateCcw size={24} />
          </button>

          <div />

          <button
            onPointerDown={handlePointerDown('rotate-right')}
            onPointerUp={release}
            onPointerCancel={release}
            disabled={disabled}
            className={buttonClass}
            style={buttonStyle}
            title="Rotate Right (hold)"
          >
            <RotateCw size={24} />
          </button>

          <div />
          <button
            onPointerDown={handlePointerDown('down')}
            onPointerUp={release}
            onPointerCancel={release}
            disabled={disabled}
            className={buttonClass}
            style={buttonStyle}
            title="Chassis Backward (hold)"
          >
            <ArrowDown size={24} />
          </button>
          <div />
        </div>

         <SectionLabel className="w-full self-stretch">Close Hand</SectionLabel>
         <div className="grid grid-cols-3 gap-2 w-full">
          <button
            onClick={() => void handleHandPose('close', 'left')}
            disabled={disabled}
            className={buttonClass}
            title="Close Hand - Left"
          >
            <Grab size={20} className="-scale-x-100" />
          </button>
          <button
            onClick={() => void handleHandPose('close', 'both')}
            disabled={disabled}
            className={buttonClass}
            title="Close Hand - Both"
          >
            <Handshake size={20} />
          </button>
          <button
            onClick={() => void handleHandPose('close', 'right')}
            disabled={disabled}
            className={buttonClass}
            title="Close Hand - Right"
          >
            <Grab size={20} />
          </button>
        </div>

         <SectionLabel className="w-full self-stretch">Open Hand</SectionLabel>
         <div className="grid grid-cols-3 gap-2 w-full">
          <button
            onClick={() => void handleHandPose('open', 'left')}
            disabled={disabled}
            className={buttonClass}
            title="Open Hand - Left"
          >
            <Hand size={20} className="-scale-x-100" />
          </button>
          <button
            onClick={() => void handleHandPose('open', 'both')}
            disabled={disabled}
            className={buttonClass}
            title="Open Hand - Both"
          >
            <Handshake size={20} />
          </button>
          <button
            onClick={() => void handleHandPose('open', 'right')}
            disabled={disabled}
            className={buttonClass}
            title="Open Hand - Right"
          >
            <Hand size={20} />
          </button>
        </div>

      </CardContent>
    </Card>
  );
};
