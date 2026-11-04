import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { SectionLabel } from './ui/section-label';
import { Minus, Plus, ArrowUp, ArrowDown, ArrowLeft, ArrowRight, Move } from 'lucide-react';
import { useHoldRepeat, holdPointerHandlers } from '../hooks/useHoldRepeat';

interface TorsoControlProps {
  apiUrl: string;
  disabled?: boolean;
}

export const TorsoControl: React.FC<TorsoControlProps> = ({ apiUrl, disabled }) => {
  const hold = useHoldRepeat(100);
  const [selectedHeadJoint, setSelectedHeadJoint] = React.useState(0);
  const [selectedTorsoJoint, setSelectedTorsoJoint] = React.useState(1);

  const handleHeadControl = async (jointIdx: number, delta: number) => {
    if (disabled) return;
    try {
      const response = await fetch(`${apiUrl}/control/head`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ joint_idx: jointIdx, delta }),
      });
      if (!response.ok) {
        console.error('Failed to control head');
      }
    } catch (error) {
      console.error('Error controlling head:', error);
    }
  };

  const handleTorsoControl = async (jointIdx: number, delta: number) => {
    if (disabled) return;
    try {
      const response = await fetch(`${apiUrl}/control/torso`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ joint_idx: jointIdx, delta }),
      });
      if (!response.ok) {
        console.error('Failed to control torso');
      }
    } catch (error) {
      console.error('Error controlling torso:', error);
    }
  };

  const handleTorsoAxisControl = async (axisIdx: number, delta: number) => {
    if (disabled) return;
    try {
      const response = await fetch(`${apiUrl}/control/torso`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ axis_idx: axisIdx, delta }),
      });
      if (!response.ok) {
        console.error('Failed to control torso axis');
      }
    } catch (error) {
      console.error('Error controlling torso axis:', error);
    }
  };

  const buttonClass =
    'p-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 active:bg-primary/70 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm flex items-center justify-center select-none';
  const buttonStyle = { touchAction: 'none' } as const;

  return (
    <Card className="border-border shadow-md overflow-hidden">
      <CardHeader className="py-2.5 px-4 bg-orange-700 text-white">
        <CardTitle className="text-xs font-bold tracking-tight text-white normal-case">Torso & Head Control</CardTitle>
      </CardHeader>
      <CardContent className="p-4 flex flex-col gap-4">
        <SectionLabel>Joint</SectionLabel>

        {/* Head Joint Control */}
        <div className="flex items-center gap-2">
          <button
            {...holdPointerHandlers(hold, () => void handleHeadControl(selectedHeadJoint, -0.01), disabled)}
            disabled={disabled}
            className={buttonClass}
            style={buttonStyle}
            title="Head Minus (hold)"
          >
            <Minus size={24} />
          </button>
          <select
            value={selectedHeadJoint}
            onChange={(e) => setSelectedHeadJoint(parseInt(e.target.value))}
            disabled={disabled}
            className="flex-1 h-10 px-2 rounded border border-border bg-background text-[10px] font-medium focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {[0, 1, 2].map((idx) => (
              <option key={idx} value={idx}>
                Head {idx + 1}
              </option>
            ))}
          </select>
          <button
            {...holdPointerHandlers(hold, () => void handleHeadControl(selectedHeadJoint, 0.01), disabled)}
            disabled={disabled}
            className={buttonClass}
            style={buttonStyle}
            title="Head Plus (hold)"
          >
            <Plus size={24} />
          </button>
        </div>

        {/* Torso Joint Control */}
        <div className="flex items-center gap-2">
          <button
            {...holdPointerHandlers(hold, () => void handleTorsoControl(selectedTorsoJoint, -0.01), disabled)}
            disabled={disabled}
            className={buttonClass}
            style={buttonStyle}
            title="Torso Minus (hold)"
          >
            <Minus size={24} />
          </button>
          <select
            value={selectedTorsoJoint}
            onChange={(e) => setSelectedTorsoJoint(parseInt(e.target.value))}
            disabled={disabled}
            className="flex-1 h-10 px-2 rounded border border-border bg-background text-[10px] font-medium focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {[0, 1, 2].map((idx) => (
              <option key={idx} value={idx}>
                Torso {idx + 1}
              </option>
            ))}
          </select>
          <button
            {...holdPointerHandlers(hold, () => void handleTorsoControl(selectedTorsoJoint, 0.01), disabled)}
            disabled={disabled}
            className={buttonClass}
            style={buttonStyle}
            title="Torso Plus (hold)"
          >
            <Plus size={24} />
          </button>
        </div>

        <SectionLabel>Cartesian</SectionLabel>

        {/* Torso Cartesian Control: front/back (X) and up/down (Z) */}
        <div className="grid grid-cols-3 gap-2">
          <div />
          <button
            {...holdPointerHandlers(hold, () => void handleTorsoAxisControl(2, 0.01), disabled)}
            disabled={disabled}
            className={`${buttonClass} flex-col gap-1`}
            style={buttonStyle}
            title="Torso Up (Z+) (hold)"
          >
            <ArrowUp size={20} />
            <span className="text-[10px] font-medium">Up</span>
          </button>
          <div />

          <button
            {...holdPointerHandlers(hold, () => void handleTorsoAxisControl(0, -0.01), disabled)}
            disabled={disabled}
            className={`${buttonClass} flex-col gap-1`}
            style={buttonStyle}
            title="Torso Back (X-) (hold)"
          >
            <ArrowLeft size={20} />
            <span className="text-[10px] font-medium">Back</span>
          </button>
          <div className="flex items-center justify-center">
            <Move size={20} className="text-muted-foreground/60" />
          </div>
          <button
            {...holdPointerHandlers(hold, () => void handleTorsoAxisControl(0, 0.01), disabled)}
            disabled={disabled}
            className={`${buttonClass} flex-col gap-1`}
            style={buttonStyle}
            title="Torso Front (X+) (hold)"
          >
            <ArrowRight size={20} />
            <span className="text-[10px] font-medium">Front</span>
          </button>

          <div />
          <button
            {...holdPointerHandlers(hold, () => void handleTorsoAxisControl(2, -0.01), disabled)}
            disabled={disabled}
            className={`${buttonClass} flex-col gap-1`}
            style={buttonStyle}
            title="Torso Down (Z-) (hold)"
          >
            <ArrowDown size={20} />
            <span className="text-[10px] font-medium">Down</span>
          </button>
          <div />
        </div>
      </CardContent>
    </Card>
  );
};
