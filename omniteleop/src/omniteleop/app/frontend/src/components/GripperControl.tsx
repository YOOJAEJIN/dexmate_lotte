import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Hand, Grab } from 'lucide-react';
import { SectionLabel } from './ui/section-label';

interface GripperControlProps {
  apiUrl: string;
  disabled?: boolean;
}

type GripperPose = 'open' | 'close';
type GripperSide = 'left' | 'right';

export const GripperControl: React.FC<GripperControlProps> = ({ apiUrl, disabled }) => {
  const sendPose = async (pose: GripperPose, side: GripperSide) => {
    if (disabled) return;
    try {
      const response = await fetch(`${apiUrl}/control/hand`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pose, side }),
      });
      if (!response.ok) {
        console.error(`Failed to ${pose} ${side} gripper`);
      }
    } catch (error) {
      console.error(`Error controlling ${side} gripper:`, error);
    }
  };

  const buttonClass =
    'py-2.5 px-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 active:bg-primary/70 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm flex items-center justify-center gap-2 text-xs font-semibold select-none';

  const sideControls = (side: GripperSide) => (
    <>
      <SectionLabel className="w-full self-stretch capitalize">{side} Gripper</SectionLabel>
      <div className="grid grid-cols-2 gap-2 w-full flex-1 min-h-10">
        <button
          onClick={() => void sendPose('close', side)}
          disabled={disabled}
          className={buttonClass}
          title={`Close ${side} gripper`}
        >
          <Grab size={18} className={side === 'left' ? '-scale-x-100' : ''} />
          Close
        </button>
        <button
          onClick={() => void sendPose('open', side)}
          disabled={disabled}
          className={buttonClass}
          title={`Open ${side} gripper`}
        >
          <Hand size={18} className={side === 'left' ? '-scale-x-100' : ''} />
          Open
        </button>
      </div>
    </>
  );

  return (
    <Card className="border-border shadow-md overflow-hidden h-full flex flex-col">
      <CardHeader className="py-2.5 px-4 bg-orange-700 text-white">
        <CardTitle className="text-xs font-bold tracking-tight text-white normal-case">Gripper Control</CardTitle>
      </CardHeader>
      <CardContent className="p-3 flex-1 flex flex-col items-center gap-2">
        {sideControls('left')}
        {sideControls('right')}
      </CardContent>
    </Card>
  );
};
