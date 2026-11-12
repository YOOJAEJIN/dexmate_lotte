import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { SectionLabel } from './ui/section-label';
import { ChevronDown, ChevronUp, Maximize2, Minimize2, RotateCcw, RotateCw, type LucideIcon } from 'lucide-react';
import { useHoldRepeat, holdPointerHandlers } from '../hooks/useHoldRepeat';

interface ArmControlProps {
  apiUrl: string;
  disabled?: boolean;
}

type JointSide = 'left' | 'right' | 'both';

const JOINT_ROWS: { side: JointSide; label: string; title: string }[] = [
  { side: 'left', label: 'L Arm', title: 'Left Arm' },
  { side: 'right', label: 'R Arm', title: 'Right Arm' },
  { side: 'both', label: 'Both', title: 'Both Arms (mirrored)' },
];

const JOINT_STEP = 0.01; // rad per hold-repeat

// Joint-delta sign that moves the hand in the right-hand button's direction,
// per arm. From FK on vega_1p_gripper.urdf; identical at home, L-pose and an
// intermediate pose. Raw +/- is not intuitive: e.g. +J1 lowers the left hand
// but raises the right one. "both" uses the L sign - the backend mirrors the
// right arm via ARM_MIRROR_SIGNS, which yields the same motion on both sides.
type JointKind = 'updown' | 'inout' | 'twist';
const JOINT_DIRS: { kind: JointKind; L: 1 | -1; R: 1 | -1 }[] = [
  { kind: 'updown', L: -1, R: 1 },  // J1 shoulder pitch
  { kind: 'inout', L: 1, R: -1 },   // J2 shoulder roll (+ moves both hands to robot-left)
  { kind: 'twist', L: 1, R: 1 },    // J3 upper-arm twist
  { kind: 'updown', L: -1, R: -1 }, // J4 elbow
  { kind: 'twist', L: 1, R: 1 },    // J5 forearm twist
  { kind: 'updown', L: -1, R: 1 },  // J6 wrist pitch
  { kind: 'twist', L: 1, R: 1 },    // J7 wrist twist
];

const KIND_BUTTONS: Record<JointKind, [{ text: string; Icon: LucideIcon }, { text: string; Icon: LucideIcon }]> = {
  updown: [{ text: 'Down', Icon: ChevronDown }, { text: 'Up', Icon: ChevronUp }],
  inout: [{ text: 'In', Icon: Minimize2 }, { text: 'Out', Icon: Maximize2 }],
  twist: [{ text: 'Rot -', Icon: RotateCcw }, { text: 'Rot +', Icon: RotateCw }],
};

export const ArmControl: React.FC<ArmControlProps> = ({ apiUrl, disabled }) => {
  const hold = useHoldRepeat(100);
  const [joint, setJoint] = React.useState<Record<JointSide, number>>({ left: 0, right: 0, both: 0 });

  const handleArmJointControl = async (side: JointSide, jointIdx: number, delta: number) => {
    if (disabled) return;
    try {
      const response = await fetch(`${apiUrl}/control/arm/joint`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ side, joint_idx: jointIdx, delta }),
      });
      if (!response.ok) {
        console.error(`Failed to control ${side} arm(s) joint`);
      }
    } catch (error) {
      console.error(`Error controlling ${side} arm(s) joint:`, error);
    }
  };

  const buttonClass =
    'w-12 shrink-0 min-h-9 px-1 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 active:bg-primary/70 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm flex flex-col items-center justify-center leading-none select-none';
  const buttonStyle = { touchAction: 'none' } as const;

  return (
    <Card className="border-border shadow-md overflow-hidden h-full flex flex-col">
      <CardHeader className="py-2.5 px-4 bg-orange-700 text-white space-y-0">
        <CardTitle className="text-xs font-bold tracking-tight text-white normal-case">Arm Control</CardTitle>
      </CardHeader>
      <CardContent className="p-3 flex-1 flex flex-col gap-1.5">
        <SectionLabel>Joint</SectionLabel>

        {JOINT_ROWS.map(({ side, label, title }) => {
          const idx = joint[side];
          const dir = JOINT_DIRS[idx];
          const sign = side === 'right' ? dir.R : dir.L;
          const [neg, pos] = KIND_BUTTONS[dir.kind];
          return (
            <div key={side} className="flex-1 min-h-9 flex items-stretch gap-2">
              <button
                {...holdPointerHandlers(hold, () => void handleArmJointControl(side, idx, -sign * JOINT_STEP), disabled)}
                disabled={disabled}
                className={buttonClass}
                style={buttonStyle}
                title={`${title} J${idx + 1} ${neg.text} (hold)`}
              >
                <neg.Icon size={18} />
                <span className="text-[9px] font-medium">{neg.text}</span>
              </button>
              <div className="flex-1 min-w-0 flex items-stretch gap-1.5">
                <span className="w-10 shrink-0 self-center text-[11px] font-semibold text-foreground">{label}</span>
                <select
                  value={idx}
                  onChange={(e) => setJoint((j) => ({ ...j, [side]: parseInt(e.target.value) }))}
                  disabled={disabled}
                  className="flex-1 min-w-0 min-h-9 px-1.5 rounded border border-border bg-background text-[11px] font-medium focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50 disabled:cursor-not-allowed"
                  title={`${title} joint`}
                >
                  {[0, 1, 2, 3, 4, 5, 6].map((j) => (
                    <option key={j} value={j}>
                      J{j + 1}
                    </option>
                  ))}
                </select>
              </div>
              <button
                {...holdPointerHandlers(hold, () => void handleArmJointControl(side, idx, sign * JOINT_STEP), disabled)}
                disabled={disabled}
                className={buttonClass}
                style={buttonStyle}
                title={`${title} J${idx + 1} ${pos.text} (hold)`}
              >
                <pos.Icon size={18} />
                <span className="text-[9px] font-medium">{pos.text}</span>
              </button>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
};
