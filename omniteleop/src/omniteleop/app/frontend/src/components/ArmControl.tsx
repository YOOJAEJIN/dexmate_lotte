import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { SectionLabel } from './ui/section-label';
import { Minus, Plus } from 'lucide-react';
import { useHoldRepeat, holdPointerHandlers } from '../hooks/useHoldRepeat';

interface ArmControlProps {
  apiUrl: string;
  disabled?: boolean;
}

// Robot base frame axes (REP-103): X forward, Y left, Z up.
// Badge colors follow the RViz RGB = XYZ convention.
// 'swap' places the positive button on the left slot so the Y row reads
// Left ... Right instead of Right ... Left.
const CARTESIAN_AXES = [
  { idx: 0, label: 'X', badgeClass: 'bg-red-600', negative: 'Back', positive: 'Front', swap: false },
  { idx: 1, label: 'Y', badgeClass: 'bg-green-600', negative: 'Right', positive: 'Left', swap: true },
  { idx: 2, label: 'Z', badgeClass: 'bg-blue-600', negative: 'Down', positive: 'Up', swap: false },
] as const;

// Rotation jog: axes named by RPY convention. The rot_frame selector picks
// which frame the axis refers to - "world" (robot base) or "ee" (gripper's
// own frame, e.g. Z spins the gripper about its grasp axis).
const ROTATION_AXES = [
  { idx: 0, label: 'X', badgeClass: 'bg-red-600', negative: 'Roll -', positive: 'Roll +', swap: false },
  { idx: 1, label: 'Y', badgeClass: 'bg-green-600', negative: 'Pitch -', positive: 'Pitch +', swap: false },
  { idx: 2, label: 'Z', badgeClass: 'bg-blue-600', negative: 'Yaw -', positive: 'Yaw +', swap: false },
] as const;

// Delta magnitudes centered on the backend default (0.005 m per repeat).
const CARTESIAN_DELTA_OPTIONS = [
  { label: '0.5×', value: 0.0025 },
  { label: '0.75×', value: 0.00375 },
  { label: '1×', value: 0.005 },
  { label: '1.5×', value: 0.0075 },
  { label: '2×', value: 0.01 },
] as const;

// Rotation deltas in radians per repeat, labeled in degrees. At the 100 ms
// hold-repeat rate a 2.9° 1× tick sweeps ~29° per second of hold.
const ROTATION_DELTA_OPTIONS = [
  { label: '1.4°', value: 0.025 },
  { label: '2.1°', value: 0.0375 },
  { label: '2.9°', value: 0.05 },
  { label: '4.3°', value: 0.075 },
  { label: '5.7°', value: 0.1 },
] as const;

const CARTESIAN_MODES = [
  { id: 'pos', label: 'Pos', deltaDefault: 0.005 },
  { id: 'rot', label: 'Rot', deltaDefault: 0.05 },
] as const;

type CartesianMode = (typeof CARTESIAN_MODES)[number]['id'];

// Reference frame for rotation jogs: robot base (World) or gripper (EE).
const ROT_FRAMES = [
  { id: 'world', label: 'World' },
  { id: 'ee', label: 'EE' },
] as const;

type RotFrame = (typeof ROT_FRAMES)[number]['id'];

type JointSide = 'left' | 'right' | 'both';
type CartesianSide = 'left' | 'right' | 'both';

const CARTESIAN_SIDES = [
  { id: 'left', label: 'L Arm' },
  { id: 'right', label: 'R Arm' },
  { id: 'both', label: 'Both' },
] as const;

export const ArmControl: React.FC<ArmControlProps> = ({ apiUrl, disabled }) => {
  const hold = useHoldRepeat(100);

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

  const handleArmCartesianControl = async (side: CartesianSide, mode: CartesianMode, axisIdx: number, delta: number, frame: RotFrame) => {
    if (disabled) return;
    try {
      const response = await fetch(`${apiUrl}/control/arm/cartesian`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ side, mode, axis_idx: axisIdx, delta, rot_frame: frame }),
      });
      if (!response.ok) {
        console.error(`Failed to control ${side} arm(s) cartesian`);
      }
    } catch (error) {
      console.error(`Error controlling ${side} arm(s) cartesian:`, error);
    }
  };

  const [selectedLeftJoint, setSelectedLeftJoint] = React.useState(0);
  const [selectedRightJoint, setSelectedRightJoint] = React.useState(0);
  const [selectedBothJoint, setSelectedBothJoint] = React.useState(0);
  const [cartesianMode, setCartesianMode] = React.useState<CartesianMode>('pos');
  const [cartesianDelta, setCartesianDelta] = React.useState(0.005);
  const [cartesianSide, setCartesianSide] = React.useState<CartesianSide>('both');
  const [rotFrame, setRotFrame] = React.useState<RotFrame>('ee');
  const cartesianAxes = cartesianMode === 'pos' ? CARTESIAN_AXES : ROTATION_AXES;
  const cartesianDeltaOptions = cartesianMode === 'pos' ? CARTESIAN_DELTA_OPTIONS : ROTATION_DELTA_OPTIONS;
  const sideLabel = CARTESIAN_SIDES.find((s) => s.id === cartesianSide)?.label ?? 'Both Arms';

  const handleCartesianModeChange = (mode: CartesianMode) => {
    setCartesianMode(mode);
    // Meter and radian scales are unrelated - snap to the new mode's 1x default.
    setCartesianDelta(CARTESIAN_MODES.find((m) => m.id === mode)?.deltaDefault ?? 0.005);
  };

  const buttonStyle = { touchAction: 'none' } as const;

  return (
    <Card className="border-border shadow-md overflow-hidden">
      <CardHeader className="py-2.5 px-4 bg-orange-700 text-white space-y-0">
        <CardTitle className="text-xs font-bold tracking-tight text-white normal-case">Arm Control</CardTitle>
      </CardHeader>
      <CardContent className="p-4 flex flex-col gap-2">
        <SectionLabel>Joint</SectionLabel>

        {/* Left Arm Joint Control */}
        <div className="flex items-center gap-2">
          <button
            {...holdPointerHandlers(hold, () => void handleArmJointControl('left', selectedLeftJoint, -0.01), disabled)}
            disabled={disabled}
            className="p-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 active:bg-primary/70 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm flex items-center justify-center select-none"
            style={buttonStyle}
            title="Left Arm Minus (hold)"
          >
            <Minus size={24} />
          </button>
          <select
            value={selectedLeftJoint}
            onChange={(e) => setSelectedLeftJoint(parseInt(e.target.value))}
            disabled={disabled}
            className="flex-1 h-10 px-2 rounded border border-border bg-background text-[10px] font-medium focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {[0, 1, 2, 3, 4, 5, 6].map((idx) => (
              <option key={idx} value={idx}>
                L arm {idx+1}
              </option>
            ))}
          </select>
          <button
            {...holdPointerHandlers(hold, () => void handleArmJointControl('left', selectedLeftJoint, 0.01), disabled)}
            disabled={disabled}
            className="p-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 active:bg-primary/70 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm flex items-center justify-center select-none"
            style={buttonStyle}
            title="Left Arm Plus (hold)"
          >
            <Plus size={24} />
          </button>
        </div>

        {/* Right Arm Joint Control */}
        <div className="flex items-center gap-2">
          <button
            {...holdPointerHandlers(hold, () => void handleArmJointControl('right', selectedRightJoint, -0.01), disabled)}
            disabled={disabled}
            className="p-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 active:bg-primary/70 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm flex items-center justify-center select-none"
            style={buttonStyle}
            title="Right Arm Minus (hold)"
          >
            <Minus size={24} />
          </button>
          <select
            value={selectedRightJoint}
            onChange={(e) => setSelectedRightJoint(parseInt(e.target.value))}
            disabled={disabled}
            className="flex-1 h-10 px-2 rounded border border-border bg-background text-[10px] font-medium focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {[0, 1, 2, 3, 4, 5, 6].map((idx) => (
              <option key={idx} value={idx}>
                R arm {idx+1}
              </option>
            ))}
          </select>
          <button
            {...holdPointerHandlers(hold, () => void handleArmJointControl('right', selectedRightJoint, 0.01), disabled)}
            disabled={disabled}
            className="p-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 active:bg-primary/70 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm flex items-center justify-center select-none"
            style={buttonStyle}
            title="Right Arm Plus (hold)"
          >
            <Plus size={24} />
          </button>
        </div>

        {/* Both Arms Mirrored Joint Control */}
        <div className="flex items-center gap-2">
          <button
            {...holdPointerHandlers(hold, () => void handleArmJointControl('both', selectedBothJoint, -0.01), disabled)}
            disabled={disabled}
            className="p-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 active:bg-primary/70 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm flex items-center justify-center select-none"
            style={buttonStyle}
            title="Both Arms Mirrored Minus (hold)"
          >
            <Minus size={24} />
          </button>
          <select
            value={selectedBothJoint}
            onChange={(e) => setSelectedBothJoint(parseInt(e.target.value))}
            disabled={disabled}
            className="flex-1 h-10 px-2 rounded border border-border bg-background text-[10px] font-medium focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {[0, 1, 2, 3, 4, 5, 6].map((idx) => (
              <option key={idx} value={idx}>
                Both {idx+1}
              </option>
            ))}
          </select>
          <button
            {...holdPointerHandlers(hold, () => void handleArmJointControl('both', selectedBothJoint, 0.01), disabled)}
            disabled={disabled}
            className="p-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 active:bg-primary/70 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm flex items-center justify-center select-none"
            style={buttonStyle}
            title="Both Arms Mirrored Plus (hold)"
          >
            <Plus size={24} />
          </button>
        </div>

        <SectionLabel>Cartesian</SectionLabel>

        {/* Cartesian Control: mode/delta/side selectors + axis jog buttons */}
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <select
              value={cartesianMode}
              onChange={(e) => handleCartesianModeChange(e.target.value as CartesianMode)}
              disabled={disabled}
              className="flex-1 h-6 px-2 rounded border border-border bg-background text-[10px] font-medium focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50 disabled:cursor-not-allowed"
              title="Jog translation (Pos) or rotation (Rot)"
            >
              {CARTESIAN_MODES.map((mode) => (
                <option key={mode.id} value={mode.id}>
                  {mode.label}
                </option>
              ))}
            </select>
            <select
              value={cartesianDelta}
              onChange={(e) => setCartesianDelta(parseFloat(e.target.value))}
              disabled={disabled}
              className="flex-1 h-6 px-2 rounded border border-border bg-background text-[10px] font-medium focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50 disabled:cursor-not-allowed"
              title="Delta magnitude per repeat"
            >
              {cartesianDeltaOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  Δ {opt.label}
                </option>
              ))}
            </select>
            {cartesianMode === 'rot' && (
              <select
                value={rotFrame}
                onChange={(e) => setRotFrame(e.target.value as RotFrame)}
                disabled={disabled}
                className="flex-1 h-6 px-2 rounded border border-border bg-background text-[10px] font-medium focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50 disabled:cursor-not-allowed"
                title="Rotation reference frame: robot base (World) or gripper (EE)"
              >
                {ROT_FRAMES.map((frame) => (
                  <option key={frame.id} value={frame.id}>
                    {frame.label}
                  </option>
                ))}
              </select>
            )}
            <select
              value={cartesianSide}
              onChange={(e) => setCartesianSide(e.target.value as CartesianSide)}
              disabled={disabled}
              className="flex-1 h-6 px-2 rounded border border-border bg-background text-[10px] font-medium focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50 disabled:cursor-not-allowed"
              title="Arm(s) to jog"
            >
              {CARTESIAN_SIDES.map((side) => (
                <option key={side.id} value={side.id}>
                  {side.label}
                </option>
              ))}
            </select>
          </div>
          {cartesianAxes.map((axis) => {
            const jogButton = (sign: 1 | -1, iconFirst: boolean) => {
              const text = sign > 0 ? axis.positive : axis.negative;
              const Icon = sign > 0 ? <Plus size={16} /> : <Minus size={16} />;
              return (
                <button
                  {...holdPointerHandlers(hold, () => void handleArmCartesianControl(cartesianSide, cartesianMode, axis.idx, sign * cartesianDelta, rotFrame), disabled)}
                  disabled={disabled}
                  className="flex-1 h-10 px-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 active:bg-primary/70 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm flex items-center justify-center gap-1.5 select-none"
                  style={buttonStyle}
                  title={`${sideLabel} ${axis.label}${sign > 0 ? '+' : '-'} (${text}) (hold)`}
                >
                  {iconFirst ? Icon : null}
                  <span className="text-[10px] font-medium">{text}</span>
                  {iconFirst ? null : Icon}
                </button>
              );
            };
            return (
              <div key={axis.label} className="flex items-center gap-2">
                {axis.swap ? jogButton(1, true) : jogButton(-1, true)}
                <span
                  className={`w-6 h-6 shrink-0 rounded flex items-center justify-center text-[10px] font-bold text-white ${axis.badgeClass}`}
                  title={`${sideLabel} along ${axis.label} axis`}
                >
                  {axis.label}
                </span>
                {axis.swap ? jogButton(-1, false) : jogButton(1, false)}
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
};
