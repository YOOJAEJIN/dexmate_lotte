import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { RotateCcw, RotateCw, Loader2 } from 'lucide-react';

interface PredefinedPoseControlProps {
  apiUrl: string;
  disabled?: boolean;
}

export const PredefinedPoseControl: React.FC<PredefinedPoseControlProps> = ({ apiUrl, disabled }) => {
  const handlePose = async (pose: string) => {
    if (disabled) return;
    try {
      const response = await fetch(`${apiUrl}/control/predefined`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ pose }),
      });
      if (!response.ok) {
        console.error('Failed to set predefined pose');
      }
    } catch (error) {
      console.error('Error setting predefined pose:', error);
    }
  };

  // Base turn in place (backend closes the loop on the head IMU, ~10 s per 90 deg).
  const [turning, setTurning] = React.useState(false);
  const [turnMsg, setTurnMsg] = React.useState<{ ok: boolean; text: string } | null>(null);
  const handleTurn = async (deg: number) => {
    if (disabled || turning) return;
    setTurning(true);
    setTurnMsg(null);
    try {
      const response = await fetch(`${apiUrl}/control/chassis_turn`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ deg }),
      });
      const data = await response.json().catch(() => ({}));
      setTurnMsg(response.ok
        ? { ok: true, text: `turned ${data.turned_deg}° (target ${deg}°, ${data.elapsed_s}s)` }
        : { ok: false, text: data.detail || `turn failed (${response.status})` });
    } catch (error) {
      setTurnMsg({ ok: false, text: error instanceof Error ? error.message : 'network error' });
    } finally {
      setTurning(false);
    }
  };
  const turns = [
    { deg: 90, label: 'Turn L 90°', Icon: RotateCcw },
    { deg: -90, label: 'Turn R 90°', Icon: RotateCw },
  ];

  const poses = [
    { id: 'home', label: 'Home', image: 'home.png', description: 'Arm + Head + Torso' },
    { id: 'l_pose', label: 'L Pose', image: 'l_pose.png', description: 'Arm + Head + Torso' },
  ];

  return (
    <Card className="border-border shadow-md overflow-hidden">
      <CardHeader className="py-2.5 px-4 bg-orange-700 text-white space-y-0">
        <CardTitle className="text-xs font-bold tracking-tight text-white normal-case">Predefined Pose</CardTitle>
      </CardHeader>
      <CardContent className="p-4">
        <div className="grid grid-cols-3 gap-2">
          {poses.map((pose) => (
            <button
              key={pose.id}
              onClick={() => handlePose(pose.id)}
              disabled={disabled}
              className="flex flex-col sm:flex-row items-center justify-center sm:justify-start gap-2 sm:gap-3 p-2.5 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm text-sm font-medium"
              title={pose.description}
            >
              <div className="shrink-0 w-14 h-14 flex items-center justify-center">
                <img src={`/assets/poses/${pose.image}`} alt={pose.label} className="w-full h-full object-contain" />
              </div>
              <div className="flex flex-col items-center sm:items-start leading-tight">
                <span className="text-xs sm:text-sm">{pose.label}</span>
                <span className="text-[9px] sm:text-[10px] opacity-70 font-normal hidden sm:block">{pose.description}</span>
              </div>
            </button>
          ))}
          {turns.map(({ deg, label, Icon }) => (
            <button
              key={deg}
              onClick={() => void handleTurn(deg)}
              disabled={disabled || turning}
              className="flex flex-col sm:flex-row items-center justify-center sm:justify-start gap-2 sm:gap-3 p-2.5 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-sm text-sm font-medium"
              title="Turn the base in place, slowly (head IMU closed loop). E-Stop stops it."
            >
              <div className="shrink-0 w-14 h-14 flex items-center justify-center">
                {turning ? <Loader2 size={30} className="animate-spin" /> : <Icon size={30} />}
              </div>
              <div className="flex flex-col items-center sm:items-start leading-tight">
                <span className="text-xs sm:text-sm">{label}</span>
                <span className="text-[9px] sm:text-[10px] opacity-70 font-normal hidden sm:block">Base, ~10 s</span>
              </div>
            </button>
          ))}
        </div>
        {turnMsg && (
          <div className={`mt-2 text-xs rounded-lg px-3 py-2 border ${turnMsg.ok
            ? 'text-emerald-700 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-950 border-emerald-200 dark:border-emerald-900'
            : 'text-red-700 dark:text-red-400 bg-red-50 dark:bg-red-950 border-red-200 dark:border-red-900'}`}>
            {turnMsg.text}
          </div>
        )}
      </CardContent>
    </Card>
  );
};
