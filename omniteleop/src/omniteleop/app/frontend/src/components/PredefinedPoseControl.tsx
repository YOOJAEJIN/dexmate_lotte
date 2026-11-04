import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

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

  const poses = [
    { id: 'home', label: 'Home', image: 'home.png', description: 'Arm + Head + Torso' },
    { id: 'l_pose', label: 'L Pose', image: 'l_pose.png', description: 'Arm + Head + Torso' },
    { id: 'pantos_pick', label: 'Pantos Pick', image: 'pantos_pick.png', description: 'Arm + Head + Torso' },
    { id: 'pantos_place', label: 'Pantos Place', image: 'pantos_place.png', description: 'Arm + Head + Torso' },
    { id: 'object', label: 'Object', image: 'object.png', description: 'Head + Torso' },
    { id: 'libero', label: 'Libero', image: 'libero.png', description: 'Head + Torso' },
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
        </div>
      </CardContent>
    </Card>
  );
};
