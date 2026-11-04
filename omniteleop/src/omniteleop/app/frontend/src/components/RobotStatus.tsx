import React, { useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import {ObservationMessage} from "../types.ts";

interface RobotStatusProps {
  observations: ObservationMessage
}

const INITIAL_JOINTS: Record<string, number> = {
  "torso_j1": 0.0, "torso_j2": 0.0, "torso_j3": 0.0,
  "head_j1": 0.0, "head_j2": 0.0, "head_j3": 0.0,
  "L_arm_j1": 0.0, "L_arm_j2": 0.0, "L_arm_j3": 0.0, "L_arm_j4": 0.0, "L_arm_j5": 0.0, "L_arm_j6": 0.0, "L_arm_j7": 0.0,
  "R_arm_j1": 0.0, "R_arm_j2": 0.0, "R_arm_j3": 0.0, "R_arm_j4": 0.0, "R_arm_j5": 0.0, "R_arm_j6": 0.0, "R_arm_j7": 0.0,
  "L_wheel_j1": 0.0, "L_wheel_j2": 0.0, "R_wheel_j1": 0.0, "R_wheel_j2": 0.0,
};

const JOINT_MAPPING: Record<string, string[]> = {
  "left_arm": Array.from({ length: 7 }, (_, i) => `L_arm_j${i + 1}`),
  "right_arm": Array.from({ length: 7 }, (_, i) => `R_arm_j${i + 1}`),
  "head": Array.from({ length: 3 }, (_, i) => `head_j${i + 1}`),
  "torso": Array.from({ length: 3 }, (_, i) => `torso_j${i + 1}`),
};

export const RobotStatus: React.FC<RobotStatusProps> = ({observations}) => {
  const iframeRef = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    if (!observations || !iframeRef.current?.contentWindow) return;

    const jointsData = observations.joints || {};
    const animatedJoints: Record<string, number> = { ...INITIAL_JOINTS };

    for (const [part, mapping] of Object.entries(JOINT_MAPPING)) {
      if (jointsData[part]) {
        let vals = jointsData[part];
        if (typeof vals === 'object' && vals !== null && 'pos' in vals) {
          vals = (vals as any).pos;
        }
        if (Array.isArray(vals)) {
          mapping.forEach((jointName, i) => {
            if (i < vals.length) {
              animatedJoints[jointName] = vals[i];
            }
          });
        }
      }
    }

    iframeRef.current.contentWindow.postMessage({
      type: 'updateJoints',
      joints: animatedJoints
    }, '*');
  }, [observations]);

  return (
    <Card className="border-border shadow-md overflow-hidden">
      <CardHeader className="py-2.5 px-4 bg-gray-600 text-white space-y-0">
        <CardTitle className="text-xs font-bold tracking-tight text-white normal-case">Robot Status</CardTitle>
      </CardHeader>
      <CardContent className="p-0 bg-muted/30">
        <iframe
          ref={iframeRef}
          src="/assets/urdf_viewer.html"
          className="w-full h-[520px] border-none block"
          title="Robot Status URDF Viewer"
        />
      </CardContent>
    </Card>
  );
};