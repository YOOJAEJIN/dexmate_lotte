import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { ChevronUp, ChevronDown, ChevronLeft, ChevronRight } from 'lucide-react';
import { useHoldRepeat, holdPointerHandlers } from '../hooks/useHoldRepeat';

interface HeadControlProps {
  apiUrl: string;
}

export const HeadControl: React.FC<HeadControlProps> = ({ apiUrl }) => {
  const hold = useHoldRepeat(150);

  const handleControl = async (direction: string) => {
    try {
      const response = await fetch(`${apiUrl}/control/head`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ direction }),
      });
      if (!response.ok) {
        console.error('Failed to control head');
      }
    } catch (error) {
      console.error('Error controlling head:', error);
    }
  };

  const buttonClass =
    'p-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 active:bg-primary/70 transition-colors shadow-sm flex items-center justify-center select-none';
  const buttonStyle = { touchAction: 'none' } as const;

  return (
    <Card className="border-border shadow-md overflow-hidden">
      <CardHeader className="py-2.5 px-4 bg-orange-700 text-white space-y-0">
        <CardTitle className="text-xs font-bold tracking-tight text-white normal-case">Head Control</CardTitle>
      </CardHeader>
      <CardContent className="p-4 flex flex-col items-center">
        <div className="grid grid-cols-3 gap-2 w-fit">
          <div />
          <button
            {...holdPointerHandlers(hold, () => void handleControl('up'))}
            className={buttonClass}
            style={buttonStyle}
            title="Head Up (hold)"
          >
            <ChevronUp size={24} />
          </button>
          <div />

          <button
            {...holdPointerHandlers(hold, () => void handleControl('left'))}
            className={buttonClass}
            style={buttonStyle}
            title="Head Left (hold)"
          >
            <ChevronLeft size={24} />
          </button>

          <button
            {...holdPointerHandlers(hold, () => void handleControl('down'))}
            className={buttonClass}
            style={buttonStyle}
            title="Head Down (hold)"
          >
            <ChevronDown size={24} />
          </button>

          <button
            {...holdPointerHandlers(hold, () => void handleControl('right'))}
            className={buttonClass}
            style={buttonStyle}
            title="Head Right (hold)"
          >
            <ChevronRight size={24} />
          </button>
        </div>
      </CardContent>
    </Card>
  );
};
