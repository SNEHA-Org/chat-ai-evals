import React, { useEffect, useRef } from 'react';
import { Terminal } from 'lucide-react';

interface ActivityLogProps {
  logs: string[];
}

export const ActivityLog: React.FC<ActivityLogProps> = ({ logs }) => {
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Auto-scroll to bottom when new logs are added
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  if (logs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-gray-500">
        <Terminal className="h-12 w-12 mb-4" />
        <p className="text-lg font-medium">No activity yet</p>
        <p className="text-sm">Activity logs will appear here</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm">
      <div className="space-y-1">
        {logs.map((log, index) => (
          <div key={index} className="text-green-400">
            {log}
          </div>
        ))}
        <div ref={logEndRef} />
      </div>
    </div>
  );
};