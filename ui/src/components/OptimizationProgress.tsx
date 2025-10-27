import React from 'react';
import { Play, Square, CheckCircle, XCircle, Clock } from 'lucide-react';
import type { OptimizationState } from '../hooks/useSocket';

interface OptimizationProgressProps {
  state: OptimizationState;
  onStart: () => void;
}

export const OptimizationProgress: React.FC<OptimizationProgressProps> = ({ state, onStart }) => {
  const getStatusIcon = () => {
    switch (state.status) {
      case 'idle':
        return <Square className="h-5 w-5 text-gray-500" />;
      case 'running':
        return <Clock className="h-5 w-5 text-blue-500 animate-spin" />;
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'error':
        return <XCircle className="h-5 w-5 text-red-500" />;
      default:
        return <Square className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusColor = () => {
    switch (state.status) {
      case 'idle':
        return 'text-gray-600';
      case 'running':
        return 'text-blue-600';
      case 'completed':
        return 'text-green-600';
      case 'error':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  const getProgressBarColor = () => {
    switch (state.status) {
      case 'running':
        return 'bg-blue-500';
      case 'completed':
        return 'bg-green-500';
      case 'error':
        return 'bg-red-500';
      default:
        return 'bg-gray-300';
    }
  };

  const formatDuration = () => {
    if (!state.start_time) return null;
    
    const start = new Date(state.start_time);
    const end = state.end_time ? new Date(state.end_time) : new Date();
    const duration = Math.floor((end.getTime() - start.getTime()) / 1000);
    
    const minutes = Math.floor(duration / 60);
    const seconds = duration % 60;
    
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border">
      <div className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            {getStatusIcon()}
            <div>
              <h2 className="text-lg font-semibold text-gray-900">Optimization Status</h2>
              <p className={`text-sm font-medium ${getStatusColor()}`}>
                Status: {state.status.charAt(0).toUpperCase() + state.status.slice(1)}
                {state.session_id && (
                  <span className="ml-2 text-gray-500">
                    (Session: {state.session_id.slice(0, 8)})
                  </span>
                )}
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            {formatDuration() && (
              <div className="text-sm text-gray-500">
                Duration: {formatDuration()}
              </div>
            )}
            <button
              onClick={onStart}
              disabled={state.status === 'running'}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              <Play className="h-4 w-4" />
              <span>{state.status === 'running' ? 'Running...' : 'Start Optimization'}</span>
            </button>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mb-4">
          <div className="flex justify-between text-sm text-gray-600 mb-2">
            <span>Progress</span>
            <span>{state.progress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div
              className={`h-3 rounded-full transition-all duration-300 ${getProgressBarColor()}`}
              style={{ width: `${state.progress}%` }}
            />
          </div>
        </div>

        {/* Current Step */}
        {state.current_step && (
          <div className="text-sm text-gray-600">
            <span className="font-medium">Current Step:</span> {state.current_step}
          </div>
        )}

        {/* Error Display */}
        {state.status === 'error' && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
            <div className="flex items-center space-x-2">
              <XCircle className="h-4 w-4 text-red-500" />
              <span className="text-sm text-red-700 font-medium">Optimization Failed</span>
            </div>
            <p className="text-sm text-red-600 mt-1">{state.current_step}</p>
          </div>
        )}

        {/* Success Display */}
        {state.status === 'completed' && (
          <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-md">
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-4 w-4 text-green-500" />
              <span className="text-sm text-green-700 font-medium">Optimization Completed</span>
            </div>
            <p className="text-sm text-green-600 mt-1">
              Optimization finished successfully! Check the results below.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};