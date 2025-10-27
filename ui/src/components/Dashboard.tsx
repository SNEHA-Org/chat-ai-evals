import React, { useEffect } from 'react';
import { useSocket } from '../hooks/useSocket';
import { OptimizationProgress } from './OptimizationProgress';
import { GoldenDatasetViewer } from './GoldenDatasetViewer';
import { PromptVersions } from './PromptVersions';
import { PromptComparison } from './PromptComparison';
import { ActivityLog } from './ActivityLog';
import { Play, Database, FileText, BarChart3 } from 'lucide-react';

export const Dashboard: React.FC = () => {
  const {
    isConnected,
    optimizationState,
    goldenDataset,
    promptVersions,
    promptComparison,
    error,
    actions,
  } = useSocket();

  const [showComparison, setShowComparison] = React.useState(false);
  const [logs, setLogs] = React.useState<string[]>([]);

  useEffect(() => {
    // Load initial data
    actions.getGoldenDataset();
    actions.getPromptVersions();
  }, [actions]);

  useEffect(() => {
    // Add logs for optimization events
    if (optimizationState.status === 'running') {
      setLogs(prev => [...prev, `${new Date().toLocaleTimeString()}: ${optimizationState.current_step}`]);
    } else if (optimizationState.status === 'completed') {
      setLogs(prev => [...prev, `${new Date().toLocaleTimeString()}: Optimization completed successfully!`]);
    } else if (optimizationState.status === 'error') {
      setLogs(prev => [...prev, `${new Date().toLocaleTimeString()}: Error - ${optimizationState.current_step}`]);
    }
  }, [optimizationState]);

  useEffect(() => {
    if (error) {
      setLogs(prev => [...prev, `${new Date().toLocaleTimeString()}: Error - ${error}`]);
    }
  }, [error]);

  const handleStartOptimization = () => {
    actions.startOptimization();
    setLogs(prev => [...prev, `${new Date().toLocaleTimeString()}: Starting optimization...`]);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">DSPy Optimization Dashboard</h1>
              <p className="mt-1 text-sm text-gray-500">
                Real-time visualization and comparison of prompt optimization results
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
                isConnected 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-red-100 text-red-800'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  isConnected ? 'bg-green-500' : 'bg-red-500'
                }`}></div>
                <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Optimization Progress */}
        <div className="mb-8">
          <OptimizationProgress 
            state={optimizationState}
            onStart={handleStartOptimization}
          />
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Golden Dataset */}
          <div className="bg-white rounded-lg shadow-sm border">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Database className="h-5 w-5 text-blue-500" />
                  <h2 className="text-lg font-semibold text-gray-900">Golden Dataset</h2>
                </div>
                <button
                  onClick={actions.getGoldenDataset}
                  className="px-3 py-1 text-sm bg-blue-50 text-blue-600 rounded-md hover:bg-blue-100 transition-colors"
                >
                  Refresh
                </button>
              </div>
            </div>
            <div className="p-6">
              <GoldenDatasetViewer dataset={goldenDataset} />
            </div>
          </div>

          {/* Prompt Versions */}
          <div className="bg-white rounded-lg shadow-sm border">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <FileText className="h-5 w-5 text-green-500" />
                  <h2 className="text-lg font-semibold text-gray-900">Prompt Versions</h2>
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={actions.getPromptVersions}
                    className="px-3 py-1 text-sm bg-green-50 text-green-600 rounded-md hover:bg-green-100 transition-colors"
                  >
                    Refresh
                  </button>
                  <button
                    onClick={() => setShowComparison(true)}
                    disabled={promptVersions.length < 2}
                    className="px-3 py-1 text-sm bg-purple-50 text-purple-600 rounded-md hover:bg-purple-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Compare
                  </button>
                </div>
              </div>
            </div>
            <div className="p-6">
              <PromptVersions versions={promptVersions} />
            </div>
          </div>
        </div>

        {/* Results Panel */}
        {optimizationState.results && Object.keys(optimizationState.results).length > 0 && (
          <div className="bg-white rounded-lg shadow-sm border mb-8">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center space-x-2">
                <BarChart3 className="h-5 w-5 text-orange-500" />
                <h2 className="text-lg font-semibold text-gray-900">Optimization Results</h2>
              </div>
            </div>
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {optimizationState.results.original_score?.toFixed(2) || 'N/A'}
                  </div>
                  <div className="text-sm text-gray-500">Original Score</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {optimizationState.results.optimized_score?.toFixed(2) || 'N/A'}
                  </div>
                  <div className="text-sm text-gray-500">Optimized Score</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-600">
                    {optimizationState.results.improvement?.toFixed(1) || 'N/A'}%
                  </div>
                  <div className="text-sm text-gray-500">Improvement</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {optimizationState.results.total_examples || 'N/A'}
                  </div>
                  <div className="text-sm text-gray-500">Examples Used</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Activity Log */}
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">Activity Log</h2>
          </div>
          <div className="p-6">
            <ActivityLog logs={logs} />
          </div>
        </div>
      </div>

      {/* Prompt Comparison Modal */}
      {showComparison && (
        <PromptComparison
          versions={promptVersions}
          comparison={promptComparison}
          onCompare={actions.comparePrompts}
          onClose={() => setShowComparison(false)}
        />
      )}
    </div>
  );
};