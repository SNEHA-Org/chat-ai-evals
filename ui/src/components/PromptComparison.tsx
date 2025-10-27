import React, { useState, useEffect } from 'react';
import { X, FileText, Calendar, Hash, Type } from 'lucide-react';
import type { PromptVersion, PromptComparison as PromptComparisonType } from '../hooks/useSocket';

interface PromptComparisonProps {
  versions: PromptVersion[];
  comparison: PromptComparisonType | null;
  onCompare: (prompt1: string, prompt2: string) => void;
  onClose: () => void;
}

export const PromptComparison: React.FC<PromptComparisonProps> = ({
  versions,
  comparison,
  onCompare,
  onClose,
}) => {
  const [selectedPrompt1, setSelectedPrompt1] = useState<string>('');
  const [selectedPrompt2, setSelectedPrompt2] = useState<string>('');

  useEffect(() => {
    // Auto-select the two most recent prompts if available
    if (versions.length >= 2) {
      setSelectedPrompt1(versions[0].path);
      setSelectedPrompt2(versions[1].path);
    }
  }, [versions]);

  useEffect(() => {
    // Trigger comparison when both prompts are selected
    if (selectedPrompt1 && selectedPrompt2) {
      onCompare(selectedPrompt1, selectedPrompt2);
    }
  }, [selectedPrompt1, selectedPrompt2, onCompare]);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  const getPromptStats = (prompt: any) => {
    if (!prompt) return null;
    
    return (
      <div className="flex items-center space-x-4 text-xs text-gray-500 mb-2">
        <div className="flex items-center space-x-1">
          <Calendar className="h-3 w-3" />
          <span>{formatDate(prompt.modified)}</span>
        </div>
        <div className="flex items-center space-x-1">
          <Hash className="h-3 w-3" />
          <span>{prompt.line_count} lines</span>
        </div>
        <div className="flex items-center space-x-1">
          <Type className="h-3 w-3" />
          <span>{prompt.char_count} chars</span>
        </div>
      </div>
    );
  };

  const renderPromptSelector = (
    value: string,
    onChange: (value: string) => void,
    label: string
  ) => (
    <div className="mb-4">
      <label className="block text-sm font-medium text-gray-700 mb-2">
        {label}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full p-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
      >
        <option value="">Select a prompt...</option>
        {versions.map((version) => (
          <option key={version.path} value={version.path}>
            {version.filename} ({version.type})
          </option>
        ))}
      </select>
    </div>
  );

  const renderPromptContent = (prompt: any, title: string) => (
    <div className="flex-1 flex flex-col">
      <div className="bg-gray-50 px-4 py-3 border-b border-gray-200">
        <div className="flex items-center space-x-2">
          <FileText className="h-4 w-4 text-gray-500" />
          <h3 className="font-medium text-gray-900">
            {prompt ? prompt.filename : title}
          </h3>
        </div>
        {prompt && getPromptStats(prompt)}
      </div>
      
      <div className="flex-1 p-4 overflow-auto">
        {prompt ? (
          <pre className="text-xs font-mono whitespace-pre-wrap break-words text-gray-800 leading-relaxed">
            {prompt.content}
          </pre>
        ) : (
          <div className="flex items-center justify-center h-full text-gray-500">
            <div className="text-center">
              <FileText className="h-12 w-12 mx-auto mb-4 text-gray-300" />
              <p>Select a prompt to view content</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-7xl h-full max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">Prompt Comparison</h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-full transition-colors"
          >
            <X className="h-5 w-5 text-gray-500" />
          </button>
        </div>

        {/* Selectors */}
        <div className="p-6 border-b border-gray-200 bg-gray-50">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {renderPromptSelector(selectedPrompt1, setSelectedPrompt1, "First Prompt")}
            {renderPromptSelector(selectedPrompt2, setSelectedPrompt2, "Second Prompt")}
          </div>
        </div>

        {/* Comparison View */}
        <div className="flex-1 flex overflow-hidden">
          {/* Left Panel */}
          <div className="flex-1 border-r border-gray-200 flex flex-col">
            {renderPromptContent(comparison?.prompt1, "First Prompt")}
          </div>

          {/* Right Panel */}
          <div className="flex-1 flex flex-col">
            {renderPromptContent(comparison?.prompt2, "Second Prompt")}
          </div>
        </div>

        {/* Footer with comparison stats */}
        {comparison?.prompt1 && comparison?.prompt2 && (
          <div className="p-4 border-t border-gray-200 bg-gray-50">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="space-y-1">
                <div className="flex justify-between">
                  <span className="text-gray-600">Lines:</span>
                  <span className="font-mono">{comparison.prompt1.line_count}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Characters:</span>
                  <span className="font-mono">{comparison.prompt1.char_count}</span>
                </div>
              </div>
              <div className="space-y-1">
                <div className="flex justify-between">
                  <span className="text-gray-600">Lines:</span>
                  <span className="font-mono">{comparison.prompt2.line_count}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Characters:</span>
                  <span className="font-mono">{comparison.prompt2.char_count}</span>
                </div>
              </div>
            </div>
            
            {/* Difference stats */}
            <div className="mt-3 pt-3 border-t border-gray-200">
              <div className="flex justify-center space-x-8 text-sm">
                <div className="text-center">
                  <div className="font-mono text-lg">
                    {Math.abs(comparison.prompt1.line_count - comparison.prompt2.line_count)}
                  </div>
                  <div className="text-gray-600">Line diff</div>
                </div>
                <div className="text-center">
                  <div className="font-mono text-lg">
                    {Math.abs(comparison.prompt1.char_count - comparison.prompt2.char_count)}
                  </div>
                  <div className="text-gray-600">Char diff</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};