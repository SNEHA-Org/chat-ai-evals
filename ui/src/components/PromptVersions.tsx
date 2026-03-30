import React from 'react';
import { FileText, Clock, Zap, HardDrive } from 'lucide-react';
import type { PromptVersion } from '../hooks/useSocket';

interface PromptVersionsProps {
  versions: PromptVersion[];
}

export const PromptVersions: React.FC<PromptVersionsProps> = ({ versions }) => {
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const getTypeIcon = (type: string) => {
    return type === 'optimized' ? (
      <Zap className="h-4 w-4 text-yellow-500" />
    ) : (
      <FileText className="h-4 w-4 text-blue-500" />
    );
  };

  const getTypeBadge = (type: string) => {
    return type === 'optimized' ? (
      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
        Optimized
      </span>
    ) : (
      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
        Original
      </span>
    );
  };

  if (versions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-gray-500">
        <FileText className="h-12 w-12 mb-4" />
        <p className="text-lg font-medium">No prompt versions found</p>
        <p className="text-sm">Prompt files will appear here after optimization</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {versions.map((version, index) => (
        <div
          key={version.path}
          className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors"
        >
          <div className="flex items-start justify-between">
            <div className="flex items-start space-x-3">
              {getTypeIcon(version.type)}
              <div className="flex-1 min-w-0">
                <div className="flex items-center space-x-2 mb-1">
                  <p className="text-sm font-medium text-gray-900 truncate">
                    {version.filename}
                  </p>
                  {getTypeBadge(version.type)}
                </div>
                
                <div className="flex items-center space-x-4 text-xs text-gray-500">
                  <div className="flex items-center space-x-1">
                    <Clock className="h-3 w-3" />
                    <span>{formatDate(version.modified)}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <HardDrive className="h-3 w-3" />
                    <span>{formatFileSize(version.size)}</span>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              {index === 0 && (
                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                  Latest
                </span>
              )}
            </div>
          </div>
        </div>
      ))}
      
      {versions.length > 0 && (
        <div className="text-xs text-gray-500 text-center pt-2">
          {versions.length} version{versions.length !== 1 ? 's' : ''} available
        </div>
      )}
    </div>
  );
};