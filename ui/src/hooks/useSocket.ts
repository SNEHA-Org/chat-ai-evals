import { useEffect, useState, useCallback, useMemo } from 'react';
import { io, type Socket } from 'socket.io-client';

export interface OptimizationState {
  status: 'idle' | 'running' | 'completed' | 'error';
  progress: number;
  current_step: string;
  results: any;
  session_id: string | null;
  start_time: string | null;
  end_time: string | null;
}

export interface GoldenDataset {
  columns: string[];
  rows: Record<string, any>[];
  total_count: number;
  filename: string;
}

export interface PromptVersion {
  filename: string;
  path: string;
  modified: string;
  size: number;
  type: 'optimized' | 'original';
}

export interface PromptComparison {
  prompt1: {
    path: string;
    filename: string;
    content: string;
    line_count: number;
    char_count: number;
    modified: string;
  } | null;
  prompt2: {
    path: string;
    filename: string;
    content: string;
    line_count: number;
    char_count: number;
    modified: string;
  } | null;
}

export const useSocket = (serverUrl = 'http://localhost:10000') => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [optimizationState, setOptimizationState] = useState<OptimizationState>({
    status: 'idle',
    progress: 0,
    current_step: '',
    results: {},
    session_id: null,
    start_time: null,
    end_time: null,
  });
  const [goldenDataset, setGoldenDataset] = useState<GoldenDataset | null>(null);
  const [promptVersions, setPromptVersions] = useState<PromptVersion[]>([]);
  const [promptComparison, setPromptComparison] = useState<PromptComparison | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const newSocket = io(serverUrl);

    newSocket.on('connect', () => {
      console.log('Connected to server');
      setIsConnected(true);
      setError(null);
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from server');
      setIsConnected(false);
    });

    newSocket.on('connect_error', (err) => {
      console.error('Connection error:', err);
      setError('Failed to connect to server');
      setIsConnected(false);
    });

    // Optimization events
    newSocket.on('state_update', (data: OptimizationState) => {
      setOptimizationState(data);
    });

    newSocket.on('optimization_started', (data: OptimizationState) => {
      setOptimizationState(data);
    });

    newSocket.on('progress_update', (data: OptimizationState) => {
      setOptimizationState(data);
    });

    newSocket.on('optimization_completed', (data: OptimizationState) => {
      setOptimizationState(data);
    });

    newSocket.on('optimization_error', (data: OptimizationState) => {
      setOptimizationState(data);
    });

    // Data events
    newSocket.on('golden_dataset', (data: GoldenDataset) => {
      setGoldenDataset(data);
    });

    newSocket.on('prompt_versions', (data: PromptVersion[]) => {
      setPromptVersions(data);
    });

    newSocket.on('prompt_comparison', (data: PromptComparison) => {
      setPromptComparison(data);
    });

    newSocket.on('error', (data: { message: string }) => {
      setError(data.message);
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, [serverUrl]);

  // Socket actions
  const startOptimization = useCallback((config = {}) => {
    if (socket) {
      socket.emit('start_optimization', { config });
    }
  }, [socket]);

  const getGoldenDataset = useCallback(() => {
    if (socket) {
      socket.emit('get_golden_dataset');
    }
  }, [socket]);

  const getPromptVersions = useCallback(() => {
    if (socket) {
      socket.emit('get_prompt_versions');
    }
  }, [socket]);

  const comparePrompts = useCallback((prompt1: string, prompt2: string) => {
    if (socket) {
      socket.emit('compare_prompts', { prompt1, prompt2 });
    }
  }, [socket]);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const actions = useMemo(() => ({
    startOptimization,
    getGoldenDataset,
    getPromptVersions,
    comparePrompts,
    clearError,
  }), [startOptimization, getGoldenDataset, getPromptVersions, comparePrompts, clearError]);

  return {
    socket,
    isConnected,
    optimizationState,
    goldenDataset,
    promptVersions,
    promptComparison,
    error,
    actions,
  };
};