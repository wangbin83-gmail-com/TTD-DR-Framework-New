import { useState, useCallback } from 'react';
import { apiClient, ApiError } from '../services/api';
import { ResearchRequirements, WorkflowStatus, TTDRState } from '../types';
import { useWebSocket } from './useWebSocket';

export const useResearchWorkflow = () => {
  const [workflowId, setWorkflowId] = useState<string | null>(null);
  const [status, setStatus] = useState<WorkflowStatus>({
    status: 'idle',
    current_node: null,
    progress: 0,
    message: 'Ready to start research'
  });
  const [state, setState] = useState<TTDRState | null>(null);
  const [finalReport, setFinalReport] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  // WebSocket connection for real-time updates
  const { isConnected: wsConnected, sendMessage } = useWebSocket(
    workflowId ? `ws://localhost:8000/api/v1/research/ws/${workflowId}` : '',
    {
      onMessage: (data) => {
        if (data.type === 'status_update') {
          setStatus(data.status);
        } else if (data.type === 'state_update') {
          setState(data.state);
        } else if (data.type === 'report_complete') {
          setFinalReport(data.report);
          setStatus(prev => ({ ...prev, status: 'completed', progress: 1 }));
        } else if (data.type === 'error') {
          setError(data.message);
          setStatus(prev => ({ ...prev, status: 'error' }));
        }
      },
      onError: (error) => {
        console.error('WebSocket error:', error);
        setError({
          message: 'Connection error occurred',
          code: 'WEBSOCKET_ERROR',
          details: error
        });
      }
    }
  );

  const startResearch = useCallback(async (topic: string, requirements: ResearchRequirements) => {
    setIsLoading(true);
    setError(null);
    setFinalReport(null);
    setState(null);
    
    try {
      const response = await apiClient.startResearch(topic, requirements);
      
      if (response.error) {
        throw new Error(response.error);
      }

      setWorkflowId(response.data.execution_id);
      setStatus({
        status: 'running',
        current_node: 'draft_generator',
        progress: 0,
        message: 'Starting research workflow...'
      });

    } catch (err) {
      const apiError = err as ApiError;
      setError(apiError);
      setStatus({
        status: 'error',
        current_node: null,
        progress: 0,
        message: apiError.message || 'Failed to start research'
      });
    } finally {
      setIsLoading(false);
    }
  }, []);

  const stopWorkflow = useCallback(async () => {
    if (!workflowId) return;

    try {
      await apiClient.stopWorkflow(workflowId);
      setStatus(prev => ({
        ...prev,
        status: 'idle',
        message: 'Workflow stopped by user'
      }));
    } catch (err) {
      console.error('Error stopping workflow:', err);
      const apiError = err as ApiError;
      setError(apiError);
    }
  }, [workflowId]);

  const restartWorkflow = useCallback(async () => {
    if (!state) return;

    await startResearch(state.topic, state.requirements);
  }, [state, startResearch]);

  const getWorkflowStatus = useCallback(async () => {
    if (!workflowId) return;

    try {
      const response = await apiClient.getWorkflowStatus(workflowId);
      if (response.data) {
        setStatus(response.data);
      }
    } catch (err) {
      console.error('Error fetching workflow status:', err);
      const apiError = err as ApiError;
      setError(apiError);
    }
  }, [workflowId]);

  const getCurrentState = useCallback(async () => {
    if (!workflowId) return;

    try {
      const response = await apiClient.getCurrentState(workflowId);
      if (response.data) {
        setState(response.data);
      }
    } catch (err) {
      console.error('Error fetching current state:', err);
      const apiError = err as ApiError;
      setError(apiError);
    }
  }, [workflowId]);

  const getFinalReport = useCallback(async () => {
    if (!workflowId) return;

    try {
      const response = await apiClient.getFinalReport(workflowId);
      if (response.data) {
        setFinalReport(response.data.report);
      }
    } catch (err) {
      console.error('Error fetching final report:', err);
      const apiError = err as ApiError;
      setError(apiError);
    }
  }, [workflowId]);

  const resetWorkflow = useCallback(() => {
    setWorkflowId(null);
    setStatus({
      status: 'idle',
      current_node: null,
      progress: 0,
      message: 'Ready to start research'
    });
    setState(null);
    setFinalReport(null);
    setError(null);
    setIsLoading(false);
  }, []);

  return {
    // State
    workflowId,
    status,
    state,
    finalReport,
    isLoading,
    error,
    wsConnected,

    // Actions
    startResearch,
    stopWorkflow,
    restartWorkflow,
    getWorkflowStatus,
    getCurrentState,
    getFinalReport,
    resetWorkflow,

    // WebSocket
    sendMessage
  };
};