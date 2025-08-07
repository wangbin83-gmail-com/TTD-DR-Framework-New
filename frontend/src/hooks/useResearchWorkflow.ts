import { useState, useCallback, useEffect } from 'react';
import { apiClient, wsClient, ApiError } from '../services/api';
import { ResearchRequirements, WorkflowStatus, TTDRState } from '../types';

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

  const [wsConnected, setWsConnected] = useState(false);

  const handleWebSocketMessage = useCallback((data: any) => {
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
  }, []);

  const handleWebSocketError = useCallback((error: Event) => {
    console.error('WebSocket error:', error);
    setError({
      message: 'Connection error occurred',
      code: 'WEBSOCKET_ERROR',
      details: error
    });
  }, []);

  const startWebSocketConnection = useCallback((id: string) => {
    const token = apiClient.getToken();
    wsClient.setToken(token);
    wsClient.connect(id, handleWebSocketMessage, handleWebSocketError);
    setWsConnected(true);
  }, [handleWebSocketMessage, handleWebSocketError]);

  const stopWebSocketConnection = useCallback(() => {
    wsClient.disconnect();
    setWsConnected(false);
  }, []);

  const sendMessage = (data: any) => {
    wsClient.send(data);
  };

  useEffect(() => {
    if (workflowId) {
      startWebSocketConnection(workflowId);
    }

    return () => {
      if (workflowId) {
        stopWebSocketConnection();
      }
    };
  }, [workflowId, startWebSocketConnection, stopWebSocketConnection]);

  const startResearch = useCallback(async (topic: string, requirements: ResearchRequirements) => {
    setIsLoading(true);
    setError(null);
    setFinalReport(null);
    setState(null);
    
    try {
      // Ensure user is logged in
      const token = apiClient.getToken();
      if (!token) {
        // Try to login with dev credentials
        await apiClient.login('dev_user', 'dev_password');
      }

      const response = await apiClient.startResearch(topic, requirements);
      
      if (response.error) {
        throw new Error(response.error);
      }

      if (!response.data || !response.data.execution_id) {
        throw new Error('Invalid response: missing execution_id');
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
      // Ensure user is logged in
      const token = apiClient.getToken();
      if (!token) {
        await apiClient.login('dev_user', 'dev_password');
      }

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
      // Ensure user is logged in
      const token = apiClient.getToken();
      if (!token) {
        await apiClient.login('dev_user', 'dev_password');
      }

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
      // Ensure user is logged in
      const token = apiClient.getToken();
      if (!token) {
        await apiClient.login('dev_user', 'dev_password');
      }

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
      // Ensure user is logged in
      const token = apiClient.getToken();
      if (!token) {
        await apiClient.login('dev_user', 'dev_password');
      }

      const response = await apiClient.getFinalReport(workflowId);
      if (response.data && response.data.report) {
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