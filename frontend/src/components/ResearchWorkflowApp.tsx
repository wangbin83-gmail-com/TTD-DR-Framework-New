import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ResearchForm } from './ResearchForm';
import { WorkflowDashboard } from './WorkflowDashboard';
import { ReportDisplay } from './ReportDisplay';
import { useResearchWorkflow } from '../hooks/useResearchWorkflow';
import { useNotifications } from './NotificationSystem';
import { useLoading } from './LoadingSystem';
import { ErrorBoundary } from './ErrorBoundary';
import { ResearchRequirements } from '../types';
import { FileText, Home, RefreshCw, AlertCircle, CheckCircle, Wifi, WifiOff } from 'lucide-react';

export const ResearchWorkflowApp: React.FC = () => {
  const { workflowId: urlWorkflowId } = useParams<{ workflowId: string }>();
  const navigate = useNavigate();
  const [currentView, setCurrentView] = useState<'form' | 'dashboard' | 'report'>('form');

  const {
    workflowId,
    status,
    state,
    finalReport,
    isLoading,
    error,
    wsConnected,
    startResearch,
    stopWorkflow,
    restartWorkflow,
    resetWorkflow,
    getCurrentState,
    getFinalReport
  } = useResearchWorkflow();

  const { showSuccess, showError, showWarning, showInfo } = useNotifications();
  const { startLoading, finishLoading, isLoading: isLoadingGlobal } = useLoading();

  // Handle URL-based workflow ID
  useEffect(() => {
    if (urlWorkflowId && urlWorkflowId !== workflowId) {
      // If there's a workflow ID in the URL, try to load that workflow
      startLoading('load-workflow', 'Loading workflow data...');
      Promise.all([getCurrentState(), getFinalReport()])
        .then(() => {
          finishLoading('load-workflow', 'success', 'Workflow data loaded');
        })
        .catch(() => {
          finishLoading('load-workflow', 'error', 'Failed to load workflow data');
        });
    }
  }, [urlWorkflowId, workflowId, getCurrentState, getFinalReport, startLoading, finishLoading]);

  // Handle error notifications
  useEffect(() => {
    if (error) {
      showError(
        'Operation Failed',
        error.message,
        {
          persistent: true,
          action: error.code === 'NETWORK_ERROR' ? {
            label: 'Retry',
            onClick: () => {
              if (status.status === 'error' && state) {
                handleRestartWorkflow();
              }
            }
          } : undefined
        }
      );
    }
  }, [error, showError, status.status, state]);

  // Handle status change notifications
  useEffect(() => {
    if (status.status === 'completed' && finalReport) {
      showSuccess(
        'Research Complete!',
        'Your research report has been generated successfully.',
        {
          action: {
            label: 'View Report',
            onClick: () => setCurrentView('report')
          }
        }
      );
    } else if (status.status === 'running' && status.current_node) {
      showInfo(
        'Research in Progress',
        `Currently processing: ${status.current_node.replace('_', ' ')}`,
        { duration: 3000 }
      );
    }
  }, [status.status, status.current_node, finalReport, showSuccess, showInfo]);

  // Handle WebSocket connection status
  useEffect(() => {
    if (workflowId) {
      if (wsConnected) {
        showSuccess('Connected', 'Real-time updates enabled', { duration: 2000 });
      } else {
        showWarning('Connection Lost', 'Real-time updates disabled. Trying to reconnect...', { duration: 4000 });
      }
    }
  }, [wsConnected, workflowId, showSuccess, showWarning]);

  // Update view based on workflow status
  useEffect(() => {
    if (workflowId && status.status === 'running') {
      setCurrentView('dashboard');
      // Update URL to include workflow ID
      navigate(`/research/${workflowId}`, { replace: true });
    } else if (finalReport) {
      setCurrentView('report');
    } else if (!workflowId) {
      setCurrentView('form');
      navigate('/', { replace: true });
    }
  }, [workflowId, status.status, finalReport, navigate]);

  const handleStartResearch = async (topic: string, requirements: ResearchRequirements) => {
    startLoading('start-research', 'Starting research workflow...', 'Initializing components');
    try {
      await startResearch(topic, requirements);
      finishLoading('start-research', 'success', 'Research workflow started');
      showSuccess('Research Started', `Research on "${topic}" has begun!`);
    } catch (err) {
      finishLoading('start-research', 'error', 'Failed to start research');
    }
  };

  const handleStopWorkflow = async () => {
    startLoading('stop-workflow', 'Stopping workflow...', 'Terminating processes');
    try {
      await stopWorkflow();
      finishLoading('stop-workflow', 'success', 'Workflow stopped');
      showInfo('Workflow Stopped', 'Research workflow has been terminated by user.');
    } catch (err) {
      finishLoading('stop-workflow', 'error', 'Failed to stop workflow');
    }
  };

  const handleRestartWorkflow = async () => {
    startLoading('restart-workflow', 'Restarting workflow...', 'Reinitializing research');
    try {
      await restartWorkflow();
      finishLoading('restart-workflow', 'success', 'Workflow restarted');
      showInfo('Workflow Restarted', 'Research workflow has been restarted.');
    } catch (err) {
      finishLoading('restart-workflow', 'error', 'Failed to restart workflow');
    }
  };

  const handleNewResearch = () => {
    resetWorkflow();
    setCurrentView('form');
  };

  const handleViewReport = () => {
    if (finalReport) {
      setCurrentView('report');
    }
  };

  const handleViewDashboard = () => {
    if (workflowId) {
      setCurrentView('dashboard');
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Skip Links for Accessibility */}
      <div className="sr-only focus:not-sr-only">
        <a 
          href="#main-content" 
          className="absolute top-0 left-0 bg-primary-600 text-white p-2 z-50 focus:relative"
          onClick={(e) => {
            e.preventDefault();
            const mainContent = document.getElementById('main-content');
            if (mainContent) {
              mainContent.focus();
              mainContent.scrollIntoView({ behavior: 'smooth' });
            }
          }}
        >
          Skip to main content
        </a>
        <a 
          href="#navigation" 
          className="absolute top-0 left-20 bg-primary-600 text-white p-2 z-50 focus:relative"
          onClick={(e) => {
            e.preventDefault();
            const navigation = document.getElementById('navigation');
            if (navigation) {
              navigation.focus();
              navigation.scrollIntoView({ behavior: 'smooth' });
            }
          }}
        >
          Skip to navigation
        </a>
      </div>

      {/* Navigation Header */}
      <header className="bg-white shadow-sm border-b" role="banner">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16 md:h-20">
            <div className="flex items-center space-x-2 md:space-x-4 min-w-0 flex-1">
              <h1 className="text-lg md:text-xl font-bold text-gray-900 truncate">TTD-DR Framework</h1>
              <span className="hidden sm:inline text-xs md:text-sm text-gray-500 truncate">Test-Time Diffusion Deep Researcher</span>
            </div>

            <div className="flex items-center space-x-2 md:space-x-4">
              {/* Connection Status */}
              {workflowId && (
                <div className="hidden sm:flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-500' : 'bg-red-500'}`} />
                  <span className="text-xs md:text-sm text-gray-600">
                    {wsConnected ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
              )}

              {/* Navigation Buttons */}
              <nav id="navigation" className="flex space-x-1 md:space-x-2" role="navigation" aria-label="Main navigation">
                <button
                  onClick={handleNewResearch}
                  className="btn-secondary space-x-1 md:space-x-2 text-sm px-2 md:px-4 py-2"
                >
                  <Home className="w-4 h-4" />
                  <span className="hidden sm:inline">New Research</span>
                </button>

                {workflowId && status.status === 'running' && (
                  <button
                    onClick={handleViewDashboard}
                    className={`btn-base space-x-1 md:space-x-2 text-sm px-2 md:px-4 py-2 ${
                      currentView === 'dashboard' 
                        ? 'bg-primary-100 text-primary-700 border-primary-300' 
                        : 'bg-neutral-100 text-neutral-700 hover:bg-neutral-200 border-neutral-300'
                    }`}
                  >
                    <RefreshCw className="w-4 h-4" />
                    <span className="hidden sm:inline">Dashboard</span>
                  </button>
                )}

                {finalReport && (
                  <button
                    onClick={handleViewReport}
                    className={`btn-base space-x-1 md:space-x-2 text-sm px-2 md:px-4 py-2 ${
                      currentView === 'report' 
                        ? 'bg-primary-100 text-primary-700 border-primary-300' 
                        : 'bg-neutral-100 text-neutral-700 hover:bg-neutral-200 border-neutral-300'
                    }`}
                  >
                    <FileText className="w-4 h-4" />
                    <span className="hidden sm:inline">Report</span>
                  </button>
                )}
              </nav>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main id="main-content" className="py-4 md:py-8" tabIndex={-1} role="main">
        {/* Enhanced Connection Status */}
        {workflowId && (
          <div className="max-w-4xl mx-auto px-4 mb-4">
            <div className={`flex items-center justify-between p-3 rounded-lg border transition-all duration-200 ${
              wsConnected 
                ? 'bg-success-50 border-success-200 text-success-800' 
                : 'bg-warning-50 border-warning-200 text-warning-800'
            }`}>
              <div className="flex items-center space-x-2">
                {wsConnected ? (
                  <Wifi className="w-4 h-4 text-success-600" />
                ) : (
                  <WifiOff className="w-4 h-4 text-warning-600" />
                )}
                <span className="text-sm font-medium">
                  {wsConnected ? 'Real-time updates active' : 'Connection interrupted'}
                </span>
              </div>
              {!wsConnected && (
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-warning-500 rounded-full animate-pulse" />
                  <span className="text-xs">Reconnecting...</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Content Views */}
        {currentView === 'form' && (
          <div className="max-w-4xl mx-auto px-4">
            <ResearchForm 
              onSubmit={handleStartResearch} 
              isLoading={isLoading} 
            />
          </div>
        )}

        {currentView === 'dashboard' && workflowId && (
          <div className="w-full">
            <WorkflowDashboard
              workflowId={workflowId}
              status={status}
              state={state}
              onStop={handleStopWorkflow}
              onRestart={handleRestartWorkflow}
            />
          </div>
        )}

        {currentView === 'report' && finalReport && state && (
          <div className="w-full">
            <ReportDisplay
              report={finalReport}
              topic={state.topic}
              workflowId={workflowId || 'unknown'}
            />
          </div>
        )}

        {/* Loading State */}
        {isLoading && currentView === 'form' && (
          <div className="max-w-4xl mx-auto px-4">
            <div className="bg-white rounded-lg shadow-lg p-6 md:p-8 text-center">
              <div className="animate-spin rounded-full h-10 w-10 md:h-12 md:w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <h3 className="text-base md:text-lg font-medium text-gray-900 mb-2">Starting Research Workflow</h3>
              <p className="text-sm md:text-base text-gray-600">Please wait while we initialize your research session...</p>
            </div>
          </div>
        )}

      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex flex-col sm:flex-row justify-between items-center space-y-2 sm:space-y-0">
            <p className="text-xs sm:text-sm text-gray-500 text-center sm:text-left">
              © 2024 TTD-DR Framework. Powered by Kimi K2 and Google Search API.
            </p>
            <div className="flex space-x-2 sm:space-x-4 text-xs sm:text-sm text-gray-500">
              <span>Version 1.0.0</span>
              {workflowId && (
                <span className="hidden sm:inline">Workflow: {workflowId.substring(0, 8)}...</span>
              )}
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};