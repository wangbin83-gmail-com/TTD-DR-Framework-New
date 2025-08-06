import React from 'react';
import { render, screen, fireEvent, waitFor } from '../../test-utils';
import { ResearchWorkflowApp } from '../ResearchWorkflowApp';
import { BrowserRouter } from 'react-router-dom';
import { AppProviders } from '../AppProviders';

// Mock the research workflow hook
const mockUseResearchWorkflow = {
  workflowId: null as string | null,
  status: {
    status: 'idle' as 'idle' | 'running' | 'completed' | 'error',
    current_node: null as string | null,
    progress: 0,
    message: 'Ready to start',
  },
  state: null as any,
  finalReport: null as string | null,
  isLoading: false,
  error: null as any,
  wsConnected: true,
  startResearch: jest.fn(),
  stopWorkflow: jest.fn(),
  restartWorkflow: jest.fn(),
  resetWorkflow: jest.fn(),
  getCurrentState: jest.fn(),
  getFinalReport: jest.fn(),
};

jest.mock('../../hooks/useResearchWorkflow', () => ({
  useResearchWorkflow: () => mockUseResearchWorkflow,
}));

describe('Integration Tests', () => {
  const renderApp = () => {
    return render(
      <BrowserRouter>
        <AppProviders>
          <ResearchWorkflowApp />
        </AppProviders>
      </BrowserRouter>
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
    // Reset mock state
    mockUseResearchWorkflow.workflowId = null;
    mockUseResearchWorkflow.status = {
      status: 'idle',
      current_node: null,
      progress: 0,
      message: 'Ready to start',
    };
    mockUseResearchWorkflow.state = null;
    mockUseResearchWorkflow.finalReport = null;
    mockUseResearchWorkflow.isLoading = false;
    mockUseResearchWorkflow.error = null;
  });

  describe('Complete User Workflow', () => {
    it('should handle complete research workflow from start to finish', async () => {
      renderApp();

      // 1. Initial state - should show research form
      expect(screen.getByText('Welcome to TTD-DR Framework')).toBeInTheDocument();
      expect(screen.getByText('Start your research journey')).toBeInTheDocument();

      // 2. Fill out research form
      const topicInput = screen.getByLabelText(/research topic/i);
      fireEvent.change(topicInput, { target: { value: 'AI in Healthcare' } });

      const submitButton = screen.getByRole('button', { name: /start research/i });
      fireEvent.click(submitButton);

      // 3. Verify research starts
      expect(mockUseResearchWorkflow.startResearch).toHaveBeenCalledWith(
        'AI in Healthcare',
        expect.any(Object)
      );

      // 4. Simulate workflow running
      mockUseResearchWorkflow.workflowId = 'test-workflow-123';
      mockUseResearchWorkflow.status = {
        status: 'running',
        current_node: 'draft_generator',
        progress: 0.2,
        message: 'Generating initial draft',
      };
      mockUseResearchWorkflow.state = {
        topic: 'AI in Healthcare',
        requirements: {
          domain: 'TECHNOLOGY',
          complexity_level: 'INTERMEDIATE',
          max_iterations: 5,
          quality_threshold: 0.8,
          max_sources: 10,
          preferred_source_types: ['academic', 'news'],
        },
        current_draft: null,
        information_gaps: [],
        retrieved_info: [],
        iteration_count: 1,
        quality_metrics: null,
        evolution_history: [],
        final_report: null,
        error_log: [],
      };

      // Re-render to reflect state change
      renderApp();

      // 5. Should show workflow dashboard
      await waitFor(() => {
        expect(screen.getByText('Workflow Status')).toBeInTheDocument();
        expect(screen.getByText('running')).toBeInTheDocument();
        expect(screen.getByText('20%')).toBeInTheDocument();
      });

      // 6. Simulate workflow completion
      mockUseResearchWorkflow.status = {
        status: 'completed',
        current_node: null,
        progress: 1.0,
        message: 'Research completed successfully',
      };
      mockUseResearchWorkflow.finalReport = '# AI in Healthcare Report\n\nThis is the final report.';

      renderApp();

      // 7. Should show final report
      await waitFor(() => {
        expect(screen.getByText('Research Report')).toBeInTheDocument();
        expect(screen.getByText('AI in Healthcare Report')).toBeInTheDocument();
      });
    });

    it('should handle workflow errors gracefully', async () => {
      renderApp();

      // Start research
      const topicInput = screen.getByLabelText(/research topic/i);
      fireEvent.change(topicInput, { target: { value: 'Test Topic' } });

      const submitButton = screen.getByRole('button', { name: /start research/i });
      fireEvent.click(submitButton);

      // Simulate error
      mockUseResearchWorkflow.error = {
        message: 'Network connection failed',
        code: 'NETWORK_ERROR',
      };
      mockUseResearchWorkflow.status = {
        status: 'error',
        current_node: 'draft_generator',
        progress: 0.1,
        message: 'Error occurred during processing',
      };

      renderApp();

      // Should show error state
      await waitFor(() => {
        expect(screen.getByText('Operation Failed')).toBeInTheDocument();
        expect(screen.getByText('Network connection failed')).toBeInTheDocument();
      });
    });
  });

  describe('Component Interactions', () => {
    it('should handle navigation between different views', async () => {
      // Start with completed workflow
      mockUseResearchWorkflow.workflowId = 'test-workflow-123';
      mockUseResearchWorkflow.status = {
        status: 'completed',
        current_node: null,
        progress: 1.0,
        message: 'Completed',
      };
      mockUseResearchWorkflow.finalReport = '# Test Report\n\nContent here.';
      mockUseResearchWorkflow.state = {
        topic: 'Test Topic',
        requirements: {
          domain: 'GENERAL',
          complexity_level: 'BASIC',
          max_iterations: 3,
          quality_threshold: 0.7,
          max_sources: 5,
          preferred_source_types: ['academic'],
        },
        current_draft: null,
        information_gaps: [],
        retrieved_info: [],
        iteration_count: 2,
        quality_metrics: null,
        evolution_history: [],
        final_report: '# Test Report\n\nContent here.',
        error_log: [],
      };

      renderApp();

      // Should show report view
      expect(screen.getByText('Research Report')).toBeInTheDocument();

      // Navigate to dashboard
      const dashboardButton = screen.getByRole('button', { name: /dashboard/i });
      fireEvent.click(dashboardButton);

      await waitFor(() => {
        expect(screen.getByText('Workflow Status')).toBeInTheDocument();
      });

      // Navigate back to report
      const reportButton = screen.getByRole('button', { name: /report/i });
      fireEvent.click(reportButton);

      await waitFor(() => {
        expect(screen.getByText('Research Report')).toBeInTheDocument();
      });

      // Start new research
      const newResearchButton = screen.getByRole('button', { name: /new research/i });
      fireEvent.click(newResearchButton);

      expect(mockUseResearchWorkflow.resetWorkflow).toHaveBeenCalled();
    });

    it('should handle workflow control actions', async () => {
      // Set up running workflow
      mockUseResearchWorkflow.workflowId = 'test-workflow-123';
      mockUseResearchWorkflow.status = {
        status: 'running',
        current_node: 'gap_analyzer',
        progress: 0.5,
        message: 'Analyzing gaps',
      };

      renderApp();

      // Should show stop and restart buttons
      await waitFor(() => {
        expect(screen.getByText('Workflow Status')).toBeInTheDocument();
      });

      // Test stop workflow
      const stopButton = screen.getByRole('button', { name: /stop/i });
      fireEvent.click(stopButton);

      expect(mockUseResearchWorkflow.stopWorkflow).toHaveBeenCalled();

      // Test restart workflow
      const restartButton = screen.getByRole('button', { name: /restart/i });
      fireEvent.click(restartButton);

      expect(mockUseResearchWorkflow.restartWorkflow).toHaveBeenCalled();
    });
  });

  describe('Real-time Updates', () => {
    it('should handle WebSocket connection status', async () => {
      mockUseResearchWorkflow.workflowId = 'test-workflow-123';
      mockUseResearchWorkflow.wsConnected = true;

      renderApp();

      // Should show connected status
      await waitFor(() => {
        expect(screen.getByText('Connected')).toBeInTheDocument();
      });

      // Simulate disconnection
      mockUseResearchWorkflow.wsConnected = false;
      renderApp();

      await waitFor(() => {
        expect(screen.getByText('Connection Lost')).toBeInTheDocument();
      });
    });

    it('should update progress in real-time', async () => {
      mockUseResearchWorkflow.workflowId = 'test-workflow-123';
      mockUseResearchWorkflow.status = {
        status: 'running',
        current_node: 'draft_generator',
        progress: 0.3,
        message: 'Processing',
      };

      renderApp();

      // Should show initial progress
      await waitFor(() => {
        expect(screen.getByText('30%')).toBeInTheDocument();
      });

      // Update progress
      mockUseResearchWorkflow.status = {
        status: 'running',
        current_node: 'gap_analyzer',
        progress: 0.6,
        message: 'Analyzing gaps',
      };

      renderApp();

      // Should show updated progress
      await waitFor(() => {
        expect(screen.getByText('60%')).toBeInTheDocument();
        expect(screen.getByText('Gap Analyzer')).toBeInTheDocument();
      });
    });
  });

  describe('Error Recovery', () => {
    it('should allow retry after errors', async () => {
      mockUseResearchWorkflow.error = {
        message: 'Temporary error',
        code: 'TEMP_ERROR',
      };
      mockUseResearchWorkflow.status = {
        status: 'error',
        current_node: 'retrieval_engine',
        progress: 0.4,
        message: 'Error occurred',
      };

      renderApp();

      // Should show error notification
      await waitFor(() => {
        expect(screen.getByText('Operation Failed')).toBeInTheDocument();
      });

      // Click retry button
      const retryButton = screen.getByRole('button', { name: /retry/i });
      fireEvent.click(retryButton);

      expect(mockUseResearchWorkflow.restartWorkflow).toHaveBeenCalled();
    });

    it('should handle network errors appropriately', async () => {
      mockUseResearchWorkflow.error = {
        message: 'Network connection failed',
        code: 'NETWORK_ERROR',
      };

      renderApp();

      await waitFor(() => {
        expect(screen.getByText('Network connection failed')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
      });
    });
  });

  describe('Data Persistence', () => {
    it('should handle URL-based workflow loading', async () => {
      // Mock URL with workflow ID
      const mockLocation = {
        pathname: '/research/test-workflow-456',
      };

      Object.defineProperty(window, 'location', {
        value: mockLocation,
        writable: true,
      });

      renderApp();

      // Should attempt to load workflow data
      expect(mockUseResearchWorkflow.getCurrentState).toHaveBeenCalled();
      expect(mockUseResearchWorkflow.getFinalReport).toHaveBeenCalled();
    });

    it('should maintain state across re-renders', async () => {
      mockUseResearchWorkflow.workflowId = 'persistent-workflow';
      mockUseResearchWorkflow.status = {
        status: 'running',
        current_node: 'quality_assessor',
        progress: 0.8,
        message: 'Assessing quality',
      };

      const { rerender } = renderApp();

      // Should show current state
      await waitFor(() => {
        expect(screen.getByText('80%')).toBeInTheDocument();
      });

      // Re-render component
      rerender(
        <BrowserRouter>
          <AppProviders>
            <ResearchWorkflowApp />
          </AppProviders>
        </BrowserRouter>
      );

      // State should persist
      await waitFor(() => {
        expect(screen.getByText('80%')).toBeInTheDocument();
        expect(screen.getByText('Quality Assessor')).toBeInTheDocument();
      });
    });
  });

  describe('Responsive Behavior', () => {
    it('should adapt to mobile viewports', async () => {
      // Mock mobile viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      renderApp();

      // Should show mobile-optimized layout
      expect(screen.getByText('TTD-DR Framework')).toBeInTheDocument();
      
      // Mobile-specific elements should be present
      const mobileElements = screen.queryAllByText(/sm:inline/);
      // In a real test, you'd check for mobile-specific classes or behavior
    });

    it('should handle tablet viewports', async () => {
      // Mock tablet viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 768,
      });

      renderApp();

      // Should show tablet-optimized layout
      expect(screen.getByText('TTD-DR Framework')).toBeInTheDocument();
    });
  });

  describe('Accessibility Integration', () => {
    it('should maintain focus management across views', async () => {
      renderApp();

      // Focus on form input
      const topicInput = screen.getByLabelText(/research topic/i);
      topicInput.focus();
      expect(document.activeElement).toBe(topicInput);

      // Submit form
      fireEvent.change(topicInput, { target: { value: 'Test Topic' } });
      const submitButton = screen.getByRole('button', { name: /start research/i });
      fireEvent.click(submitButton);

      // Focus should be managed appropriately
      // In a real implementation, focus might move to the dashboard
    });

    it('should announce status changes to screen readers', async () => {
      mockUseResearchWorkflow.workflowId = 'test-workflow';
      mockUseResearchWorkflow.status = {
        status: 'running',
        current_node: 'draft_generator',
        progress: 0.2,
        message: 'Starting research',
      };

      renderApp();

      // Status should be announced
      await waitFor(() => {
        expect(screen.getByText('Research in Progress')).toBeInTheDocument();
        expect(screen.getByText('Currently processing: draft generator')).toBeInTheDocument();
      });

      // Update status
      mockUseResearchWorkflow.status = {
        status: 'completed',
        current_node: null,
        progress: 1.0,
        message: 'Research completed',
      };
      mockUseResearchWorkflow.finalReport = '# Final Report';

      renderApp();

      // Completion should be announced
      await waitFor(() => {
        expect(screen.getByText('Research Complete!')).toBeInTheDocument();
      });
    });
  });
});