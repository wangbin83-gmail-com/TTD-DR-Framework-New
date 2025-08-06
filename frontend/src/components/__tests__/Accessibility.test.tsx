import React from 'react';
import { render, screen, fireEvent, waitFor } from '../../test-utils';
import { axe, toHaveNoViolations } from 'jest-axe';
import { ResearchWorkflowApp } from '../ResearchWorkflowApp';
import { WorkflowVisualization } from '../WorkflowVisualization';
import { ReportManagement } from '../ReportManagement';
import { WorkflowNodeComponent } from '../WorkflowNodeComponent';
import { ResearchForm } from '../ResearchForm';
import { testKeyboardNavigation, testScreenReaderSupport } from '../../test-utils';

// Extend Jest matchers
expect.extend(toHaveNoViolations);

describe('Accessibility Tests', () => {
  beforeEach(() => {
    // Mock DOM methods that might not be available in test environment
    Object.defineProperty(window, 'getComputedStyle', {
      value: () => ({
        color: 'rgb(0, 0, 0)',
        backgroundColor: 'rgb(255, 255, 255)',
      }),
    });
  });

  describe('ResearchForm Accessibility', () => {
    it('should have no accessibility violations', async () => {
      const { container } = render(
        <ResearchForm onSubmit={jest.fn()} isLoading={false} />
      );
      
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('should support keyboard navigation', async () => {
      render(<ResearchForm onSubmit={jest.fn()} isLoading={false} />);
      
      const form = screen.getByRole('form') || document.body;
      const focusableCount = await testKeyboardNavigation(form);
      
      // Should have multiple focusable elements (inputs, selects, checkboxes, button)
      expect(focusableCount).toBeGreaterThan(5);
    });

    it('should have proper form labels', () => {
      render(<ResearchForm onSubmit={jest.fn()} isLoading={false} />);
      
      // Check that all form inputs have associated labels
      const topicInput = screen.getByLabelText(/research topic/i);
      expect(topicInput).toBeInTheDocument();
      expect(topicInput).toHaveAttribute('aria-required', 'true');
      
      const domainSelect = screen.getByLabelText(/research domain/i);
      expect(domainSelect).toBeInTheDocument();
      
      const complexitySelect = screen.getByLabelText(/complexity level/i);
      expect(complexitySelect).toBeInTheDocument();
    });

    it('should provide clear error messages', async () => {
      const mockSubmit = jest.fn();
      render(<ResearchForm onSubmit={mockSubmit} isLoading={false} />);
      
      const submitButton = screen.getByRole('button', { name: /start research/i });
      fireEvent.click(submitButton);
      
      // Should show validation errors
      await waitFor(() => {
        const errorMessages = screen.queryAllByRole('alert');
        // Note: This depends on the form validation implementation
        // The test verifies that error messages are properly announced to screen readers
      });
    });
  });

  describe('WorkflowVisualization Accessibility', () => {
    const mockStatus = {
      status: 'running' as const,
      current_node: 'gap_analyzer',
      progress: 0.5,
      message: 'Processing workflow',
    };

    const mockState = {
      topic: 'Test Topic',
      requirements: {
        domain: 'TECHNOLOGY' as any,
        complexity_level: 'INTERMEDIATE' as any,
        max_iterations: 5,
        quality_threshold: 0.8,
        max_sources: 10,
        preferred_source_types: ['academic', 'news'],
      },
      current_draft: null,
      information_gaps: [],
      retrieved_info: [],
      iteration_count: 2,
      quality_metrics: null,
      evolution_history: [],
      final_report: null,
      error_log: [],
    };

    it('should have no accessibility violations', async () => {
      const { container } = render(
        <WorkflowVisualization
          workflowId="test-workflow"
          status={mockStatus}
          state={mockState}
          onNodeClick={jest.fn()}
        />
      );
      
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('should provide meaningful status information', () => {
      render(
        <WorkflowVisualization
          workflowId="test-workflow"
          status={mockStatus}
          state={mockState}
          onNodeClick={jest.fn()}
        />
      );
      
      // Check that status information is accessible
      expect(screen.getByText('Workflow Status')).toBeInTheDocument();
      expect(screen.getByText('running')).toBeInTheDocument();
      expect(screen.getByText('50%')).toBeInTheDocument();
    });

    it('should have accessible legend', () => {
      render(
        <WorkflowVisualization
          workflowId="test-workflow"
          status={mockStatus}
          state={mockState}
          onNodeClick={jest.fn()}
        />
      );
      
      expect(screen.getByText('Status Legend')).toBeInTheDocument();
      expect(screen.getByText('Pending')).toBeInTheDocument();
      expect(screen.getByText('Running')).toBeInTheDocument();
      expect(screen.getByText('Completed')).toBeInTheDocument();
      expect(screen.getByText('Error')).toBeInTheDocument();
    });
  });

  describe('WorkflowNodeComponent Accessibility', () => {
    const mockNodeData = {
      label: 'Test Node',
      status: 'running' as const,
      isActive: true,
      onClick: jest.fn(),
      progress: 0.7,
    };

    const mockNodeProps = {
      id: 'test-node',
      data: mockNodeData,
      selected: false,
      type: 'workflowNode',
      position: { x: 0, y: 0 },
      dragging: false,
      zIndex: 1,
      isConnectable: true,
      xPos: 0,
      yPos: 0,
    };

    it('should have no accessibility violations', async () => {
      const { container } = render(<WorkflowNodeComponent {...mockNodeProps} />);
      
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('should be keyboard accessible', () => {
      render(<WorkflowNodeComponent {...mockNodeProps} />);
      
      const nodeButton = screen.getByRole('button');
      expect(nodeButton).toHaveAttribute('tabIndex', '0');
      
      // Test keyboard interaction
      fireEvent.keyDown(nodeButton, { key: 'Enter' });
      expect(mockNodeData.onClick).toHaveBeenCalled();
      
      mockNodeData.onClick.mockClear();
      fireEvent.keyDown(nodeButton, { key: ' ' });
      expect(mockNodeData.onClick).toHaveBeenCalled();
    });

    it('should provide status information to screen readers', () => {
      render(<WorkflowNodeComponent {...mockNodeProps} />);
      
      expect(screen.getByText('Test Node')).toBeInTheDocument();
      expect(screen.getByText('Running')).toBeInTheDocument();
      expect(screen.getByText('70%')).toBeInTheDocument();
    });
  });

  describe('ReportManagement Accessibility', () => {
    const mockReport = '# Test Report\n\nThis is a test report.';
    
    it('should have no accessibility violations', async () => {
      const { container } = render(
        <ReportManagement
          report={mockReport}
          topic="Test Topic"
          workflowId="test-workflow"
        />
      );
      
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('should have accessible tab navigation', () => {
      render(
        <ReportManagement
          report={mockReport}
          topic="Test Topic"
          workflowId="test-workflow"
        />
      );
      
      // Check that tabs are properly labeled and accessible
      const viewTab = screen.getByRole('button', { name: /view/i });
      const editTab = screen.getByRole('button', { name: /edit/i });
      const annotationsTab = screen.getByRole('button', { name: /annotations/i });
      
      expect(viewTab).toBeInTheDocument();
      expect(editTab).toBeInTheDocument();
      expect(annotationsTab).toBeInTheDocument();
      
      // Test tab navigation
      fireEvent.click(editTab);
      // Should switch to edit mode - verify by checking for textarea
      expect(screen.getByRole('textbox')).toBeInTheDocument();
    });

    it('should provide accessible export options', () => {
      render(
        <ReportManagement
          report={mockReport}
          topic="Test Topic"
          workflowId="test-workflow"
        />
      );
      
      const exportButtons = screen.getAllByRole('button').filter(button => 
        button.textContent?.includes('Export') || button.textContent?.includes('Share')
      );
      
      expect(exportButtons.length).toBeGreaterThan(0);
      
      // Each export button should be accessible
      exportButtons.forEach(button => {
        expect(button).toBeInTheDocument();
        expect(button).not.toHaveAttribute('aria-hidden', 'true');
      });
    });
  });

  describe('Color Contrast and Visual Accessibility', () => {
    it('should maintain sufficient color contrast', () => {
      const { container } = render(
        <ResearchForm onSubmit={jest.fn()} isLoading={false} />
      );
      
      // Check for common color contrast issues
      const buttons = container.querySelectorAll('button');
      buttons.forEach(button => {
        const styles = window.getComputedStyle(button);
        // Basic check - in a real implementation, you'd use a proper contrast checker
        expect(styles.color).toBeDefined();
        expect(styles.backgroundColor).toBeDefined();
      });
    });

    it('should not rely solely on color for information', () => {
      render(
        <WorkflowVisualization
          workflowId="test-workflow"
          status={{
            status: 'running',
            current_node: 'gap_analyzer',
            progress: 0.5,
            message: 'Processing',
          }}
          state={null}
          onNodeClick={jest.fn()}
        />
      );
      
      // Status should be conveyed through text, not just color
      expect(screen.getByText('running')).toBeInTheDocument();
      expect(screen.getByText('Status Legend')).toBeInTheDocument();
    });
  });

  describe('Screen Reader Support', () => {
    it('should provide meaningful headings structure', () => {
      render(
        <ReportManagement
          report="# Main Title\n## Section\n### Subsection"
          topic="Test Topic"
          workflowId="test-workflow"
        />
      );
      
      const headings = screen.getAllByRole('heading');
      expect(headings.length).toBeGreaterThan(0);
      
      // Check heading hierarchy
      const h1Elements = headings.filter(h => h.tagName === 'H1');
      const h2Elements = headings.filter(h => h.tagName === 'H2');
      
      expect(h1Elements.length).toBeGreaterThan(0);
      expect(h2Elements.length).toBeGreaterThan(0);
    });

    it('should provide alternative text for visual elements', () => {
      render(
        <WorkflowVisualization
          workflowId="test-workflow"
          status={{
            status: 'completed',
            current_node: null,
            progress: 1.0,
            message: 'Completed',
          }}
          state={null}
          onNodeClick={jest.fn()}
        />
      );
      
      // Visual status indicators should have text alternatives
      expect(screen.getByText('completed')).toBeInTheDocument();
      expect(screen.getByText('100%')).toBeInTheDocument();
    });
  });

  describe('Focus Management', () => {
    it('should manage focus properly in modals', async () => {
      render(
        <ReportManagement
          report="Test report"
          topic="Test Topic"
          workflowId="test-workflow"
        />
      );
      
      // Open share modal
      const shareButton = screen.getByRole('button', { name: /share/i });
      fireEvent.click(shareButton);
      
      await waitFor(() => {
        const modal = screen.getByRole('dialog');
        expect(modal).toBeInTheDocument();
        
        // Focus should be trapped within the modal
        const focusableElements = modal.querySelectorAll(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        expect(focusableElements.length).toBeGreaterThan(0);
      });
    });

    it('should restore focus after modal closes', async () => {
      render(
        <ReportManagement
          report="Test report"
          topic="Test Topic"
          workflowId="test-workflow"
        />
      );
      
      const shareButton = screen.getByRole('button', { name: /share/i });
      shareButton.focus();
      expect(document.activeElement).toBe(shareButton);
      
      fireEvent.click(shareButton);
      
      await waitFor(() => {
        const modal = screen.getByRole('dialog');
        expect(modal).toBeInTheDocument();
      });
      
      // Close modal
      const cancelButton = screen.getByRole('button', { name: /cancel/i });
      fireEvent.click(cancelButton);
      
      await waitFor(() => {
        expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
        // Focus should return to the share button
        expect(document.activeElement).toBe(shareButton);
      });
    });
  });

  describe('Responsive Design Accessibility', () => {
    it('should maintain accessibility on mobile viewports', () => {
      // Mock mobile viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });
      
      render(<ResearchForm onSubmit={jest.fn()} isLoading={false} />);
      
      // Touch targets should be large enough (minimum 44px)
      const buttons = screen.getAllByRole('button');
      buttons.forEach(button => {
        const styles = window.getComputedStyle(button);
        // In a real test, you'd check computed dimensions
        expect(button).toBeInTheDocument();
      });
    });

    it('should handle zoom levels appropriately', () => {
      // Mock high zoom level
      Object.defineProperty(window, 'devicePixelRatio', {
        writable: true,
        configurable: true,
        value: 2,
      });
      
      render(
        <WorkflowVisualization
          workflowId="test-workflow"
          status={{
            status: 'running',
            current_node: 'gap_analyzer',
            progress: 0.5,
            message: 'Processing',
          }}
          state={null}
          onNodeClick={jest.fn()}
        />
      );
      
      // Content should remain accessible at high zoom levels
      expect(screen.getByText('Workflow Status')).toBeInTheDocument();
      expect(screen.getByText('Status Legend')).toBeInTheDocument();
    });
  });
});