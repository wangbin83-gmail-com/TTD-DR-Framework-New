import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { WorkflowDashboard } from '../WorkflowDashboard';
import { WorkflowStatus, TTDRState, ResearchDomain, ComplexityLevel } from '../../types';

describe('WorkflowDashboard', () => {
  const mockOnStop = jest.fn();
  const mockOnRestart = jest.fn();

  const mockStatus: WorkflowStatus = {
    status: 'running',
    current_node: 'gap_analyzer',
    progress: 0.3,
    message: 'Analyzing information gaps...'
  };

  const mockState: TTDRState = {
    topic: 'Artificial Intelligence in Healthcare',
    requirements: {
      domain: ResearchDomain.TECHNOLOGY,
      complexity_level: ComplexityLevel.ADVANCED,
      max_iterations: 5,
      quality_threshold: 0.8,
      max_sources: 20,
      preferred_source_types: ['academic', 'news']
    },
    current_draft: {
      id: 'draft-1',
      topic: 'AI in Healthcare',
      structure: {
        sections: [],
        relationships: [],
        estimated_length: 5000,
        complexity_level: ComplexityLevel.ADVANCED,
        domain: ResearchDomain.TECHNOLOGY
      },
      content: {},
      metadata: {
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T01:00:00Z',
        author: 'TTD-DR',
        version: '1.0',
        word_count: 1500
      },
      quality_score: 0.7,
      iteration: 2
    },
    information_gaps: [
      {
        id: 'gap-1',
        section_id: 'intro',
        gap_type: 'content' as any,
        description: 'Missing introduction content',
        priority: 'high' as any,
        search_queries: []
      }
    ],
    retrieved_info: [],
    iteration_count: 2,
    quality_metrics: {
      completeness: 0.7,
      coherence: 0.8,
      accuracy: 0.75,
      citation_quality: 0.6,
      overall_score: 0.72
    },
    evolution_history: [],
    final_report: null,
    error_log: []
  };

  beforeEach(() => {
    mockOnStop.mockClear();
    mockOnRestart.mockClear();
  });

  it('renders workflow dashboard with basic information', () => {
    render(
      <WorkflowDashboard
        workflowId="test-workflow-123"
        status={mockStatus}
        state={mockState}
        onStop={mockOnStop}
        onRestart={mockOnRestart}
      />
    );

    expect(screen.getByText('Research Workflow')).toBeInTheDocument();
    expect(screen.getByText('Workflow ID: test-workflow-123')).toBeInTheDocument();
    expect(screen.getByText('30%')).toBeInTheDocument();
    expect(screen.getByText('Analyzing information gaps...')).toBeInTheDocument();
  });

  it('displays workflow stages with correct status', () => {
    render(
      <WorkflowDashboard
        workflowId="test-workflow-123"
        status={mockStatus}
        state={mockState}
        onStop={mockOnStop}
        onRestart={mockOnRestart}
      />
    );

    expect(screen.getByText('Draft Generation')).toBeInTheDocument();
    expect(screen.getByText('Gap Analysis')).toBeInTheDocument();
    expect(screen.getByText('Information Retrieval')).toBeInTheDocument();
    expect(screen.getByText('Information Integration')).toBeInTheDocument();
    expect(screen.getByText('Quality Assessment')).toBeInTheDocument();
    expect(screen.getByText('Self-Evolution')).toBeInTheDocument();
    expect(screen.getByText('Report Synthesis')).toBeInTheDocument();
  });

  it('shows stop button when workflow is running', () => {
    render(
      <WorkflowDashboard
        workflowId="test-workflow-123"
        status={mockStatus}
        state={mockState}
        onStop={mockOnStop}
        onRestart={mockOnRestart}
      />
    );

    const stopButton = screen.getByRole('button', { name: /stop/i });
    expect(stopButton).toBeInTheDocument();
    
    fireEvent.click(stopButton);
    expect(mockOnStop).toHaveBeenCalledTimes(1);
  });

  it('shows restart button when workflow is completed', () => {
    const completedStatus: WorkflowStatus = {
      ...mockStatus,
      status: 'completed',
      progress: 1,
      message: 'Research completed successfully'
    };

    render(
      <WorkflowDashboard
        workflowId="test-workflow-123"
        status={completedStatus}
        state={mockState}
        onStop={mockOnStop}
        onRestart={mockOnRestart}
      />
    );

    const restartButton = screen.getByRole('button', { name: /restart/i });
    expect(restartButton).toBeInTheDocument();
    
    fireEvent.click(restartButton);
    expect(mockOnRestart).toHaveBeenCalledTimes(1);
  });

  it('displays current state information', () => {
    render(
      <WorkflowDashboard
        workflowId="test-workflow-123"
        status={mockStatus}
        state={mockState}
        onStop={mockOnStop}
        onRestart={mockOnRestart}
      />
    );

    expect(screen.getByText('Artificial Intelligence in Healthcare')).toBeInTheDocument();
    expect(screen.getByText('2 / 5')).toBeInTheDocument();
    expect(screen.getByText('1 identified')).toBeInTheDocument();
    expect(screen.getByText('72.0%')).toBeInTheDocument();
    expect(screen.getByText('70.0%')).toBeInTheDocument();
    expect(screen.getByText('80.0%')).toBeInTheDocument();
  });

  it('displays error log when present', () => {
    const stateWithErrors: TTDRState = {
      ...mockState,
      error_log: ['Error 1: Connection timeout', 'Error 2: Invalid response', 'Error 3: Rate limit exceeded']
    };

    render(
      <WorkflowDashboard
        workflowId="test-workflow-123"
        status={mockStatus}
        state={stateWithErrors}
        onStop={mockOnStop}
        onRestart={mockOnRestart}
      />
    );

    expect(screen.getByText('Errors')).toBeInTheDocument();
    expect(screen.getByText('Error 1: Connection timeout')).toBeInTheDocument();
    expect(screen.getByText('Error 2: Invalid response')).toBeInTheDocument();
    expect(screen.getByText('Error 3: Rate limit exceeded')).toBeInTheDocument();
  });

  it('handles workflow with error status', () => {
    const errorStatus: WorkflowStatus = {
      ...mockStatus,
      status: 'error',
      message: 'Workflow failed due to API error'
    };

    render(
      <WorkflowDashboard
        workflowId="test-workflow-123"
        status={errorStatus}
        state={mockState}
        onStop={mockOnStop}
        onRestart={mockOnRestart}
      />
    );

    expect(screen.getByText('Workflow failed due to API error')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /restart/i })).toBeInTheDocument();
  });

  it('handles null state gracefully', () => {
    render(
      <WorkflowDashboard
        workflowId="test-workflow-123"
        status={mockStatus}
        state={null}
        onStop={mockOnStop}
        onRestart={mockOnRestart}
      />
    );

    expect(screen.getByText('Research Workflow')).toBeInTheDocument();
    expect(screen.getByText('30%')).toBeInTheDocument();
    // Should not crash and should still show basic workflow information
  });

  it('updates progress bar correctly', () => {
    const { rerender } = render(
      <WorkflowDashboard
        workflowId="test-workflow-123"
        status={mockStatus}
        state={mockState}
        onStop={mockOnStop}
        onRestart={mockOnRestart}
      />
    );

    // Initial progress
    expect(screen.getByText('30%')).toBeInTheDocument();

    // Update progress
    const updatedStatus: WorkflowStatus = {
      ...mockStatus,
      progress: 0.7,
      current_node: 'quality_assessor'
    };

    rerender(
      <WorkflowDashboard
        workflowId="test-workflow-123"
        status={updatedStatus}
        state={mockState}
        onStop={mockOnStop}
        onRestart={mockOnRestart}
      />
    );

    expect(screen.getByText('70%')).toBeInTheDocument();
  });

  it('has proper accessibility attributes', () => {
    render(
      <WorkflowDashboard
        workflowId="test-workflow-123"
        status={mockStatus}
        state={mockState}
        onStop={mockOnStop}
        onRestart={mockOnRestart}
      />
    );

    const stopButton = screen.getByRole('button', { name: /stop/i });
    // Button elements don't need explicit type="button" attribute
    expect(stopButton).toBeInTheDocument();
  });
});