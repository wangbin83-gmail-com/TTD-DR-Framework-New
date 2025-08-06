import React from 'react';
import { render, screen, fireEvent, waitFor } from '../../test-utils';
import '@testing-library/jest-dom';
import { WorkflowVisualization } from '../WorkflowVisualization';
import { WorkflowStatus, TTDRState } from '../../types';

// Mock ReactFlow
jest.mock('reactflow', () => ({
  __esModule: true,
  default: ({ children, ...props }: any) => (
    <div data-testid="react-flow" {...props}>
      {children}
    </div>
  ),
  useNodesState: () => [[], jest.fn(), jest.fn()],
  useEdgesState: () => [[], jest.fn(), jest.fn()],
  addEdge: jest.fn(),
  Controls: () => <div data-testid="controls" />,
  MiniMap: () => <div data-testid="minimap" />,
  Background: () => <div data-testid="background" />,
  BackgroundVariant: { Dots: 'dots' },
  Panel: ({ children, position, className }: any) => (
    <div data-testid={`panel-${position}`} className={className}>
      {children}
    </div>
  ),
  Handle: ({ type, position }: any) => (
    <div data-testid={`handle-${type}-${position}`} />
  ),
  Position: {
    Top: 'top',
    Bottom: 'bottom',
    Left: 'left',
    Right: 'right',
  },
  MarkerType: {
    ArrowClosed: 'arrowclosed',
  },
}));

// Mock WorkflowNodeDetails component
jest.mock('../WorkflowNodeDetails', () => ({
  WorkflowNodeDetails: ({ onClose }: any) => (
    <div data-testid="workflow-node-details">
      <button onClick={onClose} data-testid="close-details">
        Close
      </button>
    </div>
  ),
}));

describe('WorkflowVisualization', () => {
  const mockWorkflowStatus: WorkflowStatus = {
    status: 'running',
    current_node: 'gap_analyzer',
    progress: 0.3,
    message: 'Analyzing information gaps...',
  };

  const mockTTDRState: TTDRState = {
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

  const defaultProps = {
    workflowId: 'test-workflow',
    status: mockWorkflowStatus,
    state: mockTTDRState,
    onNodeClick: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders the workflow visualization with enhanced status panel', () => {
    render(<WorkflowVisualization {...defaultProps} />);

    // Check if ReactFlow is rendered
    expect(screen.getByTestId('react-flow')).toBeInTheDocument();

    // Check if enhanced status panel is rendered
    expect(screen.getByTestId('panel-top-right')).toBeInTheDocument();
    expect(screen.getByText('Workflow Status')).toBeInTheDocument();
    expect(screen.getByText('running')).toBeInTheDocument();
    expect(screen.getByText('30%')).toBeInTheDocument(); // Progress percentage
    expect(screen.getByText('Gap Analyzer')).toBeInTheDocument(); // Current node
  });

  it('renders the enhanced legend panel', () => {
    render(<WorkflowVisualization {...defaultProps} />);

    const legendPanel = screen.getByTestId('panel-bottom-right');
    expect(legendPanel).toBeInTheDocument();
    expect(screen.getByText('Status Legend')).toBeInTheDocument();
    expect(screen.getByText('Pending')).toBeInTheDocument();
    expect(screen.getByText('Running')).toBeInTheDocument();
    expect(screen.getByText('Completed')).toBeInTheDocument();
    expect(screen.getByText('Error')).toBeInTheDocument();
  });

  it('displays iteration count when state is provided', () => {
    render(<WorkflowVisualization {...defaultProps} />);

    expect(screen.getByText('2')).toBeInTheDocument(); // Iteration count
    expect(screen.getByText('Iterations')).toBeInTheDocument();
  });

  it('displays status message in enhanced format', () => {
    render(<WorkflowVisualization {...defaultProps} />);

    expect(screen.getByText('Status Message')).toBeInTheDocument();
    expect(screen.getByText('Analyzing information gaps...')).toBeInTheDocument();
  });

  it('handles different workflow statuses correctly', () => {
    const completedStatus: WorkflowStatus = {
      status: 'completed',
      current_node: null,
      progress: 1.0,
      message: 'Workflow completed successfully',
    };

    render(<WorkflowVisualization {...defaultProps} status={completedStatus} />);

    expect(screen.getByText('completed')).toBeInTheDocument();
    expect(screen.getByText('100%')).toBeInTheDocument();
  });

  it('handles error status correctly', () => {
    const errorStatus: WorkflowStatus = {
      status: 'error',
      current_node: 'retrieval_engine',
      progress: 0.5,
      message: 'Error occurred during retrieval',
    };

    render(<WorkflowVisualization {...defaultProps} status={errorStatus} />);

    expect(screen.getByText('error')).toBeInTheDocument();
    expect(screen.getByText('Error occurred during retrieval')).toBeInTheDocument();
  });

  it('shows current node information when available', () => {
    render(<WorkflowVisualization {...defaultProps} />);

    expect(screen.getByText('Current Node')).toBeInTheDocument();
    expect(screen.getByText('Gap Analyzer')).toBeInTheDocument();
  });

  it('handles workflow without current node', () => {
    const idleStatus: WorkflowStatus = {
      status: 'idle',
      current_node: null,
      progress: 0,
      message: 'Workflow is idle',
    };

    render(<WorkflowVisualization {...defaultProps} status={idleStatus} />);

    expect(screen.queryByText('Current Node')).not.toBeInTheDocument();
  });

  it('displays active nodes count correctly', () => {
    render(<WorkflowVisualization {...defaultProps} />);

    expect(screen.getByText('1')).toBeInTheDocument(); // Active nodes count
    expect(screen.getByText('Active Nodes')).toBeInTheDocument();
  });

  it('handles workflow without state gracefully', () => {
    render(<WorkflowVisualization {...defaultProps} state={null} />);

    // Should still render the main components
    expect(screen.getByText('Workflow Status')).toBeInTheDocument();
    expect(screen.getByText('Status Legend')).toBeInTheDocument();
    
    // Should not show iteration count
    expect(screen.queryByText('Iterations')).not.toBeInTheDocument();
  });

  it('applies correct responsive classes', () => {
    render(<WorkflowVisualization {...defaultProps} />);

    const statusPanel = screen.getByTestId('panel-top-right');
    expect(statusPanel).toHaveClass('min-w-[320px]', 'max-w-[400px]');

    const legendPanel = screen.getByTestId('panel-bottom-right');
    expect(legendPanel).toHaveClass('min-w-[200px]');
  });

  it('has proper accessibility attributes', () => {
    render(<WorkflowVisualization {...defaultProps} />);

    const reactFlow = screen.getByTestId('react-flow');
    expect(reactFlow).toHaveAttribute('nodesDraggable', 'false');
    expect(reactFlow).toHaveAttribute('nodesConnectable', 'false');
    expect(reactFlow).toHaveAttribute('elementsSelectable', 'true');
  });
});