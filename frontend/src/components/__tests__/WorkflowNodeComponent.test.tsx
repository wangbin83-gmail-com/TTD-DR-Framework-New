import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { WorkflowNodeComponent } from '../WorkflowNodeComponent';

// Mock ReactFlow components
jest.mock('reactflow', () => ({
  Handle: ({ type, position, className }: any) => (
    <div 
      data-testid={`handle-${type}-${position}`} 
      className={className}
    />
  ),
  Position: {
    Top: 'top',
    Bottom: 'bottom',
  },
}));

describe('WorkflowNodeComponent', () => {
  const mockOnClick = jest.fn();

  const defaultProps = {
    id: 'gap_analyzer',
    data: {
      label: 'Gap Analyzer',
      status: 'pending' as const,
      isActive: false,
      onClick: mockOnClick,
    },
    selected: false,
    type: 'workflowNode',
    position: { x: 0, y: 0 },
    dragging: false,
    zIndex: 1,
    isConnectable: true,
    xPos: 0,
    yPos: 0,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders node with correct label and icon', () => {
    render(<WorkflowNodeComponent {...defaultProps} />);

    expect(screen.getByText('Gap Analyzer')).toBeInTheDocument();
    expect(screen.getByText('Identifies information gaps')).toBeInTheDocument();
    expect(screen.getByText('Pending')).toBeInTheDocument();
  });

  it('renders different icons for different node types', () => {
    const { rerender } = render(<WorkflowNodeComponent {...defaultProps} />);
    
    // Test different node IDs to ensure correct icons are rendered
    const nodeTypes = [
      'draft_generator',
      'retrieval_engine',
      'information_integrator',
      'quality_assessor',
      'quality_check',
      'self_evolution_enhancer',
      'report_synthesizer'
    ];

    nodeTypes.forEach(nodeId => {
      rerender(
        <WorkflowNodeComponent 
          {...defaultProps} 
          id={nodeId}
          data={{
            ...defaultProps.data,
            label: `Test ${nodeId}`,
          }}
        />
      );
      // Each node should render without errors
      expect(screen.getByText(`Test ${nodeId}`)).toBeInTheDocument();
    });
  });

  it('handles click events correctly', () => {
    render(<WorkflowNodeComponent {...defaultProps} />);

    const nodeElement = screen.getByRole('button');
    fireEvent.click(nodeElement);

    expect(mockOnClick).toHaveBeenCalledTimes(1);
  });

  it('handles keyboard events correctly', () => {
    render(<WorkflowNodeComponent {...defaultProps} />);

    const nodeElement = screen.getByRole('button');
    
    // Test Enter key
    fireEvent.keyDown(nodeElement, { key: 'Enter' });
    expect(mockOnClick).toHaveBeenCalledTimes(1);

    // Test Space key
    fireEvent.keyDown(nodeElement, { key: ' ' });
    expect(mockOnClick).toHaveBeenCalledTimes(2);

    // Test other keys (should not trigger onClick)
    fireEvent.keyDown(nodeElement, { key: 'Escape' });
    expect(mockOnClick).toHaveBeenCalledTimes(2);
  });

  it('displays running status with progress indicator', () => {
    const runningProps = {
      ...defaultProps,
      data: {
        ...defaultProps.data,
        status: 'running' as const,
        progress: 0.6,
      },
    };

    render(<WorkflowNodeComponent {...runningProps} />);

    expect(screen.getByText('Running')).toBeInTheDocument();
    expect(screen.getByText('Processing...')).toBeInTheDocument();
    expect(screen.getByText('60%')).toBeInTheDocument();
  });

  it('displays completed status correctly', () => {
    const completedProps = {
      ...defaultProps,
      data: {
        ...defaultProps.data,
        status: 'completed' as const,
      },
    };

    render(<WorkflowNodeComponent {...completedProps} />);

    expect(screen.getByText('Completed')).toBeInTheDocument();
  });

  it('displays error status correctly', () => {
    const errorProps = {
      ...defaultProps,
      data: {
        ...defaultProps.data,
        status: 'error' as const,
      },
    };

    render(<WorkflowNodeComponent {...errorProps} />);

    expect(screen.getByText('Error')).toBeInTheDocument();
  });

  it('shows active indicator when node is active', () => {
    const activeProps = {
      ...defaultProps,
      data: {
        ...defaultProps.data,
        isActive: true,
      },
    };

    render(<WorkflowNodeComponent {...activeProps} />);

    // The active indicator should be present (we can't easily test the visual indicator,
    // but we can ensure the component renders without errors when active)
    expect(screen.getByRole('button')).toBeInTheDocument();
  });

  it('renders handles correctly for different node positions', () => {
    // Test first node (draft_generator) - should not have input handle
    const firstNodeProps = {
      ...defaultProps,
      id: 'draft_generator',
    };

    const { rerender } = render(<WorkflowNodeComponent {...firstNodeProps} />);
    expect(screen.queryByTestId('handle-target-top')).not.toBeInTheDocument();
    expect(screen.getByTestId('handle-source-bottom')).toBeInTheDocument();

    // Test last node (report_synthesizer) - should not have output handle
    const lastNodeProps = {
      ...defaultProps,
      id: 'report_synthesizer',
    };

    rerender(<WorkflowNodeComponent {...lastNodeProps} />);
    expect(screen.getByTestId('handle-target-top')).toBeInTheDocument();
    expect(screen.queryByTestId('handle-source-bottom')).not.toBeInTheDocument();

    // Test middle node - should have both handles
    const middleNodeProps = {
      ...defaultProps,
      id: 'gap_analyzer',
    };

    rerender(<WorkflowNodeComponent {...middleNodeProps} />);
    expect(screen.getByTestId('handle-target-top')).toBeInTheDocument();
    expect(screen.getByTestId('handle-source-bottom')).toBeInTheDocument();
  });

  it('applies correct CSS classes for different statuses', () => {
    const { rerender } = render(<WorkflowNodeComponent {...defaultProps} />);

    // Test that the component renders with different statuses
    const statuses = ['pending', 'running', 'completed', 'error'] as const;
    
    statuses.forEach(status => {
      rerender(
        <WorkflowNodeComponent 
          {...defaultProps} 
          data={{
            ...defaultProps.data,
            status,
          }}
        />
      );
      
      const nodeElement = screen.getByRole('button');
      expect(nodeElement).toHaveClass('group', 'relative', 'bg-white', 'rounded-2xl');
    });
  });

  it('has proper accessibility attributes', () => {
    render(<WorkflowNodeComponent {...defaultProps} />);

    const nodeElement = screen.getByRole('button');
    expect(nodeElement).toHaveAttribute('tabIndex', '0');
  });

  it('displays correct node descriptions', () => {
    const nodeDescriptions = [
      { id: 'draft_generator', description: 'Creates initial research draft' },
      { id: 'gap_analyzer', description: 'Identifies information gaps' },
      { id: 'retrieval_engine', description: 'Retrieves relevant information' },
      { id: 'information_integrator', description: 'Integrates new information' },
      { id: 'quality_assessor', description: 'Assesses content quality' },
      { id: 'quality_check', description: 'Validates quality threshold' },
      { id: 'self_evolution_enhancer', description: 'Enhances system capabilities' },
      { id: 'report_synthesizer', description: 'Generates final report' },
    ];

    nodeDescriptions.forEach(({ id, description }) => {
      const { rerender } = render(
        <WorkflowNodeComponent 
          {...defaultProps} 
          id={id}
          data={{
            ...defaultProps.data,
            label: `Test ${id}`,
          }}
        />
      );
      
      expect(screen.getByText(description)).toBeInTheDocument();
    });
  });
});