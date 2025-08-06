import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { WorkflowNodeDetails } from '../WorkflowNodeDetails';
import { TTDRState, ResearchDomain, ComplexityLevel } from '../../types';

describe('WorkflowNodeDetails', () => {
  const mockOnClose = jest.fn();

  const mockState: TTDRState = {
    topic: 'Test Research Topic',
    requirements: {
      domain: ResearchDomain.TECHNOLOGY,
      complexity_level: ComplexityLevel.INTERMEDIATE,
      max_iterations: 5,
      quality_threshold: 0.8,
      max_sources: 20,
      preferred_source_types: ['academic', 'news']
    },
    current_draft: {
      id: 'draft-1',
      topic: 'Test Research Topic',
      structure: {
        sections: [
          { id: '1', title: 'Introduction', content: '', subsections: [], estimated_length: 200 },
          { id: '2', title: 'Background', content: '', subsections: [], estimated_length: 300 }
        ],
        relationships: [],
        estimated_length: 1000,
        complexity_level: ComplexityLevel.INTERMEDIATE,
        domain: ResearchDomain.TECHNOLOGY
      },
      content: {},
      metadata: {
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T01:00:00Z',
        author: 'test-user',
        version: '1.0',
        word_count: 500
      },
      quality_score: 0.7,
      iteration: 1
    },
    information_gaps: [
      {
        id: 'gap-1',
        section_id: 'section-1',
        gap_type: 'CONTENT' as any,
        description: 'Missing technical details',
        priority: 'HIGH' as any,
        search_queries: []
      }
    ],
    retrieved_info: [
      {
        source: {
          url: 'https://example.com',
          title: 'Test Source',
          domain: 'example.com',
          credibility_score: 0.8,
          last_accessed: '2024-01-01T00:00:00Z'
        },
        content: 'Test content',
        relevance_score: 0.9,
        credibility_score: 0.8,
        extraction_timestamp: new Date()
      }
    ],
    iteration_count: 2,
    quality_metrics: {
      completeness: 0.6,
      coherence: 0.8,
      accuracy: 0.7,
      citation_quality: 0.5,
      overall_score: 0.65
    },
    evolution_history: [
      { component: 'draft_generator', improvement: 'Enhanced structure' }
    ],
    final_report: null,
    error_log: []
  };

  const defaultProps = {
    nodeId: 'draft_generator',
    nodeName: 'Draft Generator',
    status: 'completed' as const,
    state: mockState,
    onClose: mockOnClose
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders modal with correct node name and status', () => {
    render(<WorkflowNodeDetails {...defaultProps} />);
    
    expect(screen.getByText('Draft Generator')).toBeInTheDocument();
    expect(screen.getByText('completed')).toBeInTheDocument();
  });

  it('displays node description', () => {
    render(<WorkflowNodeDetails {...defaultProps} />);
    
    expect(screen.getByText(/Creates the initial research draft/)).toBeInTheDocument();
  });

  it('calls onClose when close button is clicked', () => {
    render(<WorkflowNodeDetails {...defaultProps} />);
    
    const closeButton = screen.getByText('Close');
    fireEvent.click(closeButton);
    
    expect(mockOnClose).toHaveBeenCalledTimes(1);
  });

  it('calls onClose when X button is clicked', () => {
    render(<WorkflowNodeDetails {...defaultProps} />);
    
    const xButton = screen.getByRole('button', { name: /close/i });
    fireEvent.click(xButton);
    
    expect(mockOnClose).toHaveBeenCalledTimes(1);
  });

  it('displays execution details for draft_generator node', () => {
    render(<WorkflowNodeDetails {...defaultProps} />);
    
    expect(screen.getByText('Execution Details')).toBeInTheDocument();
    expect(screen.getByText('Input')).toBeInTheDocument();
    expect(screen.getByText('Output')).toBeInTheDocument();
    expect(screen.getByText('Details')).toBeInTheDocument();
    
    expect(screen.getByText('Topic: Test Research Topic')).toBeInTheDocument();
    expect(screen.getByText('Draft created (500 words)')).toBeInTheDocument();
  });

  it('displays execution details for gap_analyzer node', () => {
    const gapAnalyzerProps = {
      ...defaultProps,
      nodeId: 'gap_analyzer',
      nodeName: 'Gap Analyzer'
    };

    render(<WorkflowNodeDetails {...gapAnalyzerProps} />);
    
    expect(screen.getByText('1 gaps identified')).toBeInTheDocument();
    expect(screen.getByText(/CONTENT: Missing technical details/)).toBeInTheDocument();
  });

  it('displays execution details for quality_assessor node', () => {
    const qualityAssessorProps = {
      ...defaultProps,
      nodeId: 'quality_assessor',
      nodeName: 'Quality Assessor'
    };

    render(<WorkflowNodeDetails {...qualityAssessorProps} />);
    
    expect(screen.getByText('Quality metrics calculated')).toBeInTheDocument();
    expect(screen.getByText('Completeness: 60.0%')).toBeInTheDocument();
    expect(screen.getByText('Coherence: 80.0%')).toBeInTheDocument();
    expect(screen.getByText('Overall Score: 65.0%')).toBeInTheDocument();
  });

  it('displays execution details for retrieval_engine node', () => {
    const retrievalEngineProps = {
      ...defaultProps,
      nodeId: 'retrieval_engine',
      nodeName: 'Retrieval Engine'
    };

    render(<WorkflowNodeDetails {...retrievalEngineProps} />);
    
    expect(screen.getByText('1 sources retrieved')).toBeInTheDocument();
    expect(screen.getByText('Source 1: Test Source')).toBeInTheDocument();
  });

  it('displays correct status icon for running status', () => {
    const runningProps = {
      ...defaultProps,
      status: 'running' as const
    };

    render(<WorkflowNodeDetails {...runningProps} />);
    
    expect(screen.getByText('running')).toBeInTheDocument();
    expect(screen.getByText('Node is currently executing...')).toBeInTheDocument();
  });

  it('displays correct status icon for error status', () => {
    const errorProps = {
      ...defaultProps,
      status: 'error' as const
    };

    render(<WorkflowNodeDetails {...errorProps} />);
    
    expect(screen.getByText('error')).toBeInTheDocument();
  });

  it('displays error log when present', () => {
    const stateWithErrors = {
      ...mockState,
      error_log: ['Error 1: Connection failed', 'Error 2: Timeout occurred']
    };

    const propsWithErrors = {
      ...defaultProps,
      state: stateWithErrors
    };

    render(<WorkflowNodeDetails {...propsWithErrors} />);
    
    expect(screen.getByText('Error Log')).toBeInTheDocument();
    expect(screen.getByText('Error 1: Connection failed')).toBeInTheDocument();
    expect(screen.getByText('Error 2: Timeout occurred')).toBeInTheDocument();
  });

  it('displays completion message for completed status', () => {
    render(<WorkflowNodeDetails {...defaultProps} />);
    
    expect(screen.getByText('Node completed successfully')).toBeInTheDocument();
  });

  it('handles null state gracefully', () => {
    const propsWithNullState = {
      ...defaultProps,
      state: null
    };

    render(<WorkflowNodeDetails {...propsWithNullState} />);
    
    expect(screen.getByText('Draft Generator')).toBeInTheDocument();
    expect(screen.getByText(/Creates the initial research draft/)).toBeInTheDocument();
  });

  it('displays different descriptions for different node types', () => {
    const nodeDescriptions = [
      { nodeId: 'draft_generator', description: 'Creates the initial research draft' },
      { nodeId: 'gap_analyzer', description: 'Identifies specific areas in the draft' },
      { nodeId: 'retrieval_engine', description: 'Retrieves relevant information from external sources' },
      { nodeId: 'information_integrator', description: 'Seamlessly incorporates retrieved information' },
      { nodeId: 'quality_assessor', description: 'Evaluates draft quality' },
      { nodeId: 'quality_check', description: 'Decision node that determines whether to continue' },
      { nodeId: 'self_evolution_enhancer', description: 'Applies self-improvement algorithms' },
      { nodeId: 'report_synthesizer', description: 'Generates the final polished research report' }
    ];

    nodeDescriptions.forEach(({ nodeId, description }) => {
      const props = {
        ...defaultProps,
        nodeId,
        nodeName: nodeId.replace('_', ' ')
      };

      const { unmount } = render(<WorkflowNodeDetails {...props} />);
      
      expect(screen.getByText(new RegExp(description))).toBeInTheDocument();
      
      unmount();
    });
  });

  it('displays quality check decision logic', () => {
    const qualityCheckProps = {
      ...defaultProps,
      nodeId: 'quality_check',
      nodeName: 'Quality Check'
    };

    render(<WorkflowNodeDetails {...qualityCheckProps} />);
    
    expect(screen.getByText('Quality threshold: 80.0%')).toBeInTheDocument();
    expect(screen.getByText('Current score: 65.0%')).toBeInTheDocument();
    expect(screen.getByText('Continue iteration')).toBeInTheDocument();
  });

  it('displays evolution history for self_evolution_enhancer', () => {
    const evolutionProps = {
      ...defaultProps,
      nodeId: 'self_evolution_enhancer',
      nodeName: 'Self Evolution Enhancer'
    };

    render(<WorkflowNodeDetails {...evolutionProps} />);
    
    expect(screen.getByText('1 evolution records')).toBeInTheDocument();
    expect(screen.getByText(/Evolution 1: draft_generator enhanced/)).toBeInTheDocument();
  });

  it('displays final report information for report_synthesizer', () => {
    const stateWithReport = {
      ...mockState,
      final_report: 'This is the final research report content...'
    };

    const reportProps = {
      ...defaultProps,
      nodeId: 'report_synthesizer',
      nodeName: 'Report Synthesizer',
      state: stateWithReport
    };

    render(<WorkflowNodeDetails {...reportProps} />);
    
    expect(screen.getByText('Final report generated')).toBeInTheDocument();
    expect(screen.getByText(/Report length: \d+ characters/)).toBeInTheDocument();
  });

  it('has proper modal styling and overlay', () => {
    render(<WorkflowNodeDetails {...defaultProps} />);
    
    const overlay = screen.getByText('Draft Generator').closest('.fixed');
    expect(overlay).toHaveClass('inset-0', 'bg-black', 'bg-opacity-50');
    
    const modal = screen.getByText('Draft Generator').closest('.bg-white');
    expect(modal).toHaveClass('rounded-lg', 'shadow-xl');
  });
});