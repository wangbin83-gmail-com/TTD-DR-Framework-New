import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { AppProviders } from '../AppProviders';
import { ResearchForm } from '../ResearchForm';
import { ReportManagement } from '../ReportManagement';

// Mock the hooks that aren't available in test environment
jest.mock('react-hook-form', () => ({
  useForm: () => ({
    register: jest.fn(),
    handleSubmit: (fn: any) => (e: any) => {
      e.preventDefault();
      fn({
        topic: 'Test topic for research',
        domain: 'general',
        complexity_level: 'intermediate',
        max_iterations: 5,
        quality_threshold: 0.8,
        max_sources: 20,
        preferred_source_types: ['academic', 'news']
      });
    },
    formState: { errors: {} },
    watch: jest.fn(() => ['academic', 'news']),
    setValue: jest.fn()
  })
}));

// Mock file-saver
jest.mock('file-saver', () => ({
  saveAs: jest.fn()
}));

// Mock jsPDF and html2canvas
jest.mock('jspdf', () => {
  return jest.fn().mockImplementation(() => ({
    addImage: jest.fn(),
    addPage: jest.fn(),
    save: jest.fn()
  }));
});

jest.mock('html2canvas', () => {
  return jest.fn().mockResolvedValue({
    toDataURL: jest.fn(() => 'data:image/png;base64,test'),
    height: 800,
    width: 600
  });
});

describe('Error Handling Integration', () => {
  it('should render ResearchForm with error handling providers', () => {
    const mockOnSubmit = jest.fn();
    
    render(
      <AppProviders>
        <ResearchForm onSubmit={mockOnSubmit} isLoading={false} />
      </AppProviders>
    );

    expect(screen.getByText('Start New Research')).toBeInTheDocument();
    expect(screen.getByLabelText(/research topic/i)).toBeInTheDocument();
  });

  it('should render ReportManagement with error handling providers', () => {
    const mockReport = '# Test Report\n\nThis is a test report.';
    
    render(
      <AppProviders>
        <ReportManagement
          report={mockReport}
          topic="Test Topic"
          workflowId="test-123"
        />
      </AppProviders>
    );

    expect(screen.getByText('Research Report Management')).toBeInTheDocument();
    expect(screen.getByText('Test Topic')).toBeInTheDocument();
  });

  it('should show notifications when form validation fails', async () => {
    const mockOnSubmit = jest.fn();
    
    render(
      <AppProviders>
        <ResearchForm onSubmit={mockOnSubmit} isLoading={false} />
      </AppProviders>
    );

    // Try to submit form without filling required fields
    const submitButton = screen.getByRole('button', { name: /start research/i });
    fireEvent.click(submitButton);

    // Should show validation notification
    await waitFor(() => {
      expect(screen.getByText('Validation Error')).toBeInTheDocument();
    });
  });

  it('should show loading states during export operations', async () => {
    const mockReport = '# Test Report\n\nThis is a test report.';
    
    render(
      <AppProviders>
        <ReportManagement
          report={mockReport}
          topic="Test Topic"
          workflowId="test-123"
        />
      </AppProviders>
    );

    // Click export markdown button
    const exportButton = screen.getByText('Export Markdown');
    fireEvent.click(exportButton);

    // Should show success notification
    await waitFor(() => {
      expect(screen.getByText('Export Complete')).toBeInTheDocument();
    });
  });

  it('should handle errors gracefully with error boundary', () => {
    // Component that throws an error
    const ErrorComponent = () => {
      throw new Error('Test error');
    };

    render(
      <AppProviders>
        <ErrorComponent />
      </AppProviders>
    );

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    expect(screen.getByText('An unexpected error occurred in the application')).toBeInTheDocument();
  });

  it('should provide retry functionality in error boundary', () => {
    let shouldThrow = true;
    
    const ConditionalErrorComponent = () => {
      if (shouldThrow) {
        throw new Error('Test error');
      }
      return <div>Component working</div>;
    };

    const { rerender } = render(
      <AppProviders>
        <ConditionalErrorComponent />
      </AppProviders>
    );

    // Should show error boundary
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();

    // Click retry button
    const retryButton = screen.getByText('Try Again');
    fireEvent.click(retryButton);

    // Change the condition and rerender
    shouldThrow = false;
    rerender(
      <AppProviders>
        <ConditionalErrorComponent />
      </AppProviders>
    );

    // Should show working component
    expect(screen.getByText('Component working')).toBeInTheDocument();
  });
});