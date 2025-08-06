import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { ResearchForm } from '../ResearchForm';
import { ResearchDomain, ComplexityLevel } from '../../types';

describe('ResearchForm', () => {
  const mockOnSubmit = jest.fn();

  beforeEach(() => {
    mockOnSubmit.mockClear();
  });

  it('renders form with all required fields', () => {
    render(<ResearchForm onSubmit={mockOnSubmit} isLoading={false} />);

    expect(screen.getByLabelText(/research topic/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/research domain/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/complexity level/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/max iterations/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/max sources/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/quality threshold/i)).toBeInTheDocument();
    expect(screen.getByText(/preferred source types/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /start research/i })).toBeInTheDocument();
  });

  it('shows validation error for empty topic', async () => {
    const user = userEvent.setup();
    render(<ResearchForm onSubmit={mockOnSubmit} isLoading={false} />);

    const submitButton = screen.getByRole('button', { name: /start research/i });
    await user.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText(/research topic is required/i)).toBeInTheDocument();
    });

    expect(mockOnSubmit).not.toHaveBeenCalled();
  });

  it('shows validation error for topic too short', async () => {
    const user = userEvent.setup();
    render(<ResearchForm onSubmit={mockOnSubmit} isLoading={false} />);

    const topicInput = screen.getByLabelText(/research topic/i);
    await user.type(topicInput, 'AI');

    const submitButton = screen.getByRole('button', { name: /start research/i });
    await user.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText(/topic must be at least 10 characters long/i)).toBeInTheDocument();
    });

    expect(mockOnSubmit).not.toHaveBeenCalled();
  });

  it('validates max iterations range', async () => {
    const user = userEvent.setup();
    render(<ResearchForm onSubmit={mockOnSubmit} isLoading={false} />);

    const maxIterationsInput = screen.getByLabelText(/max iterations/i);
    
    // Test minimum validation
    await user.clear(maxIterationsInput);
    await user.type(maxIterationsInput, '0');
    
    const submitButton = screen.getByRole('button', { name: /start research/i });
    await user.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText(/must be at least 1/i)).toBeInTheDocument();
    });

    // Test maximum validation
    await user.clear(maxIterationsInput);
    await user.type(maxIterationsInput, '25');
    await user.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText(/cannot exceed 20/i)).toBeInTheDocument();
    });
  });

  it('validates max sources range', async () => {
    const user = userEvent.setup();
    render(<ResearchForm onSubmit={mockOnSubmit} isLoading={false} />);

    const maxSourcesInput = screen.getByLabelText(/max sources/i);
    
    // Test minimum validation
    await user.clear(maxSourcesInput);
    await user.type(maxSourcesInput, '3');
    
    const submitButton = screen.getByRole('button', { name: /start research/i });
    await user.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText(/must be at least 5/i)).toBeInTheDocument();
    });

    // Test maximum validation
    await user.clear(maxSourcesInput);
    await user.type(maxSourcesInput, '150');
    await user.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText(/cannot exceed 100/i)).toBeInTheDocument();
    });
  });

  it('handles source type selection', async () => {
    const user = userEvent.setup();
    render(<ResearchForm onSubmit={mockOnSubmit} isLoading={false} />);

    const academicCheckbox = screen.getByLabelText(/academic papers/i);
    const blogCheckbox = screen.getByLabelText(/blog posts/i);

    // Academic should be checked by default
    expect(academicCheckbox).toBeChecked();
    expect(blogCheckbox).not.toBeChecked();

    // Uncheck academic and check blog
    await user.click(academicCheckbox);
    await user.click(blogCheckbox);

    expect(academicCheckbox).not.toBeChecked();
    expect(blogCheckbox).toBeChecked();
  });

  it('submits form with correct data', async () => {
    const user = userEvent.setup();
    render(<ResearchForm onSubmit={mockOnSubmit} isLoading={false} />);

    const topicInput = screen.getByLabelText(/research topic/i);
    const domainSelect = screen.getByLabelText(/research domain/i);
    const complexitySelect = screen.getByLabelText(/complexity level/i);

    await user.type(topicInput, 'Artificial Intelligence in Healthcare');
    await user.selectOptions(domainSelect, ResearchDomain.TECHNOLOGY);
    await user.selectOptions(complexitySelect, ComplexityLevel.ADVANCED);

    const submitButton = screen.getByRole('button', { name: /start research/i });
    await user.click(submitButton);

    await waitFor(() => {
      expect(mockOnSubmit).toHaveBeenCalledWith(
        'Artificial Intelligence in Healthcare',
        expect.objectContaining({
          domain: ResearchDomain.TECHNOLOGY,
          complexity_level: ComplexityLevel.ADVANCED,
          max_iterations: 5,
          quality_threshold: 0.8,
          max_sources: 20,
          preferred_source_types: expect.arrayContaining(['academic', 'news', 'official'])
        })
      );
    });
  });

  it('disables form when loading', () => {
    render(<ResearchForm onSubmit={mockOnSubmit} isLoading={true} />);

    const topicInput = screen.getByLabelText(/research topic/i);
    const submitButton = screen.getByRole('button', { name: /starting research/i });

    expect(topicInput).toBeDisabled();
    expect(submitButton).toBeDisabled();
    expect(submitButton).toHaveTextContent('Starting Research...');
  });

  it('updates quality threshold display', async () => {
    const user = userEvent.setup();
    render(<ResearchForm onSubmit={mockOnSubmit} isLoading={false} />);

    const qualitySlider = screen.getByLabelText(/quality threshold/i);
    
    // For range inputs, we need to use fireEvent.change instead of user.type
    fireEvent.change(qualitySlider, { target: { value: '0.9' } });

    expect(screen.getByText(/quality threshold: 0\.9/i)).toBeInTheDocument();
  });

  it('has proper accessibility attributes', () => {
    render(<ResearchForm onSubmit={mockOnSubmit} isLoading={false} />);

    const topicInput = screen.getByLabelText(/research topic/i);
    const submitButton = screen.getByRole('button', { name: /start research/i });

    expect(topicInput).toHaveAttribute('aria-required', 'true');
    expect(submitButton).toHaveAttribute('type', 'submit');
  });
});