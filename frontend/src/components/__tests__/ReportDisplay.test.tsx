import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ReportDisplay } from '../ReportDisplay';

// Mock the file-saver library
jest.mock('file-saver', () => ({
  saveAs: jest.fn()
}));

// Mock jsPDF
jest.mock('jspdf', () => {
  return jest.fn().mockImplementation(() => ({
    addImage: jest.fn(),
    addPage: jest.fn(),
    save: jest.fn()
  }));
});

// Mock html2canvas
jest.mock('html2canvas', () => {
  return jest.fn().mockResolvedValue({
    toDataURL: jest.fn().mockReturnValue('data:image/png;base64,mock-image-data'),
    height: 1000,
    width: 800
  });
});

// Mock navigator.clipboard
Object.assign(navigator, {
  clipboard: {
    writeText: jest.fn().mockResolvedValue(undefined)
  }
});

// Mock navigator.share
Object.assign(navigator, {
  share: jest.fn().mockResolvedValue(undefined)
});

describe('ReportDisplay', () => {
  const mockReport = `# Research Report: AI in Healthcare

## Introduction

Artificial Intelligence (AI) has revolutionized healthcare by providing innovative solutions for diagnosis, treatment, and patient care.

## Key Findings

### Machine Learning Applications

- **Diagnostic Imaging**: AI algorithms can detect anomalies in medical images with high accuracy.
- **Drug Discovery**: ML models accelerate the identification of potential therapeutic compounds.

### Challenges

1. Data privacy concerns
2. Regulatory compliance
3. Integration with existing systems

## Conclusion

AI continues to transform healthcare, offering unprecedented opportunities for improving patient outcomes.

[Reference 1](https://example.com/ref1)
[Reference 2](https://example.com/ref2)
`;

  const defaultProps = {
    report: mockReport,
    topic: 'AI in Healthcare',
    workflowId: 'test-workflow-123'
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders report display with header information', () => {
    render(<ReportDisplay {...defaultProps} />);

    expect(screen.getByText('Research Report')).toBeInTheDocument();
    expect(screen.getByText('AI in Healthcare')).toBeInTheDocument();
    expect(screen.getByText('Workflow ID: test-workflow-123')).toBeInTheDocument();
  });

  it('displays export buttons', () => {
    render(<ReportDisplay {...defaultProps} />);

    expect(screen.getByRole('button', { name: /export pdf/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /export word/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /export markdown/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /share/i })).toBeInTheDocument();
  });

  it('renders formatted report content by default', () => {
    render(<ReportDisplay {...defaultProps} />);

    expect(screen.getByText('Research Report: AI in Healthcare')).toBeInTheDocument();
    expect(screen.getByText('Introduction')).toBeInTheDocument();
    expect(screen.getByText('Key Findings')).toBeInTheDocument();
    expect(screen.getByText('Machine Learning Applications')).toBeInTheDocument();
    expect(screen.getByText('Conclusion')).toBeInTheDocument();
  });

  it('toggles between formatted and raw markdown view', () => {
    render(<ReportDisplay {...defaultProps} />);

    const toggleButton = screen.getByTitle(/show raw markdown/i);
    fireEvent.click(toggleButton);

    expect(screen.getByText('Raw Markdown')).toBeInTheDocument();
    expect(screen.getByText(/# Research Report: AI in Healthcare/)).toBeInTheDocument();

    fireEvent.click(toggleButton);
    expect(screen.queryByText('Raw Markdown')).not.toBeInTheDocument();
  });

  it('displays report statistics', () => {
    render(<ReportDisplay {...defaultProps} />);

    expect(screen.getByText('Words')).toBeInTheDocument();
    expect(screen.getByText('Characters')).toBeInTheDocument();
    expect(screen.getByText('Sections')).toBeInTheDocument();
    expect(screen.getByText('References')).toBeInTheDocument();

    // Check if statistics are calculated correctly
    const wordCount = mockReport.split(' ').length;
    expect(screen.getByText(wordCount.toLocaleString())).toBeInTheDocument();
    
    const charCount = mockReport.length;
    expect(screen.getByText(charCount.toLocaleString())).toBeInTheDocument();
  });

  it('exports to PDF when PDF button is clicked', async () => {
    const html2canvas = require('html2canvas');
    const jsPDF = require('jspdf');
    
    render(<ReportDisplay {...defaultProps} />);

    const pdfButton = screen.getByRole('button', { name: /export pdf/i });
    fireEvent.click(pdfButton);

    // Wait for the async operation to complete
    await waitFor(() => {
      expect(html2canvas).toHaveBeenCalled();
    }, { timeout: 3000 });

    expect(jsPDF).toHaveBeenCalled();
  });

  it('exports to markdown when markdown button is clicked', () => {
    const { saveAs } = require('file-saver');
    
    render(<ReportDisplay {...defaultProps} />);

    const markdownButton = screen.getByRole('button', { name: /export markdown/i });
    fireEvent.click(markdownButton);

    expect(saveAs).toHaveBeenCalledWith(
      expect.any(Blob),
      'ai_in_healthcare_report.md'
    );
  });

  it('exports to Word when Word button is clicked', () => {
    const { saveAs } = require('file-saver');
    
    render(<ReportDisplay {...defaultProps} />);

    const wordButton = screen.getByRole('button', { name: /export word/i });
    fireEvent.click(wordButton);

    expect(saveAs).toHaveBeenCalledWith(
      expect.any(Blob),
      'ai_in_healthcare_report.doc'
    );
  });

  it('copies to clipboard when share button is clicked and navigator.share is not available', async () => {
    // Mock navigator.share to be undefined
    Object.defineProperty(navigator, 'share', {
      value: undefined,
      writable: true
    });

    render(<ReportDisplay {...defaultProps} />);

    const shareButton = screen.getByRole('button', { name: /share/i });
    fireEvent.click(shareButton);

    await waitFor(() => {
      expect(navigator.clipboard.writeText).toHaveBeenCalledWith(mockReport);
    });
  });

  it('uses native share when available', async () => {
    // Ensure navigator.share is properly mocked
    const mockShare = jest.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'share', {
      value: mockShare,
      writable: true
    });

    render(<ReportDisplay {...defaultProps} />);

    const shareButton = screen.getByRole('button', { name: /share/i });
    fireEvent.click(shareButton);

    await waitFor(() => {
      expect(mockShare).toHaveBeenCalledWith({
        title: 'Research Report: AI in Healthcare',
        text: expect.stringContaining('Artificial Intelligence (AI) has revolutionized healthcare'),
        url: window.location.href
      });
    });
  });

  it('shows loading state during PDF export', async () => {
    // Mock html2canvas to return a promise that we can control
    const html2canvas = require('html2canvas');
    let resolveCanvas: (value: any) => void;
    const canvasPromise = new Promise((resolve) => {
      resolveCanvas = resolve;
    });
    html2canvas.mockReturnValueOnce(canvasPromise);

    render(<ReportDisplay {...defaultProps} />);

    const pdfButton = screen.getByRole('button', { name: /export pdf/i });
    fireEvent.click(pdfButton);

    // Check loading state immediately after click
    expect(screen.getByText('Exporting...')).toBeInTheDocument();
    expect(pdfButton).toBeDisabled();

    // Resolve the promise to complete the test
    resolveCanvas!({
      toDataURL: jest.fn().mockReturnValue('data:image/png;base64,mock-image-data'),
      height: 1000,
      width: 800
    });

    await waitFor(() => {
      expect(screen.queryByText('Exporting...')).not.toBeInTheDocument();
    });
  });

  it('handles PDF export error gracefully', async () => {
    const html2canvas = require('html2canvas');
    html2canvas.mockRejectedValueOnce(new Error('Canvas error'));

    // Mock alert
    window.alert = jest.fn();

    render(<ReportDisplay {...defaultProps} />);

    const pdfButton = screen.getByRole('button', { name: /export pdf/i });
    fireEvent.click(pdfButton);

    await waitFor(() => {
      expect(window.alert).toHaveBeenCalledWith('Error exporting to PDF. Please try again.');
    });
  });

  it('converts markdown to HTML correctly', () => {
    render(<ReportDisplay {...defaultProps} />);

    // Check if markdown headers are converted to HTML
    const h1Element = screen.getByRole('heading', { level: 1 });
    expect(h1Element).toHaveTextContent('Research Report: AI in Healthcare');

    const h2Elements = screen.getAllByRole('heading', { level: 2 });
    // There are 4 h2 elements: "Research Report" (from header) + "Introduction", "Key Findings", "Conclusion"
    expect(h2Elements).toHaveLength(4);

    const h3Element = screen.getByRole('heading', { level: 3 });
    expect(h3Element).toHaveTextContent('Machine Learning Applications');
  });

  it('has proper accessibility attributes', () => {
    render(<ReportDisplay {...defaultProps} />);

    const buttons = screen.getAllByRole('button');
    // Buttons don't need explicit type="button" attribute in React
    expect(buttons.length).toBeGreaterThan(0);

    const headings = screen.getAllByRole('heading');
    expect(headings.length).toBeGreaterThan(0);
  });

  it('handles empty report gracefully', () => {
    render(<ReportDisplay {...defaultProps} report="" />);

    expect(screen.getByText('Research Report')).toBeInTheDocument();
    // Use getAllByText since there are multiple "0" values in the statistics
    const zeroElements = screen.getAllByText('0');
    expect(zeroElements.length).toBeGreaterThan(0);
  });
});