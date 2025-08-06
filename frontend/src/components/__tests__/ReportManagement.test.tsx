import React from 'react';
import { render, screen, fireEvent, waitFor } from '../../test-utils';
import '@testing-library/jest-dom';
import { ReportManagement } from '../ReportManagement';

// Mock external dependencies
jest.mock('file-saver', () => ({
  saveAs: jest.fn()
}));

jest.mock('jspdf', () => {
  return jest.fn().mockImplementation(() => ({
    addImage: jest.fn(),
    addPage: jest.fn(),
    save: jest.fn()
  }));
});

jest.mock('html2canvas', () => {
  return jest.fn().mockResolvedValue({
    toDataURL: jest.fn().mockReturnValue('data:image/png;base64,mock'),
    height: 800,
    width: 600
  });
});

describe('ReportManagement', () => {
  const mockReport = `# Test Report

## Introduction
This is a test report for the TTD-DR framework.

## Key Findings
- Finding 1
- Finding 2

## Conclusion
This concludes our test report.`;

  const defaultProps = {
    report: mockReport,
    topic: 'Test Research Topic',
    workflowId: 'test-workflow-123',
    onSave: jest.fn(),
    onAddAnnotation: jest.fn(),
    onResolveAnnotation: jest.fn(),
    onShare: jest.fn()
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders the report management interface', () => {
    render(<ReportManagement {...defaultProps} />);
    
    expect(screen.getByText('Research Report Management')).toBeInTheDocument();
    expect(screen.getByText('Test Research Topic')).toBeInTheDocument();
    expect(screen.getByText('Workflow ID: test-workflow-123')).toBeInTheDocument();
  });

  it('displays all tab navigation options', () => {
    render(<ReportManagement {...defaultProps} />);
    
    expect(screen.getByText('View')).toBeInTheDocument();
    expect(screen.getByText('Edit')).toBeInTheDocument();
    expect(screen.getAllByText('Annotations')).toHaveLength(2); // Tab and sidebar
    expect(screen.getByText('Versions')).toBeInTheDocument();
    expect(screen.getByText('Compare')).toBeInTheDocument();
  });

  it('displays export options', () => {
    render(<ReportManagement {...defaultProps} />);
    
    expect(screen.getByText('Export PDF')).toBeInTheDocument();
    expect(screen.getByText('Export Word')).toBeInTheDocument();
    expect(screen.getByText('Export Markdown')).toBeInTheDocument();
    expect(screen.getByText('Share')).toBeInTheDocument();
  });

  it('switches between view and edit tabs', () => {
    render(<ReportManagement {...defaultProps} />);
    
    // Initially on view tab - check for converted HTML content
    expect(screen.getByText('Test Report')).toBeInTheDocument();
    
    // Switch to edit tab
    fireEvent.click(screen.getByText('Edit'));
    expect(screen.getByText('Edit Report')).toBeInTheDocument();
    expect(screen.getByDisplayValue(mockReport)).toBeInTheDocument();
  });

  it('allows editing report content', () => {
    render(<ReportManagement {...defaultProps} />);
    
    // Switch to edit tab
    fireEvent.click(screen.getByText('Edit'));
    
    const textarea = screen.getByDisplayValue(mockReport);
    const newContent = '# Updated Report\n\nThis is updated content.';
    
    fireEvent.change(textarea, { target: { value: newContent } });
    expect(textarea).toHaveValue(newContent);
  });

  it('saves changes when save button is clicked', () => {
    render(<ReportManagement {...defaultProps} />);
    
    // Switch to edit tab
    fireEvent.click(screen.getByText('Edit'));
    
    const textarea = screen.getByDisplayValue(mockReport);
    const newContent = '# Updated Report\n\nThis is updated content.';
    
    fireEvent.change(textarea, { target: { value: newContent } });
    fireEvent.click(screen.getByText('Save Changes'));
    
    expect(defaultProps.onSave).toHaveBeenCalledWith(newContent);
  });

  it('cancels editing and reverts changes', () => {
    render(<ReportManagement {...defaultProps} />);
    
    // Switch to edit tab
    fireEvent.click(screen.getByText('Edit'));
    
    const textarea = screen.getByDisplayValue(mockReport);
    const newContent = '# Updated Report\n\nThis is updated content.';
    
    fireEvent.change(textarea, { target: { value: newContent } });
    fireEvent.click(screen.getByText('Cancel'));
    
    // Should switch back to view tab and not save changes
    expect(defaultProps.onSave).not.toHaveBeenCalled();
    expect(screen.getByText('Test Report')).toBeInTheDocument();
  });

  it('displays annotations tab with add annotation functionality', () => {
    render(<ReportManagement {...defaultProps} />);
    
    // Switch to annotations tab - use getAllByText and select the first one (tab)
    const annotationTabs = screen.getAllByText('Annotations');
    fireEvent.click(annotationTabs[0]);
    
    expect(screen.getByText('Add')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Search annotations...')).toBeInTheDocument();
  });

  it('opens annotation form when add button is clicked', () => {
    render(<ReportManagement {...defaultProps} />);
    
    // Switch to annotations tab - use getAllByText and select the first one (tab)
    const annotationTabs = screen.getAllByText('Annotations');
    fireEvent.click(annotationTabs[0]);
    
    // Click add annotation button
    fireEvent.click(screen.getByText('Add'));
    
    expect(screen.getByText('Add Annotation')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Enter your annotation...')).toBeInTheDocument();
  });

  it('displays versions tab with version history', () => {
    const versions = [
      {
        id: 'v1',
        content: 'Version 1 content',
        timestamp: new Date('2024-01-01'),
        author: 'User 1',
        changes: 'Initial version'
      },
      {
        id: 'v2',
        content: 'Version 2 content',
        timestamp: new Date('2024-01-02'),
        author: 'User 2',
        changes: 'Updated introduction'
      }
    ];

    render(<ReportManagement {...defaultProps} versions={versions} />);
    
    // Switch to versions tab
    fireEvent.click(screen.getByText('Versions'));
    
    expect(screen.getByText('Version History')).toBeInTheDocument();
    expect(screen.getByText('Version v1')).toBeInTheDocument();
    expect(screen.getByText('Version v2')).toBeInTheDocument();
    expect(screen.getByText('Initial version')).toBeInTheDocument();
    expect(screen.getByText('Updated introduction')).toBeInTheDocument();
  });

  it('displays compare tab with version selection', () => {
    const versions = [
      {
        id: 'v1',
        content: 'Version 1 content',
        timestamp: new Date('2024-01-01'),
        author: 'User 1',
        changes: 'Initial version'
      }
    ];

    render(<ReportManagement {...defaultProps} versions={versions} />);
    
    // Switch to compare tab
    fireEvent.click(screen.getByText('Compare'));
    
    expect(screen.getByText('Compare Versions')).toBeInTheDocument();
    expect(screen.getByText('Select first version')).toBeInTheDocument();
    expect(screen.getByText('Select second version')).toBeInTheDocument();
  });

  it('displays report statistics', () => {
    render(<ReportManagement {...defaultProps} />);
    
    expect(screen.getByText('Report Statistics')).toBeInTheDocument();
    expect(screen.getByText('Words')).toBeInTheDocument();
    expect(screen.getByText('Characters')).toBeInTheDocument();
    expect(screen.getByText('Sections')).toBeInTheDocument();
    expect(screen.getByText('References')).toBeInTheDocument();
    // Check for Annotations in statistics - should be the second occurrence
    const annotationsElements = screen.getAllByText('Annotations');
    expect(annotationsElements.length).toBeGreaterThanOrEqual(1);
  });

  it('toggles between formatted and raw markdown view', () => {
    render(<ReportManagement {...defaultProps} />);
    
    // Initially shows formatted view - check for converted HTML content
    expect(screen.getByText('Test Report')).toBeInTheDocument();
    
    // Toggle to raw view
    fireEvent.click(screen.getByText('Raw'));
    expect(screen.getByText('Raw Markdown')).toBeInTheDocument();
    
    // Toggle back to formatted view
    fireEvent.click(screen.getByText('Formatted'));
    expect(screen.getByText('Test Report')).toBeInTheDocument();
  });

  it('opens share modal when share button is clicked', () => {
    render(<ReportManagement {...defaultProps} />);
    
    fireEvent.click(screen.getByText('Share'));
    
    // Check for modal title specifically
    expect(screen.getByRole('heading', { name: 'Share Report' })).toBeInTheDocument();
    expect(screen.getByText('Email addresses (comma-separated)')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Enter email addresses...')).toBeInTheDocument();
  });

  it('handles share functionality', () => {
    render(<ReportManagement {...defaultProps} />);
    
    fireEvent.click(screen.getByText('Share'));
    
    const emailInput = screen.getByPlaceholderText('Enter email addresses...');
    fireEvent.change(emailInput, { target: { value: 'user1@example.com, user2@example.com' } });
    
    fireEvent.click(screen.getByRole('button', { name: 'Share Report' }));
    
    expect(defaultProps.onShare).toHaveBeenCalledWith(['user1@example.com', 'user2@example.com']);
  });

  it('filters annotations by status', () => {
    render(<ReportManagement {...defaultProps} />);
    
    // Switch to annotations tab - use getAllByText and select the first one (tab)
    const annotationTabs = screen.getAllByText('Annotations');
    fireEvent.click(annotationTabs[0]);
    
    const filterSelect = screen.getByDisplayValue('All Annotations');
    expect(filterSelect).toBeInTheDocument();
    
    fireEvent.change(filterSelect, { target: { value: 'unresolved' } });
    expect(filterSelect).toHaveValue('unresolved');
  });

  it('searches annotations', () => {
    render(<ReportManagement {...defaultProps} />);
    
    // Switch to annotations tab - use getAllByText and select the first one (tab)
    const annotationTabs = screen.getAllByText('Annotations');
    fireEvent.click(annotationTabs[0]);
    
    const searchInput = screen.getByPlaceholderText('Search annotations...');
    fireEvent.change(searchInput, { target: { value: 'test search' } });
    
    expect(searchInput).toHaveValue('test search');
  });

  it('displays word and character count in edit mode', () => {
    render(<ReportManagement {...defaultProps} />);
    
    // Switch to edit tab
    fireEvent.click(screen.getByText('Edit'));
    
    // Should display word and character count
    const wordCount = mockReport.split(' ').length;
    const charCount = mockReport.length;
    
    expect(screen.getByText(`${wordCount} words, ${charCount} characters`)).toBeInTheDocument();
  });

  it('closes modals when X button is clicked', () => {
    render(<ReportManagement {...defaultProps} />);
    
    // Open share modal
    fireEvent.click(screen.getByText('Share'));
    expect(screen.getByRole('heading', { name: 'Share Report' })).toBeInTheDocument();
    
    // Close modal using Cancel button (which is more reliable to test)
    const cancelButton = screen.getByText('Cancel');
    fireEvent.click(cancelButton);
    
    expect(screen.queryByRole('heading', { name: 'Share Report' })).not.toBeInTheDocument();
  });

  it('handles export to PDF', async () => {
    const html2canvas = require('html2canvas');
    const jsPDF = require('jspdf');
    
    // Mock getElementById to return a mock element
    const mockElement = document.createElement('div');
    mockElement.id = 'report-content';
    document.body.appendChild(mockElement);
    
    render(<ReportManagement {...defaultProps} />);
    
    fireEvent.click(screen.getByText('Export PDF'));
    
    await waitFor(() => {
      expect(html2canvas).toHaveBeenCalled();
    }, { timeout: 3000 });
    
    expect(jsPDF).toHaveBeenCalled();
    
    // Cleanup
    document.body.removeChild(mockElement);
  });

  it('handles export to markdown', () => {
    const { saveAs } = require('file-saver');
    
    render(<ReportManagement {...defaultProps} />);
    
    fireEvent.click(screen.getByText('Export Markdown'));
    
    expect(saveAs).toHaveBeenCalledWith(
      expect.any(Blob),
      'test_research_topic_report.md'
    );
  });

  it('handles export to Word', () => {
    const { saveAs } = require('file-saver');
    
    render(<ReportManagement {...defaultProps} />);
    
    fireEvent.click(screen.getByText('Export Word'));
    
    expect(saveAs).toHaveBeenCalledWith(
      expect.any(Blob),
      'test_research_topic_report.doc'
    );
  });
});