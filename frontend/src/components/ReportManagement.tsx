import React, { useState, useRef, useEffect } from 'react';
import { 
  Download, 
  FileText, 
  Share2, 
  Eye, 
  EyeOff, 
  Edit3, 
  Save, 
  X, 
  MessageSquare, 
  Plus, 
  Trash2,
  GitCompare,
  Users,
  Copy,
  History,
  BookOpen,
  Filter,
  Search
} from 'lucide-react';
import { saveAs } from 'file-saver';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import { useNotifications } from './NotificationSystem';
import { useLoading, LoadingButton } from './LoadingSystem';
import { InlineErrorBoundary } from './ErrorBoundary';

interface Annotation {
  id: string;
  text: string;
  position: { start: number; end: number };
  author: string;
  timestamp: Date;
  resolved: boolean;
}

interface ReportVersion {
  id: string;
  content: string;
  timestamp: Date;
  author: string;
  changes: string;
}

interface ReportManagementProps {
  report: string;
  topic: string;
  workflowId: string;
  versions?: ReportVersion[];
  onSave?: (content: string) => void;
  onAddAnnotation?: (annotation: Omit<Annotation, 'id' | 'timestamp'>) => void;
  onResolveAnnotation?: (annotationId: string) => void;
  onShare?: (recipients: string[]) => void;
}

export const ReportManagement: React.FC<ReportManagementProps> = ({ 
  report, 
  topic, 
  workflowId,
  versions = [],
  onSave,
  onAddAnnotation,
  onResolveAnnotation,
  onShare
}) => {
  const { showSuccess, showError, showWarning } = useNotifications();
  const { startLoading, finishLoading } = useLoading();
  
  const [activeTab, setActiveTab] = useState<'view' | 'edit' | 'annotations' | 'versions' | 'compare'>('view');
  const [showRawMarkdown, setShowRawMarkdown] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [editContent, setEditContent] = useState(report);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [selectedText, setSelectedText] = useState('');
  const [showAnnotationForm, setShowAnnotationForm] = useState(false);
  const [newAnnotation, setNewAnnotation] = useState('');
  const [compareVersions, setCompareVersions] = useState<{ v1: string; v2: string }>({ v1: '', v2: '' });
  const [searchTerm, setSearchTerm] = useState('');
  const [annotationFilter, setAnnotationFilter] = useState<'all' | 'unresolved' | 'resolved'>('all');
  const [shareRecipients, setShareRecipients] = useState<string[]>([]);
  const [showShareModal, setShowShareModal] = useState(false);
  
  const contentRef = useRef<HTMLDivElement>(null);
  const editTextareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    setEditContent(report);
  }, [report]);

  // Handle text selection for annotations
  const handleTextSelection = () => {
    const selection = window.getSelection();
    if (selection && selection.toString().trim()) {
      setSelectedText(selection.toString());
      setShowAnnotationForm(true);
    }
  };

  const addAnnotation = () => {
    if (selectedText && newAnnotation.trim()) {
      const annotation: Omit<Annotation, 'id' | 'timestamp'> = {
        text: newAnnotation,
        position: { start: 0, end: selectedText.length }, // Simplified for demo
        author: 'Current User',
        resolved: false
      };
      
      const fullAnnotation: Annotation = {
        ...annotation,
        id: Date.now().toString(),
        timestamp: new Date()
      };
      
      setAnnotations(prev => [...prev, fullAnnotation]);
      onAddAnnotation?.(annotation);
      setNewAnnotation('');
      setShowAnnotationForm(false);
      setSelectedText('');
    }
  };

  const resolveAnnotation = (annotationId: string) => {
    setAnnotations(prev => 
      prev.map(ann => 
        ann.id === annotationId ? { ...ann, resolved: true } : ann
      )
    );
    onResolveAnnotation?.(annotationId);
  };

  const deleteAnnotation = (annotationId: string) => {
    setAnnotations(prev => prev.filter(ann => ann.id !== annotationId));
  };

  const saveChanges = () => {
    const loadingId = 'save-changes';
    startLoading(loadingId, 'Saving changes...', 'Updating report content');
    
    try {
      onSave?.(editContent);
      finishLoading(loadingId, 'success', 'Changes saved successfully');
      showSuccess('Changes Saved', 'Your report has been updated successfully.');
      setActiveTab('view');
    } catch (error) {
      console.error('Error saving changes:', error);
      finishLoading(loadingId, 'error', 'Failed to save changes');
      showError('Save Failed', 'Failed to save your changes. Please try again.');
    }
  };

  const exportToPDF = async () => {
    const loadingId = 'export-pdf';
    startLoading(loadingId, 'Exporting to PDF...', 'Generating PDF document');
    setIsExporting(true);
    
    try {
      const element = document.getElementById('report-content');
      if (!element) {
        throw new Error('Report content not found. Please ensure the report is visible.');
      }

      // Update progress
      finishLoading(loadingId, 'success');
      startLoading(loadingId, 'Capturing content...', 'Taking screenshot of report');

      const canvas = await html2canvas(element, {
        scale: 2,
        useCORS: true,
        allowTaint: true,
        backgroundColor: '#ffffff'
      });

      // Update progress
      finishLoading(loadingId, 'success');
      startLoading(loadingId, 'Creating PDF...', 'Converting to PDF format');

      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('p', 'mm', 'a4');
      
      const imgWidth = 210;
      const pageHeight = 295;
      const imgHeight = (canvas.height * imgWidth) / canvas.width;
      let heightLeft = imgHeight;
      let position = 0;

      pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
      heightLeft -= pageHeight;

      while (heightLeft >= 0) {
        position = heightLeft - imgHeight;
        pdf.addPage();
        pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
        heightLeft -= pageHeight;
      }

      const filename = `${topic.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_report.pdf`;
      pdf.save(filename);
      
      finishLoading(loadingId, 'success', 'PDF exported successfully');
      showSuccess('Export Complete', `Report exported as ${filename}`);
    } catch (error) {
      console.error('Error exporting to PDF:', error);
      finishLoading(loadingId, 'error', 'PDF export failed');
      showError(
        'Export Failed', 
        error instanceof Error ? error.message : 'Failed to export PDF. Please try again.',
        {
          action: {
            label: 'Retry',
            onClick: () => exportToPDF()
          }
        }
      );
    } finally {
      setIsExporting(false);
    }
  };

  const exportToMarkdown = () => {
    const loadingId = 'export-markdown';
    startLoading(loadingId, 'Exporting to Markdown...', 'Preparing markdown file');
    
    try {
      const blob = new Blob([editContent], { type: 'text/markdown;charset=utf-8' });
      const filename = `${topic.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_report.md`;
      saveAs(blob, filename);
      
      finishLoading(loadingId, 'success', 'Markdown exported successfully');
      showSuccess('Export Complete', `Report exported as ${filename}`);
    } catch (error) {
      console.error('Error exporting to Markdown:', error);
      finishLoading(loadingId, 'error', 'Markdown export failed');
      showError('Export Failed', 'Failed to export Markdown file. Please try again.');
    }
  };

  const exportToWord = () => {
    const loadingId = 'export-word';
    startLoading(loadingId, 'Exporting to Word...', 'Converting to Word format');
    
    try {
      const htmlContent = `
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>${topic}</title>
          <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }
            h1 { color: #333; border-bottom: 2px solid #333; }
            h2 { color: #666; border-bottom: 1px solid #666; }
            h3 { color: #888; }
            blockquote { border-left: 4px solid #ddd; margin: 0; padding-left: 20px; }
            code { background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }
            pre { background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }
          </style>
        </head>
        <body>
          ${convertMarkdownToHTML(editContent)}
        </body>
        </html>
      `;
      
      const blob = new Blob([htmlContent], { type: 'application/msword;charset=utf-8' });
      const filename = `${topic.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_report.doc`;
      saveAs(blob, filename);
      
      finishLoading(loadingId, 'success', 'Word document exported successfully');
      showSuccess('Export Complete', `Report exported as ${filename}`);
    } catch (error) {
      console.error('Error exporting to Word:', error);
      finishLoading(loadingId, 'error', 'Word export failed');
      showError('Export Failed', 'Failed to export Word document. Please try again.');
    }
  };

  const convertMarkdownToHTML = (markdown: string): string => {
    return markdown
      .replace(/^### (.*$)/gim, '<h3>$1</h3>')
      .replace(/^## (.*$)/gim, '<h2>$1</h2>')
      .replace(/^# (.*$)/gim, '<h1>$1</h1>')
      .replace(/\*\*(.*)\*\*/gim, '<strong>$1</strong>')
      .replace(/\*(.*)\*/gim, '<em>$1</em>')
      .replace(/!\[([^\]]*)\]\(([^\)]*)\)/gim, '<img alt="$1" src="$2" />')
      .replace(/\[([^\]]*)\]\(([^\)]*)\)/gim, '<a href="$2">$1</a>')
      .replace(/\n$/gim, '<br />')
      .replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>')
      .replace(/```([^`]*)```/gim, '<pre><code>$1</code></pre>')
      .replace(/`([^`]*)`/gim, '<code>$1</code>')
      .replace(/\n/gim, '<br />');
  };

  const shareReport = () => {
    setShowShareModal(true);
  };

  const handleShare = () => {
    onShare?.(shareRecipients);
    setShowShareModal(false);
    setShareRecipients([]);
  };

  const filteredAnnotations = annotations.filter(annotation => {
    const matchesFilter = annotationFilter === 'all' || 
      (annotationFilter === 'resolved' && annotation.resolved) ||
      (annotationFilter === 'unresolved' && !annotation.resolved);
    
    const matchesSearch = !searchTerm || 
      annotation.text.toLowerCase().includes(searchTerm.toLowerCase()) ||
      annotation.author.toLowerCase().includes(searchTerm.toLowerCase());
    
    return matchesFilter && matchesSearch;
  });

  const highlightSearchTerm = (text: string) => {
    if (!searchTerm) return text;
    const regex = new RegExp(`(${searchTerm})`, 'gi');
    return text.replace(regex, '<mark>$1</mark>');
  };

  return (
    <div className="max-w-7xl mx-auto p-4 md:p-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-lg p-4 md:p-6 mb-4 md:mb-6">
        <div className="flex flex-col sm:flex-row justify-between items-start mb-4 space-y-4 sm:space-y-0">
          <div className="min-w-0 flex-1">
            <h2 className="text-xl md:text-2xl font-bold text-gray-800 mb-2 truncate">研究报告管理</h2>
            <p className="text-sm md:text-base text-gray-600 truncate">{topic}</p>
            <p className="text-xs md:text-sm text-gray-500 truncate">工作流ID: {workflowId}</p>
          </div>
          
          <div className="flex items-center space-x-2 flex-shrink-0">
            <button
              onClick={() => setShowRawMarkdown(!showRawMarkdown)}
              className="btn-secondary space-x-1 md:space-x-2 text-sm px-3 py-2"
              title={showRawMarkdown ? 'Show formatted view' : 'Show raw markdown'}
            >
              {showRawMarkdown ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
              <span className="text-xs md:text-sm">{showRawMarkdown ? 'Formatted' : 'Raw'}</span>
            </button>
          </div>
        </div>

        {/* Tab Navigation - Modern Design */}
        <div className="relative mb-8">
          <div className="border-b border-neutral-200">
            <nav className="flex space-x-1 -mb-px">
              {[
                { id: 'view', label: '查看', icon: Eye },
        { id: 'edit', label: '编辑', icon: Edit3 },
        { id: 'annotations', label: '注释', icon: MessageSquare },
        { id: 'versions', label: '版本', icon: History },
        { id: 'compare', label: '对比', icon: GitCompare }
              ].map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => setActiveTab(id as any)}
                  className={`group relative px-6 py-3 font-medium text-sm transition-all duration-300 ease-out ${
                    activeTab === id
                      ? 'text-primary-700 bg-gradient-to-b from-primary-50 to-white border-b-2 border-primary-500'
                      : 'text-neutral-600 hover:text-neutral-900 hover:bg-neutral-50 border-b-2 border-transparent hover:border-neutral-200'
                  } rounded-t-xl relative overflow-hidden`}
                >
                  {/* Active tab background gradient */}
                  {activeTab === id && (
                    <div className="absolute inset-0 bg-gradient-to-r from-primary-50/50 via-primary-50/30 to-primary-50/50 rounded-t-xl" />
                  )}
                  
                  <div className="relative flex items-center space-x-2.5">
                    <Icon className={`w-4 h-4 transition-all duration-200 ${
                      activeTab === id ? 'text-primary-600' : 'text-neutral-500 group-hover:text-neutral-700'
                    }`} />
                    <span className="font-medium">{label}</span>
                    {id === 'annotations' && annotations.length > 0 && (
                      <div className="relative">
                        <span className={`inline-flex items-center justify-center min-w-[20px] h-5 text-xs font-bold rounded-full transition-all duration-200 ${
                          annotations.filter(a => !a.resolved).length > 0
                            ? 'bg-error-500 text-white shadow-sm'
                            : 'bg-success-100 text-success-700'
                        }`}>
                          {annotations.filter(a => !a.resolved).length}
                        </span>
                        {annotations.filter(a => !a.resolved).length > 0 && (
                          <span className="absolute -top-1 -right-1 w-2 h-2 bg-error-400 rounded-full animate-pulse" />
                        )}
                      </div>
                    )}
                  </div>
                  
                  {/* Hover effect */}
                  <div className="absolute inset-x-0 bottom-0 h-0.5 bg-gradient-to-r from-transparent via-neutral-300 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
                </button>
              ))}
            </nav>
          </div>
          
          {/* Active tab indicator line */}
          <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-primary-400 via-primary-500 to-primary-400 opacity-20" />
        </div>

        {/* Export Options */}
        <div className="flex flex-wrap gap-2">
          <LoadingButton
            onClick={exportToPDF}
            loading={isExporting}
            loadingText="Exporting..."
            className="btn-error space-x-1 md:space-x-2 text-sm px-3 py-2"
          >
            <Download className="w-4 h-4" />
            <span className="hidden sm:inline">Export PDF</span>
            <span className="sm:hidden">PDF</span>
          </LoadingButton>

          <LoadingButton
            onClick={exportToWord}
            className="btn-primary space-x-1 md:space-x-2 text-sm px-3 py-2"
          >
            <FileText className="w-4 h-4" />
            <span className="hidden sm:inline">Export Word</span>
            <span className="sm:hidden">Word</span>
          </LoadingButton>

          <LoadingButton
            onClick={exportToMarkdown}
            className="btn-secondary space-x-1 md:space-x-2 text-sm px-3 py-2"
          >
            <Download className="w-4 h-4" />
            <span className="hidden sm:inline">Export Markdown</span>
            <span className="sm:hidden">MD</span>
          </LoadingButton>

          <LoadingButton
            onClick={shareReport}
            className="btn-success space-x-1 md:space-x-2 text-sm px-3 py-2"
          >
            <Share2 className="w-4 h-4" />
            <span>Share</span>
          </LoadingButton>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 md:gap-6">
        {/* Main Content */}
        <div className="lg:col-span-3">
          {activeTab === 'view' && (
            <div className="card-base overflow-hidden">
              {showRawMarkdown ? (
                <div className="p-6">
                  <div className="flex items-center space-x-3 mb-6">
                    <div className="w-8 h-8 bg-neutral-100 rounded-lg flex items-center justify-center">
                      <FileText className="w-4 h-4 text-neutral-600" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-neutral-900">Raw Markdown</h3>
                      <p className="text-sm text-neutral-600">Source code view</p>
                    </div>
                  </div>
                  <div className="relative">
                    <pre className="bg-neutral-900 text-neutral-100 p-6 rounded-xl overflow-x-auto text-sm font-mono whitespace-pre-wrap leading-relaxed border border-neutral-200">
                      {editContent}
                    </pre>
                    <button
                      onClick={() => navigator.clipboard.writeText(editContent)}
                      className="absolute top-4 right-4 p-2 bg-neutral-800 hover:bg-neutral-700 text-neutral-300 hover:text-white rounded-lg transition-all duration-200"
                      title="Copy to clipboard"
                    >
                      <Copy className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ) : (
                <div className="relative">
                  {/* Content Header */}
                  <div className="bg-gradient-to-r from-neutral-50 to-neutral-100 px-6 py-4 border-b border-neutral-200">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-primary-100 rounded-lg flex items-center justify-center">
                        <BookOpen className="w-4 h-4 text-primary-600" />
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-neutral-900">Report Content</h3>
                        <p className="text-sm text-neutral-600">Formatted view • Select text to annotate</p>
                      </div>
                    </div>
                  </div>

                  {/* Report Content */}
                  <div 
                    id="report-content" 
                    className="p-8 prose prose-lg max-w-none selection:bg-primary-100 selection:text-primary-900"
                    onMouseUp={handleTextSelection}
                    ref={contentRef}
                    style={{ 
                      lineHeight: '1.8',
                      fontSize: '16px'
                    }}
                  >
                    <div 
                      dangerouslySetInnerHTML={{ 
                        __html: convertMarkdownToHTML(editContent) 
                      }} 
                    />
                  </div>

                  {/* Reading Progress Indicator */}
                  <div className="sticky bottom-0 left-0 right-0 h-1 bg-neutral-200">
                    <div 
                      className="h-full bg-gradient-to-r from-primary-500 to-primary-600 transition-all duration-300"
                      style={{ width: '0%' }}
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'edit' && (
            <div className="card-base overflow-hidden">
              {/* Editor Header */}
              <div className="bg-gradient-to-r from-neutral-50 to-neutral-100 px-6 py-4 border-b border-neutral-200">
                <div className="flex justify-between items-center">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-primary-100 rounded-lg flex items-center justify-center">
                      <Edit3 className="w-4 h-4 text-primary-600" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-neutral-900">内容编辑器</h3>
                      <p className="text-sm text-neutral-600">Markdown supported</p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <div className="text-sm text-neutral-500 bg-white px-3 py-1.5 rounded-lg border border-neutral-200">
                      {editContent.split(' ').length.toLocaleString()} words
                    </div>
                    <div className="flex space-x-2">
                      <button
                        onClick={saveChanges}
                        className="btn-success space-x-2 shadow-sm"
                      >
                        <Save className="w-4 h-4" />
                        <span>Save Changes</span>
                      </button>
                      <button
                        onClick={() => {
                          setEditContent(report);
                          setActiveTab('view');
                        }}
                        className="btn-secondary space-x-2"
                      >
                        <X className="w-4 h-4" />
                        <span>取消</span>
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Editor Content */}
              <div className="p-6">
                <div className="relative">
                  <textarea
                    ref={editTextareaRef}
                    value={editContent}
                    onChange={(e) => setEditContent(e.target.value)}
                    className="w-full h-[600px] p-6 text-sm font-mono leading-relaxed border-2 border-neutral-200 rounded-xl bg-neutral-50/30 placeholder-neutral-400 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 focus:bg-white resize-none"
                    placeholder="# Your Report Title

## Introduction
Start writing your report content here...

## Main Content
- Use markdown formatting
- **Bold text** and *italic text*
- [Links](https://example.com)
- Code blocks with ```

## Conclusion
Wrap up your findings..."
                    style={{
                      fontFamily: 'JetBrains Mono, Consolas, Monaco, "Courier New", monospace',
                      lineHeight: '1.6',
                      tabSize: 2
                    }}
                  />
                  
                  {/* Line numbers overlay (optional) */}
                  <div className="absolute left-2 top-6 text-xs text-neutral-400 font-mono leading-relaxed pointer-events-none select-none">
                    {editContent.split('\n').map((_, index) => (
                      <div key={index} className="h-[1.6em]">
                        {index + 1}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Editor Footer */}
                <div className="mt-4 flex justify-between items-center text-sm text-neutral-500 bg-neutral-50 px-4 py-3 rounded-lg border border-neutral-200">
                  <div className="flex items-center space-x-6">
                    <span className="flex items-center space-x-1">
                      <span className="w-2 h-2 bg-success-400 rounded-full"></span>
                      <span>Auto-save enabled</span>
                    </span>
                    <span>{editContent.length.toLocaleString()} characters</span>
                    <span>{editContent.split('\n').length} lines</span>
                    <span>{(editContent.match(/^#+\s/gm) || []).length} headings</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs bg-neutral-200 px-2 py-1 rounded">Markdown</span>
                    <span className="text-xs bg-neutral-200 px-2 py-1 rounded">UTF-8</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'versions' && (
            <div className="card-base overflow-hidden">
              {/* Versions Header */}
              <div className="bg-gradient-to-r from-neutral-50 to-neutral-100 px-6 py-4 border-b border-neutral-200">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-neutral-100 rounded-lg flex items-center justify-center">
                    <History className="w-4 h-4 text-neutral-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-neutral-900">Version History</h3>
                    <p className="text-sm text-neutral-600">
                      {versions.length} {versions.length === 1 ? '个版本' : '个版本'}可用
                    </p>
                  </div>
                </div>
              </div>

              <div className="p-6">
                {versions.length === 0 ? (
                  <div className="text-center py-12">
                    <History className="w-12 h-12 text-neutral-300 mx-auto mb-4" />
                    <p className="text-neutral-500 text-sm mb-2">No version history available</p>
                    <p className="text-neutral-400 text-xs">保存更改后，版本将在此处显示</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {/* Current Version */}
                    <div className="relative p-4 bg-primary-50 border-2 border-primary-200 rounded-xl">
                      <div className="absolute top-3 right-3">
                        <span className="bg-primary-500 text-white text-xs px-2 py-1 rounded-full font-medium">
                          Current
                        </span>
                      </div>
                      <div className="flex items-start space-x-3">
                        <div className="w-8 h-8 bg-primary-500 rounded-lg flex items-center justify-center flex-shrink-0">
                          <span className="text-white text-sm font-bold">C</span>
                        </div>
                        <div className="flex-1">
                          <h4 className="font-semibold text-primary-900 mb-1">Current Version</h4>
                          <p className="text-sm text-primary-700 mb-2">
                            Last modified: {new Date().toLocaleString()}
                          </p>
                          <p className="text-sm text-primary-600">Active working version</p>
                        </div>
                      </div>
                    </div>

                    {/* Version History */}
                    {versions.map((version, index) => (
                      <div key={version.id} className="group relative p-4 bg-white border-2 border-neutral-200 rounded-xl hover:border-neutral-300 hover:shadow-md transition-all duration-200">
                        <div className="flex items-start space-x-3">
                          <div className="w-8 h-8 bg-neutral-100 rounded-lg flex items-center justify-center flex-shrink-0">
                            <span className="text-neutral-600 text-sm font-bold">{versions.length - index}</span>
                          </div>
                          <div className="flex-1">
                            <div className="flex justify-between items-start mb-2">
                              <div>
                                <h4 className="font-semibold text-neutral-900 mb-1">
                                  Version {version.id}
                                </h4>
                                <div className="flex items-center space-x-4 text-sm text-neutral-500">
                                  <span className="flex items-center space-x-1">
                                    <div className="w-4 h-4 bg-neutral-300 rounded-full flex items-center justify-center">
                                      <span className="text-xs text-neutral-600">
                                        {version.author.charAt(0).toUpperCase()}
                                      </span>
                                    </div>
                                    <span>{version.author}</span>
                                  </span>
                                  <span>{new Date(version.timestamp).toLocaleDateString()}</span>
                                  <span>{new Date(version.timestamp).toLocaleTimeString([], { 
                                    hour: '2-digit', 
                                    minute: '2-digit' 
                                  })}</span>
                                </div>
                              </div>
                              <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex space-x-2">
                                <button
                                  onClick={() => setEditContent(version.content)}
                                  className="btn-primary text-sm px-3 py-1.5"
                                  title="Restore this version"
                                >
                                  <History className="w-3 h-3 mr-1" />
                                  Restore
                                </button>
                                <button
                                  onClick={() => {
                                    setCompareVersions({ v1: version.id, v2: 'current' });
                                    setActiveTab('compare');
                                  }}
                                  className="btn-secondary text-sm px-3 py-1.5"
                                  title="Compare with current"
                                >
                                  <GitCompare className="w-3 h-3 mr-1" />
                                  Compare
                                </button>
                              </div>
                            </div>
                            <div className="bg-neutral-50 p-3 rounded-lg border border-neutral-200">
                              <p className="text-sm text-neutral-700">{version.changes}</p>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'compare' && (
            <div className="card-base overflow-hidden">
              {/* Compare Header */}
              <div className="bg-gradient-to-r from-neutral-50 to-neutral-100 px-6 py-4 border-b border-neutral-200">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-warning-100 rounded-lg flex items-center justify-center">
                    <GitCompare className="w-4 h-4 text-warning-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-neutral-900">版本对比</h3>
                    <p className="text-sm text-neutral-600">Side-by-side version comparison</p>
                  </div>
                </div>
              </div>

              <div className="p-6">
                {/* Version Selectors */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                  <div className="space-y-2">
                    <label className="label-base">First Version</label>
                    <select
                      value={compareVersions.v1}
                      onChange={(e) => setCompareVersions(prev => ({ ...prev, v1: e.target.value }))}
                      className="input-base"
                    >
                      <option value="">Select first version</option>
                      <option value="current">Current Version</option>
                      {versions.map(v => (
                        <option key={v.id} value={v.id}>Version {v.id} - {v.author}</option>
                      ))}
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="label-base">Second Version</label>
                    <select
                      value={compareVersions.v2}
                      onChange={(e) => setCompareVersions(prev => ({ ...prev, v2: e.target.value }))}
                      className="input-base"
                    >
                      <option value="">Select second version</option>
                      <option value="current">Current Version</option>
                      {versions.map(v => (
                        <option key={v.id} value={v.id}>Version {v.id} - {v.author}</option>
                      ))}
                    </select>
                  </div>
                </div>

                {compareVersions.v1 && compareVersions.v2 ? (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* First Version */}
                    <div className="space-y-3">
                      <div className="flex items-center space-x-2 p-3 bg-primary-50 rounded-lg border border-primary-200">
                        <div className="w-6 h-6 bg-primary-500 rounded-full flex items-center justify-center">
                          <span className="text-white text-xs font-bold">1</span>
                        </div>
                        <div>
                          <h4 className="font-semibold text-primary-900">
                            {compareVersions.v1 === 'current' ? '当前版本' : `版本 ${compareVersions.v1}`}
                          </h4>
                          <p className="text-sm text-primary-600">
                            {compareVersions.v1 === 'current' 
                              ? 'Active working version' 
                              : versions.find(v => v.id === compareVersions.v1)?.timestamp.toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                      <div className="bg-neutral-50 border-2 border-neutral-200 rounded-xl h-96 overflow-y-auto">
                        <pre className="p-4 text-sm font-mono leading-relaxed whitespace-pre-wrap text-neutral-700">
                          {compareVersions.v1 === 'current' ? editContent : 
                            versions.find(v => v.id === compareVersions.v1)?.content || ''}
                        </pre>
                      </div>
                    </div>

                    {/* Second Version */}
                    <div className="space-y-3">
                      <div className="flex items-center space-x-2 p-3 bg-success-50 rounded-lg border border-success-200">
                        <div className="w-6 h-6 bg-success-500 rounded-full flex items-center justify-center">
                          <span className="text-white text-xs font-bold">2</span>
                        </div>
                        <div>
                          <h4 className="font-semibold text-success-900">
                            {compareVersions.v2 === 'current' ? '当前版本' : `版本 ${compareVersions.v2}`}
                          </h4>
                          <p className="text-sm text-success-600">
                            {compareVersions.v2 === 'current' 
                              ? 'Active working version' 
                              : versions.find(v => v.id === compareVersions.v2)?.timestamp.toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                      <div className="bg-neutral-50 border-2 border-neutral-200 rounded-xl h-96 overflow-y-auto">
                        <pre className="p-4 text-sm font-mono leading-relaxed whitespace-pre-wrap text-neutral-700">
                          {compareVersions.v2 === 'current' ? editContent : 
                            versions.find(v => v.id === compareVersions.v2)?.content || ''}
                        </pre>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <GitCompare className="w-12 h-12 text-neutral-300 mx-auto mb-4" />
                    <p className="text-neutral-500 text-sm mb-2">选择两个版本进行对比</p>
            <p className="text-neutral-400 text-xs">从上方下拉菜单中选择版本以查看差异</p>
                  </div>
                )}

                {/* Comparison Actions */}
                {compareVersions.v1 && compareVersions.v2 && (
                  <div className="mt-6 flex justify-center space-x-3">
                    <button
                      onClick={() => {
                        const content1 = compareVersions.v1 === 'current' ? editContent : 
                          versions.find(v => v.id === compareVersions.v1)?.content || '';
                        navigator.clipboard.writeText(content1);
                      }}
                      className="btn-secondary space-x-2"
                    >
                      <Copy className="w-4 h-4" />
                      <span>Copy Version 1</span>
                    </button>
                    <button
                      onClick={() => {
                        const content2 = compareVersions.v2 === 'current' ? editContent : 
                          versions.find(v => v.id === compareVersions.v2)?.content || '';
                        navigator.clipboard.writeText(content2);
                      }}
                      className="btn-secondary space-x-2"
                    >
                      <Copy className="w-4 h-4" />
                      <span>Copy Version 2</span>
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="lg:col-span-1">
          {activeTab === 'annotations' && (
            <div className="card-base overflow-hidden">
              {/* Annotations Header */}
              <div className="bg-gradient-to-r from-neutral-50 to-neutral-100 px-6 py-4 border-b border-neutral-200">
                <div className="flex justify-between items-center">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-warning-100 rounded-lg flex items-center justify-center">
                      <MessageSquare className="w-4 h-4 text-warning-600" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-neutral-900">注释</h3>
                      <p className="text-sm text-neutral-600">
                        {annotations.length} total, {annotations.filter(a => !a.resolved).length} pending
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => setShowAnnotationForm(true)}
                    className="btn-primary space-x-2 text-sm px-4 py-2 shadow-sm"
                  >
                    <Plus className="w-4 h-4" />
                    <span>添加注释</span>
                  </button>
                </div>
              </div>

              <div className="p-6">
                {/* Search and Filter */}
                <div className="space-y-3 mb-6">
                  <div className="relative">
                    <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-neutral-400" />
                    <input
                      type="text"
                      placeholder="搜索注释和评论..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="w-full pl-10 pr-4 py-2.5 text-sm border border-neutral-300 rounded-xl bg-white placeholder-neutral-400 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                    />
                  </div>
                  <div className="flex space-x-2">
                    {['all', 'unresolved', 'resolved'].map((filter) => (
                      <button
                        key={filter}
                        onClick={() => setAnnotationFilter(filter as any)}
                        className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all duration-200 ${
                          annotationFilter === filter
                            ? 'bg-primary-100 text-primary-700 border border-primary-200'
                            : 'bg-neutral-100 text-neutral-600 hover:bg-neutral-200 border border-neutral-200'
                        }`}
                      >
                        {filter === 'all' ? '全部' : filter === 'unresolved' ? '待处理' : '已解决'}
                        {filter !== 'all' && (
                          <span className="ml-1 text-xs">
                            ({filter === 'unresolved' 
                              ? annotations.filter(a => !a.resolved).length 
                              : annotations.filter(a => a.resolved).length})
                          </span>
                        )}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Annotations List */}
                <div className="space-y-4 max-h-[500px] overflow-y-auto">
                  {filteredAnnotations.map((annotation) => (
                    <div
                      key={annotation.id}
                      className={`group relative p-4 rounded-xl border-2 transition-all duration-200 hover:shadow-md ${
                        annotation.resolved 
                          ? 'bg-success-50/50 border-success-200 hover:border-success-300' 
                          : 'bg-warning-50/50 border-warning-200 hover:border-warning-300'
                      }`}
                    >
                      {/* Status indicator */}
                      <div className="absolute top-3 right-3 flex items-center space-x-2">
                        <div className={`w-2 h-2 rounded-full ${
                          annotation.resolved ? 'bg-success-400' : 'bg-warning-400 animate-pulse'
                        }`} />
                        <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex space-x-1">
                          {!annotation.resolved && (
                            <button
                              onClick={() => resolveAnnotation(annotation.id)}
                              className="p-1.5 text-success-600 hover:text-success-800 hover:bg-success-100 rounded-lg transition-all duration-200"
                              title="标记为已解决"
                            >
                              <MessageSquare className="w-4 h-4" />
                            </button>
                          )}
                          <button
                            onClick={() => deleteAnnotation(annotation.id)}
                            className="p-1.5 text-error-600 hover:text-error-800 hover:bg-error-100 rounded-lg transition-all duration-200"
                            title="删除注释"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>

                      {/* Author and timestamp */}
                      <div className="flex items-center space-x-2 mb-3">
                        <div className="w-6 h-6 bg-primary-100 rounded-full flex items-center justify-center">
                          <span className="text-xs font-medium text-primary-700">
                            {annotation.author.charAt(0).toUpperCase()}
                          </span>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-neutral-900">{annotation.author}</p>
                          <p className="text-xs text-neutral-500">
                            {new Date(annotation.timestamp).toLocaleDateString()} at{' '}
                            {new Date(annotation.timestamp).toLocaleTimeString([], { 
                              hour: '2-digit', 
                              minute: '2-digit' 
                            })}
                          </p>
                        </div>
                        {annotation.resolved && (
                          <span className="ml-auto text-xs bg-success-100 text-success-700 px-2 py-1 rounded-full font-medium">
                            已解决
                          </span>
                        )}
                      </div>

                      {/* Annotation content */}
                      <div className="pl-8">
                        <p 
                          className="text-sm text-neutral-700 leading-relaxed"
                          dangerouslySetInnerHTML={{ 
                            __html: highlightSearchTerm(annotation.text) 
                          }}
                        />
                      </div>
                    </div>
                  ))}
                  
                  {filteredAnnotations.length === 0 && (
                    <div className="text-center py-12">
                      <MessageSquare className="w-12 h-12 text-neutral-300 mx-auto mb-4" />
                      <p className="text-neutral-500 text-sm">
                        {searchTerm || annotationFilter !== 'all'
            ? '没有符合搜索条件的注释。'
            : '还没有注释。选择文本以添加您的第一个注释。'}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Report Statistics - Enhanced Design */}
          <div className="card-base overflow-hidden mt-6">
            <div className="bg-gradient-to-r from-neutral-50 to-neutral-100 px-6 py-4 border-b border-neutral-200">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-primary-100 rounded-lg flex items-center justify-center">
                  <FileText className="w-4 h-4 text-primary-600" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-neutral-900">Report Statistics</h3>
                  <p className="text-sm text-neutral-600">Content analysis</p>
                </div>
              </div>
            </div>
            
            <div className="p-6">
              <div className="grid grid-cols-1 gap-4">
                {[
                  { 
                    label: 'Words', 
                    value: editContent.split(' ').length.toLocaleString(),
                    icon: '📝',
                    color: 'primary'
                  },
                  { 
                    label: 'Characters', 
                    value: editContent.length.toLocaleString(),
                    icon: '🔤',
                    color: 'success'
                  },
                  { 
                    label: 'Sections', 
                    value: (editContent.match(/^#+\s/gm) || []).length,
                    icon: '📋',
                    color: 'warning'
                  },
                  { 
                    label: 'References', 
                    value: (editContent.match(/\[([^\]]*)\]\(([^\)]*)\)/g) || []).length,
                    icon: '🔗',
                    color: 'error'
                  },
                  { 
                    label: 'Annotations', 
                    value: annotations.length,
                    icon: '💬',
                    color: 'neutral'
                  }
                ].map(({ label, value, icon, color }) => (
                  <div key={label} className={`p-4 rounded-xl border-2 transition-all duration-200 hover:shadow-md ${
                    color === 'primary' ? 'bg-primary-50/50 border-primary-200 hover:border-primary-300' :
                    color === 'success' ? 'bg-success-50/50 border-success-200 hover:border-success-300' :
                    color === 'warning' ? 'bg-warning-50/50 border-warning-200 hover:border-warning-300' :
                    color === 'error' ? 'bg-error-50/50 border-error-200 hover:border-error-300' :
                    'bg-neutral-50/50 border-neutral-200 hover:border-neutral-300'
                  }`}>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <span className="text-lg">{icon}</span>
                        <span className="text-sm font-medium text-neutral-700">{label}</span>
                      </div>
                      <span className={`text-lg font-bold ${
                        color === 'primary' ? 'text-primary-700' :
                        color === 'success' ? 'text-success-700' :
                        color === 'warning' ? 'text-warning-700' :
                        color === 'error' ? 'text-error-700' :
                        'text-neutral-700'
                      }`}>
                        {value}
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Reading Time Estimate */}
              <div className="mt-6 p-4 bg-gradient-to-r from-primary-50 to-primary-100 rounded-xl border border-primary-200">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span className="text-lg">⏱️</span>
                    <span className="text-sm font-medium text-primary-900">Reading Time</span>
                  </div>
                  <span className="text-lg font-bold text-primary-700">
                    {Math.ceil(editContent.split(' ').length / 200)} min
                  </span>
                </div>
                <p className="text-xs text-primary-600 mt-1">
                  Based on average reading speed of 200 words per minute
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Annotation Form Modal - Enhanced Design */}
      {showAnnotationForm && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-fade-in">
          <div className="bg-white rounded-2xl shadow-strong max-w-lg w-full transform animate-slide-up">
            {/* Modal Header */}
            <div className="bg-gradient-to-r from-primary-50 to-primary-100 px-6 py-4 rounded-t-2xl border-b border-primary-200">
              <div className="flex justify-between items-center">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-primary-500 rounded-lg flex items-center justify-center">
                    <MessageSquare className="w-4 h-4 text-white" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-neutral-900">添加注释</h3>
                    <p className="text-sm text-neutral-600">为选中的文本添加注释或评论</p>
                  </div>
                </div>
                <button
                  onClick={() => {
                    setShowAnnotationForm(false);
                    setNewAnnotation('');
                    setSelectedText('');
                  }}
                  className="p-2 text-neutral-400 hover:text-neutral-600 hover:bg-white/50 rounded-lg transition-all duration-200"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Modal Content */}
            <div className="p-6">
              {selectedText && (
                <div className="mb-6 p-4 bg-neutral-50 border border-neutral-200 rounded-xl">
                  <div className="flex items-start space-x-2">
                    <div className="w-5 h-5 bg-warning-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                      <span className="text-xs text-warning-600">📝</span>
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-neutral-700 mb-1">Selected text:</p>
                      <p className="text-sm text-neutral-600 italic bg-white p-2 rounded-lg border border-neutral-200">
                        "{selectedText}"
                      </p>
                    </div>
                  </div>
                </div>
              )}
              
              <div className="space-y-2">
                <label className="label-base">Your annotation</label>
                <textarea
                  value={newAnnotation}
                  onChange={(e) => setNewAnnotation(e.target.value)}
                  placeholder="Write your annotation, comment, or feedback here..."
                  className="w-full h-32 p-4 text-sm border-2 border-neutral-200 rounded-xl bg-white placeholder-neutral-400 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 resize-none"
                  autoFocus
                />
                <div className="flex justify-between items-center text-xs text-neutral-500">
                  <span>{newAnnotation.length} characters</span>
                  <span>Markdown supported</span>
                </div>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="bg-neutral-50 px-6 py-4 rounded-b-2xl border-t border-neutral-200">
              <div className="flex justify-end space-x-3">
                <button
                  onClick={() => {
                    setShowAnnotationForm(false);
                    setNewAnnotation('');
                    setSelectedText('');
                  }}
                  className="btn-secondary"
                >
                  取消
                </button>
                <button
                  onClick={addAnnotation}
                  disabled={!newAnnotation.trim()}
                  className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <MessageSquare className="w-4 h-4 mr-2" />
                  添加注释
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Share Modal - Enhanced Design */}
      {showShareModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-fade-in">
          <div className="bg-white rounded-2xl shadow-strong max-w-lg w-full transform animate-slide-up">
            {/* Modal Header */}
            <div className="bg-gradient-to-r from-success-50 to-success-100 px-6 py-4 rounded-t-2xl border-b border-success-200">
              <div className="flex justify-between items-center">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-success-500 rounded-lg flex items-center justify-center">
                    <Share2 className="w-4 h-4 text-white" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-neutral-900">Share Report</h3>
                    <p className="text-sm text-neutral-600">Send this report to collaborators</p>
                  </div>
                </div>
                <button
                  onClick={() => setShowShareModal(false)}
                  className="p-2 text-neutral-400 hover:text-neutral-600 hover:bg-white/50 rounded-lg transition-all duration-200"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Modal Content */}
            <div className="p-6">
              <div className="space-y-4">
                <div>
                  <label className="label-base flex items-center space-x-2">
                    <Users className="w-4 h-4 text-neutral-500" />
                    <span>Recipients</span>
                  </label>
                  <textarea
                    value={shareRecipients.join(', ')}
                    onChange={(e) => setShareRecipients(e.target.value.split(',').map(email => email.trim()))}
                    placeholder="Enter email addresses separated by commas...
example@company.com, colleague@domain.org"
                    className="w-full h-24 p-4 text-sm border-2 border-neutral-200 rounded-xl bg-white placeholder-neutral-400 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-success-500 focus:border-success-500 resize-none"
                  />
                  <div className="flex justify-between items-center mt-2 text-xs text-neutral-500">
                    <span>{shareRecipients.filter(email => email.trim()).length} recipients</span>
                    <span>Separate multiple emails with commas</span>
                  </div>
                </div>

                {/* Share Options */}
                <div className="bg-neutral-50 p-4 rounded-xl border border-neutral-200">
                  <h4 className="text-sm font-medium text-neutral-900 mb-3">Share Options</h4>
                  <div className="space-y-2">
                    <label className="flex items-center space-x-3">
                      <input type="checkbox" defaultChecked className="rounded border-neutral-300 text-success-600 focus:ring-success-500" />
                      <span className="text-sm text-neutral-700">Include PDF attachment</span>
                    </label>
                    <label className="flex items-center space-x-3">
                      <input type="checkbox" defaultChecked className="rounded border-neutral-300 text-success-600 focus:ring-success-500" />
                      <span className="text-sm text-neutral-700">Allow recipients to add annotations</span>
                    </label>
                    <label className="flex items-center space-x-3">
                      <input type="checkbox" className="rounded border-neutral-300 text-success-600 focus:ring-success-500" />
                      <span className="text-sm text-neutral-700">Send notification when report is updated</span>
                    </label>
                  </div>
                </div>

                {/* Preview */}
                <div className="bg-primary-50 p-4 rounded-xl border border-primary-200">
                  <h4 className="text-sm font-medium text-primary-900 mb-2">Email Preview</h4>
                  <div className="text-xs text-primary-700 space-y-1">
                    <p><strong>Subject:</strong> Research Report: {topic}</p>
                    <p><strong>From:</strong> TTD-DR Framework</p>
                    <p><strong>Content:</strong> Report shared with annotations and export options</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="bg-neutral-50 px-6 py-4 rounded-b-2xl border-t border-neutral-200">
              <div className="flex justify-end space-x-3">
                <button
                  onClick={() => setShowShareModal(false)}
                  className="btn-secondary"
                >
                  Cancel
                </button>
                <button
                  onClick={handleShare}
                  disabled={shareRecipients.length === 0 || shareRecipients.every(email => !email)}
                  className="btn-success disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Share2 className="w-4 h-4 mr-2" />
                  Send Report
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};