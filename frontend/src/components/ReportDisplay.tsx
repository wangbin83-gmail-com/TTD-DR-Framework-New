import React, { useState } from 'react';
import { Download, FileText, Share2, Eye, EyeOff, Settings } from 'lucide-react';
import { saveAs } from 'file-saver';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import { ReportManagement } from './ReportManagement';

interface ReportDisplayProps {
  report: string;
  topic: string;
  workflowId: string;
}

export const ReportDisplay: React.FC<ReportDisplayProps> = ({ report, topic, workflowId }) => {
  const [useAdvancedMode, setUseAdvancedMode] = useState(false);

  if (useAdvancedMode) {
    return (
      <ReportManagement
        report={report}
        topic={topic}
        workflowId={workflowId}
        onSave={(content) => {
          console.log('Report saved:', content);
        }}
        onAddAnnotation={(annotation) => {
          console.log('Annotation added:', annotation);
        }}
        onResolveAnnotation={(annotationId) => {
          console.log('Annotation resolved:', annotationId);
        }}
        onShare={(recipients) => {
          console.log('Report shared with:', recipients);
        }}
      />
    );
  }
  
  const [showRawMarkdown, setShowRawMarkdown] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  const exportToPDF = async () => {
    setIsExporting(true);
    try {
      const element = document.getElementById('report-content');
      if (!element) return;

      const canvas = await html2canvas(element, {
        scale: 2,
        useCORS: true,
        allowTaint: true
      });

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

      pdf.save(`${topic.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_report.pdf`);
    } catch (error) {
      console.error('导出PDF时出错:', error);
      alert('导出PDF时出错。请重试。');
    } finally {
      setIsExporting(false);
    }
  };

  const exportToMarkdown = () => {
    const blob = new Blob([report], { type: 'text/markdown;charset=utf-8' });
    saveAs(blob, `${topic.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_report.md`);
  };

  const exportToWord = () => {
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
        ${convertMarkdownToHTML(report)}
      </body>
      </html>
    `;
    
    const blob = new Blob([htmlContent], { type: 'application/msword;charset=utf-8' });
    saveAs(blob, `${topic.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_report.doc`);
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

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(report);
      alert('报告已复制到剪贴板!');
    } catch (error) {
      console.error('Error copying to clipboard:', error);
      alert('复制到剪贴板时出错。请重试。');
    }
  };

  const shareReport = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: `研究报告: ${topic}`,
          text: report.substring(0, 200) + '...',
          url: window.location.href
        });
      } catch (error) {
        console.error('Error sharing:', error);
      }
    } else {
      copyToClipboard();
    }
  };

  return (
    <div className="max-w-5xl mx-auto p-4 sm:p-6 lg:p-8">
      {/* Header */}
      <div className="bg-white/80 backdrop-blur-sm rounded-xl shadow-lg p-6 mb-6 border border-gray-200/80">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-4">
          <div className="mb-4 md:mb-0">
            <h1 className="text-2xl md:text-3xl font-bold text-gray-900">研究报告</h1>
            <p className="text-lg text-gray-600 mt-1">{topic}</p>
            <p className="text-sm text-gray-500 mt-2">工作流ID: <span className="font-mono bg-gray-100 px-2 py-1 rounded-md">{workflowId}</span></p>
          </div>
          
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
            <button
              onClick={() => setShowRawMarkdown(!showRawMarkdown)}
              className="flex items-center justify-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md shadow-sm text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-all duration-150 ease-in-out"
              title={showRawMarkdown ? '显示格式化视图' : '显示原始Markdown'}
            >
              {showRawMarkdown ? <Eye className="w-5 h-5 mr-2" /> : <EyeOff className="w-5 h-5 mr-2" />}
              <span>{showRawMarkdown ? '格式化' : '原始Markdown'}</span>
            </button>
            <button
              onClick={() => setUseAdvancedMode(true)}
              className="flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-all duration-150 ease-in-out"
              title="切换到高级报告管理"
            >
              <Settings className="w-5 h-5 mr-2" />
              <span>高级模式</span>
            </button>
          </div>
        </div>

        {/* Export Options */}
        <div className="mt-6 pt-4 border-t border-gray-200">
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-sm font-medium text-gray-600">导出为:</span>
            <button
              onClick={exportToPDF}
              disabled={isExporting}
              className="flex items-center justify-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50 transition-all duration-150 ease-in-out"
            >
              <Download className="w-4 h-4 mr-1.5" />
              PDF
            </button>
            <button
              onClick={exportToMarkdown}
              className="flex items-center justify-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-all duration-150 ease-in-out"
            >
              <FileText className="w-4 h-4 mr-1.5" />
              Markdown
            </button>
            <button
              onClick={exportToWord}
              className="flex items-center justify-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 transition-all duration-150 ease-in-out"
            >
              <FileText className="w-4 h-4 mr-1.5" />
              Word
            </button>
            <button
              onClick={shareReport}
              className="flex items-center justify-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md shadow-sm text-white bg-purple-500 hover:bg-purple-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 transition-all duration-150 ease-in-out"
            >
              <Share2 className="w-4 h-4 mr-1.5" />
              Share
            </button>
          </div>
        </div>
      </div>

      {/* Report Content */}
      <div id="report-content" className="bg-gray-800/80 backdrop-blur-sm rounded-xl shadow-lg p-6 md:p-8 border border-gray-700/80">
        {showRawMarkdown ? (
          <pre className="whitespace-pre-wrap text-sm font-mono bg-gray-700/50 p-4 rounded-lg overflow-x-auto text-gray-100">
            {report}
          </pre>
        ) : (
          <div
            className="prose prose-lg max-w-none prose-invert prose-headings:text-gray-100 prose-p:text-gray-300 prose-a:text-indigo-400 hover:prose-a:text-indigo-300 prose-strong:text-gray-100 prose-ul:text-gray-300 prose-ol:text-gray-300 prose-li:text-gray-300 prose-blockquote:text-gray-400 prose-blockquote:border-l-gray-500 prose-code:text-gray-100 prose-pre:bg-gray-900 prose-pre:text-gray-100"
            dangerouslySetInnerHTML={{ __html: convertMarkdownToHTML(report) }}
          />
        )}
      </div>
    </div>
  );
};