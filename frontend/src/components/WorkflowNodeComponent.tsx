import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { 
  FileText, 
  Search, 
  Download, 
  Merge, 
  CheckCircle, 
  GitBranch, 
  Zap, 
  BookOpen,
  Play,
  Pause,
  AlertCircle,
  Clock,
  Loader2
} from 'lucide-react';

interface WorkflowNodeData {
  label: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  isActive: boolean;
  onClick: () => void;
  progress?: number;
}

export const WorkflowNodeComponent: React.FC<NodeProps<WorkflowNodeData>> = ({ 
  data, 
  id 
}) => {
  const { label, status, isActive, onClick, progress } = data;

  const getNodeIcon = (nodeId: string) => {
    const iconClass = "w-6 h-6";
    switch (nodeId) {
      case 'draft_generator':
        return <FileText className={iconClass} />;
      case 'gap_analyzer':
        return <Search className={iconClass} />;
      case 'retrieval_engine':
        return <Download className={iconClass} />;
      case 'information_integrator':
        return <Merge className={iconClass} />;
      case 'quality_assessor':
        return <CheckCircle className={iconClass} />;
      case 'quality_check':
        return <GitBranch className={iconClass} />;
      case 'self_evolution_enhancer':
        return <Zap className={iconClass} />;
      case 'report_synthesizer':
        return <BookOpen className={iconClass} />;
      default:
        return <Clock className={iconClass} />;
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'running':
        return <Loader2 className="w-5 h-5 text-primary-600 animate-spin" />;
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-success-600" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-error-600" />;
      default:
        return <Clock className="w-5 h-5 text-neutral-400" />;
    }
  };

  const getNodeStyles = () => {
    const baseStyles = "group relative bg-white rounded-2xl border-2 transition-all duration-300 cursor-pointer overflow-hidden min-w-[280px] hover:shadow-strong hover:-translate-y-1 active:scale-[0.98]";
    
    switch (status) {
      case 'running':
        return `${baseStyles} border-primary-300 shadow-medium ring-4 ring-primary-100 bg-gradient-to-br from-primary-50/80 to-white`;
      case 'completed':
        return `${baseStyles} border-success-300 shadow-medium bg-gradient-to-br from-success-50/80 to-white hover:border-success-400`;
      case 'error':
        return `${baseStyles} border-error-300 shadow-medium bg-gradient-to-br from-error-50/80 to-white hover:border-error-400`;
      default:
        return `${baseStyles} border-neutral-200 shadow-soft hover:border-neutral-300 hover:shadow-medium`;
    }
  };

  const getIconContainerStyles = () => {
    switch (status) {
      case 'running':
        return 'bg-primary-100 text-primary-700 ring-2 ring-primary-200';
      case 'completed':
        return 'bg-success-100 text-success-700 ring-2 ring-success-200';
      case 'error':
        return 'bg-error-100 text-error-700 ring-2 ring-error-200';
      default:
        return 'bg-neutral-100 text-neutral-600 group-hover:bg-neutral-200 group-hover:text-neutral-700';
    }
  };

  const getTextColor = () => {
    switch (status) {
      case 'running':
        return 'text-primary-900';
      case 'completed':
        return 'text-success-900';
      case 'error':
        return 'text-error-900';
      default:
        return 'text-neutral-800';
    }
  };

  const getStatusBadgeStyles = () => {
    switch (status) {
      case 'running':
        return 'bg-primary-100 text-primary-700 border-primary-200';
      case 'completed':
        return 'bg-success-100 text-success-700 border-success-200';
      case 'error':
        return 'bg-error-100 text-error-700 border-error-200';
      default:
        return 'bg-neutral-100 text-neutral-600 border-neutral-200';
    }
  };

  return (
    <>
      {/* Enhanced input handle */}
      {id !== 'draft_generator' && (
        <Handle
          type="target"
          position={Position.Top}
          className="w-5 h-5 !bg-white border-3 border-neutral-300 shadow-medium hover:!bg-primary-500 hover:border-primary-400 transition-all duration-200 hover:scale-110"
        />
      )}
      
      <div 
        className={getNodeStyles()}
        onClick={onClick}
        role="button"
        tabIndex={0}
        aria-label={`${label} - ${status} ${progress ? `(${Math.round(progress * 100)}% complete)` : ''}`}
        aria-describedby={`${id}-description`}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onClick();
          }
        }}
      >
        {/* Header section with icon and status */}
        <div className="flex items-start justify-between p-6 pb-4">
          <div className="flex items-center space-x-4">
            {/* Enhanced icon container */}
            <div className={`p-3 rounded-xl transition-all duration-300 ${getIconContainerStyles()}`}>
              {getNodeIcon(id)}
            </div>
            
            {/* Node title and description */}
            <div className="flex-1">
              <h3 className={`font-bold text-lg leading-tight ${getTextColor()}`}>
                {label}
              </h3>
              <p id={`${id}-description`} className="text-sm text-neutral-600 mt-1 opacity-80">
                {getNodeDescription(id)}
              </p>
            </div>
          </div>
          
          {/* Enhanced status indicator */}
          <div className="flex flex-col items-end space-y-2">
            <div className="flex items-center space-x-2">
              {getStatusIcon()}
            </div>
            
            {/* Status badge */}
            <div className={`px-3 py-1 rounded-full text-xs font-semibold border transition-all duration-200 ${getStatusBadgeStyles()}`}>
              {status.charAt(0).toUpperCase() + status.slice(1)}
            </div>
          </div>
        </div>
        
        {/* Progress section for running nodes */}
        {status === 'running' && (
          <div className="px-6 pb-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-primary-700">Processing...</span>
              {progress && (
                <span className="text-sm font-bold text-primary-800">{Math.round(progress * 100)}%</span>
              )}
            </div>
            <div className="w-full bg-primary-100 rounded-full h-2 overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-primary-400 to-primary-600 rounded-full transition-all duration-500 ease-out"
                style={{ width: `${progress ? progress * 100 : 0}%` }}
              />
            </div>
          </div>
        )}
        
        {/* Active pulse indicator */}
        {isActive && (
          <>
            <div className="absolute -top-1 -right-1">
              <div className="w-6 h-6 bg-primary-500 rounded-full animate-bounce-soft shadow-strong flex items-center justify-center">
                <div className="w-3 h-3 bg-white rounded-full"></div>
              </div>
            </div>
            {/* Animated border for active nodes */}
            <div className="absolute inset-0 rounded-2xl border-2 border-primary-400 animate-pulse"></div>
          </>
        )}
        
        {/* Hover glow effect */}
        <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-transparent via-transparent to-neutral-50/50 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
      </div>
      
      {/* Enhanced output handle */}
      {id !== 'report_synthesizer' && (
        <Handle
          type="source"
          position={Position.Bottom}
          className="w-5 h-5 !bg-white border-3 border-neutral-300 shadow-medium hover:!bg-primary-500 hover:border-primary-400 transition-all duration-200 hover:scale-110"
        />
      )}
    </>
  );
};

// Helper function to get node descriptions
function getNodeDescription(nodeId: string): string {
  switch (nodeId) {
    case 'draft_generator':
      return 'Creates initial research draft';
    case 'gap_analyzer':
      return 'Identifies information gaps';
    case 'retrieval_engine':
      return 'Retrieves relevant information';
    case 'information_integrator':
      return 'Integrates new information';
    case 'quality_assessor':
      return 'Assesses content quality';
    case 'quality_check':
      return 'Validates quality threshold';
    case 'self_evolution_enhancer':
      return 'Enhances system capabilities';
    case 'report_synthesizer':
      return 'Generates final report';
    default:
      return 'Processing node';
  }
}