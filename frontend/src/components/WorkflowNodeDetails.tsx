import React from 'react';
import { X, Clock, CheckCircle, AlertCircle, Play, FileText, BarChart3 } from 'lucide-react';
import { TTDRState } from '../types';

interface WorkflowNodeDetailsProps {
  nodeId: string;
  nodeName: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  state: TTDRState | null;
  onClose: () => void;
}

export const WorkflowNodeDetails: React.FC<WorkflowNodeDetailsProps> = ({
  nodeId,
  nodeName,
  status,
  state,
  onClose
}) => {
  const getNodeDescription = (nodeId: string): string => {
    switch (nodeId) {
      case 'draft_generator':
        return '创建初始研究草稿，为研究过程提供结构化基础。';
      case 'gap_analyzer':
        return '识别草稿中需要额外信息或改进的特定区域。';
      case 'retrieval_engine':
        return '从外部来源检索相关信息以填补已识别的空白。';
      case 'information_integrator':
        return '将检索到的信息无缝整合到现有草稿结构中。';
      case 'quality_assessor':
        return '评估草稿质量并确定是否需要进一步迭代。';
      case 'quality_check':
        return '决策节点，确定是继续迭代还是进行最终综合。';
      case 'self_evolution_enhancer':
        return '应用自我改进算法来增强框架组件。';
      case 'report_synthesizer':
        return '生成具有正确格式和引用的最终完善研究报告。';
      default:
        return 'TTD-DR研究工作流节点。';
    }
  };

  const getNodeData = (nodeId: string) => {
    if (!state) return null;

    switch (nodeId) {
      case 'draft_generator':
        return {
          input: `Topic: ${state.topic}`,
          output: state.current_draft ? `Draft created (${state.current_draft.metadata.word_count} words)` : 'No draft yet',
          details: state.current_draft ? [
            `Quality Score: ${state.current_draft.quality_score.toFixed(2)}`,
            `Sections: ${state.current_draft.structure.sections.length}`,
            `Complexity: ${state.current_draft.structure.complexity_level}`,
          ] : []
        };
      
      case 'gap_analyzer':
        return {
          input: state.current_draft ? 'Current draft' : 'No input',
          output: `${state.information_gaps.length} gaps identified`,
          details: state.information_gaps.map(gap => 
            `${gap.gap_type.toUpperCase()}: ${gap.description} (${gap.priority})`
          ).slice(0, 5)
        };
      
      case 'retrieval_engine':
        return {
          input: `${state.information_gaps.length} search queries`,
          output: `${state.retrieved_info.length} sources retrieved`,
          details: state.retrieved_info.slice(0, 5).map((info: any, index) => 
            `Source ${index + 1}: ${info.source?.title || 'Unknown'}`
          )
        };
      
      case 'information_integrator':
        return {
          input: `Draft + ${state.retrieved_info.length} sources`,
          output: state.current_draft ? `Updated draft (iteration ${state.iteration_count})` : 'No output',
          details: [
            `Integration iteration: ${state.iteration_count}`,
            `Sources integrated: ${state.retrieved_info.length}`,
          ]
        };
      
      case 'quality_assessor':
        return {
          input: 'Current draft',
          output: state.quality_metrics ? 'Quality metrics calculated' : 'No metrics',
          details: state.quality_metrics ? [
            `Completeness: ${(state.quality_metrics.completeness * 100).toFixed(1)}%`,
            `Coherence: ${(state.quality_metrics.coherence * 100).toFixed(1)}%`,
            `Accuracy: ${(state.quality_metrics.accuracy * 100).toFixed(1)}%`,
            `Overall Score: ${(state.quality_metrics.overall_score * 100).toFixed(1)}%`,
          ] : []
        };
      
      case 'quality_check':
        return {
          input: state.quality_metrics ? 'Quality metrics' : 'No input',
          output: state.quality_metrics ? 
            (state.quality_metrics.overall_score >= state.requirements.quality_threshold ? 
              'Proceed to synthesis' : 'Continue iteration') : 'No decision',
          details: [
            `Quality threshold: ${(state.requirements.quality_threshold * 100).toFixed(1)}%`,
            `Current score: ${state.quality_metrics ? (state.quality_metrics.overall_score * 100).toFixed(1) : 'N/A'}%`,
            `Max iterations: ${state.requirements.max_iterations}`,
            `Current iteration: ${state.iteration_count}`,
          ]
        };
      
      case 'self_evolution_enhancer':
        return {
          input: 'Quality metrics and evolution history',
          output: `${state.evolution_history.length} evolution records`,
          details: state.evolution_history.slice(-3).map((record: any, index) => 
            `Evolution ${state.evolution_history.length - index}: ${record.component || 'Component'} enhanced`
          )
        };
      
      case 'report_synthesizer':
        return {
          input: 'Final draft and metadata',
          output: state.final_report ? 'Final report generated' : 'No report yet',
          details: state.final_report ? [
            `Report length: ${state.final_report.length} characters`,
            `Total iterations: ${state.iteration_count}`,
            `Sources used: ${state.retrieved_info.length}`,
          ] : []
        };
      
      default:
        return null;
    }
  };

  const nodeData = getNodeData(nodeId);

  const getStatusIcon = () => {
    switch (status) {
      case 'running':
        return <Play className="w-5 h-5 text-blue-500" />;
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return <Clock className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'running':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'completed':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'error':
        return 'text-red-600 bg-red-50 border-red-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div className="flex items-center space-x-3">
            {getStatusIcon()}
            <div>
              <h2 className="text-xl font-bold text-gray-900">{nodeName}</h2>
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize ${getStatusColor()}`}>
                {status}
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-neutral-400 hover:text-neutral-600 transition-colors p-1 rounded-lg hover:bg-neutral-100"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Description */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Description</h3>
            <p className="text-gray-600">{getNodeDescription(nodeId)}</p>
          </div>

          {/* Node Data */}
          {nodeData && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900">Execution Details</h3>
              
              {/* Input */}
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <FileText className="w-4 h-4 text-gray-500" />
                  <h4 className="font-medium text-gray-900">Input</h4>
                </div>
                <p className="text-sm text-gray-600">{nodeData.input}</p>
              </div>

              {/* Output */}
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <BarChart3 className="w-4 h-4 text-gray-500" />
                  <h4 className="font-medium text-gray-900">Output</h4>
                </div>
                <p className="text-sm text-gray-600">{nodeData.output}</p>
              </div>

              {/* Details */}
              {nodeData.details.length > 0 && (
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-2">Details</h4>
                  <ul className="space-y-1">
                    {nodeData.details.map((detail, index) => (
                      <li key={index} className="text-sm text-gray-600 flex items-start">
                        <span className="w-2 h-2 bg-gray-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                        {detail}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Error Log */}
          {state?.error_log && state.error_log.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Error Log</h3>
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 max-h-40 overflow-y-auto">
                {state.error_log.map((error, index) => (
                  <div key={index} className="text-sm text-red-700 mb-1">
                    {error}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Execution Time */}
          {status === 'completed' && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="flex items-center space-x-2">
                <CheckCircle className="w-4 h-4 text-green-500" />
                <span className="text-sm font-medium text-green-800">Node completed successfully</span>
              </div>
            </div>
          )}

          {status === 'running' && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex items-center space-x-2">
                <Play className="w-4 h-4 text-blue-500" />
                <span className="text-sm font-medium text-blue-800">Node is currently executing...</span>
              </div>
              <div className="mt-2">
                <div className="w-full bg-blue-200 rounded-full h-2">
                  <div className="bg-blue-500 h-2 rounded-full animate-pulse" style={{ width: '60%' }}></div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end p-6 border-t bg-neutral-50">
          <button
            onClick={onClose}
            className="btn-secondary"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};