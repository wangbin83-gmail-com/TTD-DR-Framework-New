import React, { useEffect, useState } from 'react';
import { CheckCircle, Clock, AlertCircle, Pause, RotateCcw, BarChart3, GitBranch } from 'lucide-react';
import { WorkflowStatus, TTDRState } from '../types';
import { WorkflowVisualization } from './WorkflowVisualization';

interface WorkflowDashboardProps {
  workflowId: string;
  status: WorkflowStatus;
  state: TTDRState | null;
  onStop: () => void;
  onRestart: () => void;
}

interface WorkflowNode {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'error';
}

export const WorkflowDashboard: React.FC<WorkflowDashboardProps> = ({
  workflowId,
  status,
  state,
  onStop,
  onRestart
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'visualization'>('overview');
  const [nodes, setNodes] = useState<WorkflowNode[]>([
    {
      id: 'draft_generator',
      name: '草稿生成',
      description: '创建初步的研究结构和框架',
      status: 'pending'
    },
    {
      id: 'gap_analyzer',
      name: '差距分析',
      description: '识别草稿中的信息缺口',
      status: 'pending'
    },
    {
      id: 'retrieval_engine',
      name: '信息检索',
      description: '从外部来源搜索相关信息',
      status: 'pending'
    },
    {
      id: 'information_integrator',
      name: '信息整合',
      description: '将检索到的信息整合到草稿中',
      status: 'pending'
    },
    {
      id: 'quality_assessor',
      name: '质量评估',
      description: '评估草稿质量和完整性',
      status: 'pending'
    },
    {
      id: 'self_evolution_enhancer',
      name: '自我进化',
      description: '应用学习算法改进组件',
      status: 'pending'
    },
    {
      id: 'report_synthesizer',
      name: '报告综合',
      description: '生成最终完善的研究报告',
      status: 'pending'
    }
  ]);

  useEffect(() => {
    if (status.current_node) {
      setNodes(prevNodes => 
        prevNodes.map(node => {
          if (node.id === status.current_node) {
            return { ...node, status: 'running' };
          } else if (prevNodes.findIndex(n => n.id === node.id) < prevNodes.findIndex(n => n.id === status.current_node!)) {
            return { ...node, status: 'completed' };
          } else {
            return { ...node, status: 'pending' };
          }
        })
      );
    }
  }, [status.current_node]);

  const getStatusIcon = (nodeStatus: string) => {
    switch (nodeStatus) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'running':
        return <Clock className="w-5 h-5 text-blue-500 animate-spin" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return <div className="w-5 h-5 rounded-full border-2 border-gray-300" />;
    }
  };

  const getStatusColor = (nodeStatus: string) => {
    switch (nodeStatus) {
      case 'completed':
        return 'bg-green-50 border-green-200';
      case 'running':
        return 'bg-blue-50 border-blue-200';
      case 'error':
        return 'bg-red-50 border-red-200';
      default:
        return 'bg-gray-50 border-gray-200';
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-4 sm:p-6 lg:p-8 bg-gray-900 min-h-screen">
      {/* Header */}
      <div className="bg-gray-800/80 backdrop-blur-sm rounded-xl shadow-lg p-6 mb-6 border border-gray-700/80">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-4">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold text-gray-100">Research Workflow</h1>
            <p className="text-sm md:text-base text-gray-300 mt-1">Tracking ID: <span className="font-mono bg-gray-700 px-2 py-1 rounded-md">{workflowId}</span></p>
          </div>
          <div className="flex items-center space-x-3 mt-4 sm:mt-0">
            {status.status === 'running' && (
              <button
                onClick={onStop}
                className="flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 transition-all duration-150 ease-in-out"
              >
                <Pause className="w-5 h-5 mr-2" />
                <span>停止</span>
              </button>
            )}
            {(status.status === 'completed' || status.status === 'error') && (
              <button
                onClick={onRestart}
                className="flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-all duration-150 ease-in-out"
              >
                <RotateCcw className="w-5 h-5 mr-2" />
                <span>重新开始</span>
              </button>
            )}
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mt-6">
          <div className="flex justify-between items-center mb-1">
            <span className="text-sm font-medium text-gray-200">Overall Progress</span>
            <span className="text-sm font-bold text-indigo-400">{Math.round(status.progress * 100)}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2.5">
            <div
              className="bg-indigo-500 h-2.5 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${status.progress * 100}%` }}
            />
          </div>
        </div>

        {/* Status Message */}
        <div className="mt-4 p-3 bg-gray-700/50 rounded-lg flex items-center space-x-3 border border-gray-600">
          <div
            className={`w-4 h-4 rounded-full flex-shrink-0 ${status.status === 'running' ? 'bg-blue-400 animate-pulse' : status.status === 'completed' ? 'bg-green-400' : status.status === 'error' ? 'bg-red-400' : 'bg-gray-500'}`}>
          </div>
          <p className="text-sm text-gray-200">{status.message}</p>
        </div>

        {/* Tab Navigation */}
        <div className="mt-6 border-b border-gray-700">
          <nav className="-mb-px flex space-x-6" aria-label="Tabs">
            <button
              onClick={() => setActiveTab('overview')}
              className={`group inline-flex items-center py-3 px-1 border-b-2 font-medium text-sm transition-colors duration-200 ease-in-out ${
                activeTab === 'overview'
                  ? 'border-indigo-400 text-indigo-400'
                  : 'border-transparent text-gray-400 hover:text-gray-200 hover:border-gray-500'
              }`}>
              <BarChart3 className="-ml-0.5 mr-2 h-5 w-5" />
              <span>Overview</span>
            </button>
            <button
              onClick={() => setActiveTab('visualization')}
              className={`group inline-flex items-center py-3 px-1 border-b-2 font-medium text-sm transition-colors duration-200 ease-in-out ${
                activeTab === 'visualization'
                  ? 'border-indigo-400 text-indigo-400'
                  : 'border-transparent text-gray-400 hover:text-gray-200 hover:border-gray-500'
              }`}>
              <GitBranch className="-ml-0.5 mr-2 h-5 w-5" />
              <span>Workflow Graph</span>
            </button>
          </nav>
        </div>
      </div>

      {/* Tab Content */}
      <div className="mt-6">
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Workflow Steps */}
            <div className="bg-gray-800/80 backdrop-blur-sm rounded-xl shadow-lg p-6 border border-gray-700/80">
              <h2 className="text-xl font-semibold text-gray-100 mb-4">工作流步骤</h2>
              <div className="space-y-4">
                {nodes.map((node, index) => (
                  <div
                    key={node.id}
                    className={`p-4 rounded-lg border-2 transition-all duration-300 ${getStatusColor(node.status)}`}
                  >
                    <div className="flex items-center space-x-3">
                      <div className="flex-shrink-0">
                        {getStatusIcon(node.status)}
                      </div>
                      <div className="flex-1">
                        <h3 className="text-sm font-medium text-gray-100">{node.name}</h3>
                        <p className="text-sm text-gray-300 mt-1">{node.description}</p>
                      </div>
                      <div className="text-xs text-gray-400">
                        Step {index + 1}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Current Status Details */}
            <div className="bg-gray-800/80 backdrop-blur-sm rounded-xl shadow-lg p-6 border border-gray-700/80">
              <h2 className="text-xl font-semibold text-gray-100 mb-4">当前状态</h2>
              <div className="space-y-4">
                <div>
                  <h3 className="text-sm font-medium text-gray-300">当前节点</h3>
                  <p className="text-sm text-gray-100 mt-1">
                    {status.current_node ? nodes.find(n => n.id === status.current_node)?.name : 'Initializing...'}
                  </p>
                </div>
                
                {state && (
                  <>
                    <div>
                      <h3 className="text-sm font-medium text-gray-300">迭代</h3>
                      <p className="text-sm text-gray-100 mt-1">{state.iteration_count} / {state.requirements.max_iterations}</p>
                    </div>
                    
                    <div>
                      <h3 className="text-sm font-medium text-gray-300">质量评分</h3>
                      <p className="text-sm text-gray-100 mt-1">
                        {state.quality_metrics ? `${Math.round(state.quality_metrics.overall_score * 100)}%` : 'Calculating...'}
                      </p>
                    </div>
                    
                    <div>
                      <h3 className="text-sm font-medium text-gray-300">使用来源</h3>
                      <p className="text-sm text-gray-100 mt-1">{state.retrieved_info?.length || 0}</p>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'visualization' && (
          <div className="bg-gray-800/80 backdrop-blur-sm rounded-xl shadow-lg p-6 border border-gray-700/80">
            <h2 className="text-xl font-semibold text-gray-100 mb-4">工作流可视化</h2>
            <WorkflowVisualization 
              workflowId={workflowId}
              status={status}
              state={state}
            />
          </div>
        )}
      </div>
    </div>
  );
};