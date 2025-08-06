import React, { useCallback, useEffect, useMemo, useState } from 'react';
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Connection,
  useNodesState,
  useEdgesState,
  Controls,
  MiniMap,
  Background,
  BackgroundVariant,
  Panel,
  NodeTypes,
  MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { WorkflowStatus, TTDRState } from '../types';
import { WorkflowNodeComponent } from './WorkflowNodeComponent';
import { WorkflowNodeDetails } from './WorkflowNodeDetails';
import { Clock, Play, Pause, CheckCircle, XCircle, AlertCircle } from 'lucide-react';

interface WorkflowVisualizationProps {
  workflowId: string;
  status: WorkflowStatus;
  state: TTDRState | null;
  onNodeClick?: (nodeId: string) => void;
}

// Define the workflow nodes based on the TTD-DR framework with improved grid-based positioning
const WORKFLOW_NODES = [
  { id: 'draft_generator', label: 'Draft Generator', position: { x: 400, y: 100 } },
  { id: 'gap_analyzer', label: 'Gap Analyzer', position: { x: 400, y: 240 } },
  { id: 'retrieval_engine', label: 'Retrieval Engine', position: { x: 400, y: 380 } },
  { id: 'information_integrator', label: 'Information Integrator', position: { x: 400, y: 520 } },
  { id: 'quality_assessor', label: 'Quality Assessor', position: { x: 400, y: 660 } },
  { id: 'quality_check', label: 'Quality Check', position: { x: 400, y: 800 } },
  { id: 'self_evolution_enhancer', label: 'Self Evolution Enhancer', position: { x: 750, y: 940 } },
  { id: 'report_synthesizer', label: 'Report Synthesizer', position: { x: 400, y: 1080 } },
];

const WORKFLOW_EDGES = [
  { id: 'e1', source: 'draft_generator', target: 'gap_analyzer' },
  { id: 'e2', source: 'gap_analyzer', target: 'retrieval_engine' },
  { id: 'e3', source: 'retrieval_engine', target: 'information_integrator' },
  { id: 'e4', source: 'information_integrator', target: 'quality_assessor' },
  { id: 'e5', source: 'quality_assessor', target: 'quality_check' },
  { id: 'e6', source: 'quality_check', target: 'gap_analyzer', label: 'Below Threshold' },
  { id: 'e7', source: 'quality_check', target: 'self_evolution_enhancer', label: 'Above Threshold' },
  { id: 'e8', source: 'self_evolution_enhancer', target: 'report_synthesizer' },
];

export const WorkflowVisualization: React.FC<WorkflowVisualizationProps> = ({
  workflowId,
  status,
  state,
  onNodeClick
}) => {
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  // Custom node types
  const nodeTypes: NodeTypes = useMemo(() => ({
    workflowNode: WorkflowNodeComponent,
  }), []);

  // Create nodes with status information
  const initialNodes: Node[] = useMemo(() => {
    return WORKFLOW_NODES.map(node => ({
      id: node.id,
      type: 'workflowNode',
      position: node.position,
      data: {
        label: node.label,
        status: getNodeStatus(node.id, status, state),
        isActive: status.current_node === node.id,
        onClick: () => handleNodeClick(node.id),
      },
      draggable: false,
    }));
  }, [status, state]);

  // Create edges with enhanced styling based on execution flow
  const initialEdges: Edge[] = useMemo(() => {
    return WORKFLOW_EDGES.map(edge => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      label: edge.label,
      animated: isEdgeActive(edge, status),
      style: {
        stroke: getEdgeColor(edge, status, state),
        strokeWidth: isEdgeActive(edge, status) ? 4 : 2,
        strokeDasharray: edge.label ? '8,4' : undefined,
        filter: isEdgeActive(edge, status) ? 'drop-shadow(0 2px 4px rgba(59, 130, 246, 0.3))' : undefined,
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: getEdgeColor(edge, status, state),
        width: 20,
        height: 20,
      },
      labelStyle: {
        fontSize: '12px',
        fontWeight: '600',
        fill: getEdgeLabelColor(edge, status, state),
        backgroundColor: 'rgba(255, 255, 255, 0.9)',
        padding: '4px 8px',
        borderRadius: '8px',
        border: `1px solid ${getEdgeColor(edge, status, state)}`,
      },
      labelBgStyle: {
        fill: 'rgba(255, 255, 255, 0.95)',
        fillOpacity: 0.9,
      },
    }));
  }, [status, state]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Update nodes when status changes
  useEffect(() => {
    setNodes(currentNodes =>
      currentNodes.map(node => ({
        ...node,
        data: {
          ...node.data,
          status: getNodeStatus(node.id, status, state),
          isActive: status.current_node === node.id,
        },
      }))
    );
  }, [status, state, setNodes]);

  // Update edges when status changes
  useEffect(() => {
    setEdges(currentEdges =>
      currentEdges.map(edge => ({
        ...edge,
        animated: isEdgeActive(edge, status),
        style: {
          ...edge.style,
          stroke: getEdgeColor(edge, status, state),
          strokeWidth: isEdgeActive(edge, status) ? 4 : 2,
          filter: isEdgeActive(edge, status) ? 'drop-shadow(0 2px 4px rgba(59, 130, 246, 0.3))' : undefined,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: getEdgeColor(edge, status, state),
          width: 20,
          height: 20,
        },
        labelStyle: {
          fontSize: '12px',
          fontWeight: '600',
          fill: getEdgeLabelColor(edge, status, state),
          backgroundColor: 'rgba(255, 255, 255, 0.9)',
          padding: '4px 8px',
          borderRadius: '8px',
          border: `1px solid ${getEdgeColor(edge, status, state)}`,
        },
      }))
    );
  }, [status, state, setEdges]);

  const handleNodeClick = useCallback((nodeId: string) => {
    setSelectedNode(nodeId);
    setShowDetails(true);
    onNodeClick?.(nodeId);
  }, [onNodeClick]);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  return (
    <div className="h-full w-full bg-gradient-to-br from-neutral-50 to-neutral-100">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{
          padding: 0.2,
          includeHiddenNodes: false,
          minZoom: 0.5,
          maxZoom: 1.5,
        }}
        minZoom={0.3}
        maxZoom={2}
        defaultViewport={{ x: 0, y: 0, zoom: 0.8 }}
        attributionPosition="bottom-left"
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={true}
      >
        <Controls />
        <MiniMap 
          nodeColor={(node) => {
            const nodeStatus = getNodeStatus(node.id, status, state);
            return getNodeColor(nodeStatus);
          }}
          nodeStrokeWidth={3}
          zoomable
          pannable
        />
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} />
        
        {/* Enhanced Status Panel */}
        <Panel position="top-right" className="bg-white/95 backdrop-blur-md p-0 rounded-2xl shadow-strong border border-neutral-200/80 overflow-hidden">
          <div className="min-w-[280px] max-w-[320px] sm:min-w-[320px] sm:max-w-[400px]">
            {/* Header */}
            <div className="bg-gradient-to-r from-primary-50 to-primary-100 p-6 border-b border-primary-200/50">
              <div className="flex items-center space-x-4">
                <div className="p-2 bg-white rounded-xl shadow-soft">
                  <StatusIcon status={status.status} />
                </div>
                <div className="flex-1">
                  <h3 className="font-bold text-primary-900 text-xl">Workflow Status</h3>
                  <div className="flex items-center space-x-2 mt-1">
                    <div className={`w-2 h-2 rounded-full ${getStatusDotColor(status.status)} animate-pulse`}></div>
                    <p className="text-sm text-primary-700 capitalize font-semibold">{status.status}</p>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Content */}
            <div className="p-6 space-y-5">
              {/* Current Node */}
              {status.current_node && (
                <div className="bg-gradient-to-r from-primary-50 to-primary-100 p-4 rounded-xl border border-primary-200 relative overflow-hidden">
                  <div className="relative z-10">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-sm font-bold text-primary-800">Current Node</p>
                      <div className="w-3 h-3 bg-primary-500 rounded-full animate-bounce-soft"></div>
                    </div>
                    <p className="text-base text-primary-900 font-bold">{formatNodeName(status.current_node)}</p>
                  </div>
                  <div className="absolute top-0 right-0 w-20 h-20 bg-primary-200/30 rounded-full -translate-y-10 translate-x-10"></div>
                </div>
              )}
              
              {/* Progress */}
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <p className="text-sm font-bold text-neutral-800">Overall Progress</p>
                  <div className="flex items-center space-x-2">
                    <p className="text-lg font-black text-neutral-900">{Math.round(status.progress * 100)}%</p>
                  </div>
                </div>
                <div className="relative">
                  <div className="w-full bg-neutral-200 rounded-full h-4 overflow-hidden shadow-inner">
                    <div 
                      className="bg-gradient-to-r from-primary-500 via-primary-600 to-primary-700 h-4 rounded-full transition-all duration-700 ease-out shadow-sm relative overflow-hidden"
                      style={{ width: `${status.progress * 100}%` }}
                    >
                      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-pulse"></div>
                    </div>
                  </div>
                  <div className="flex justify-between text-xs text-neutral-500 mt-1">
                    <span>0%</span>
                    <span>50%</span>
                    <span>100%</span>
                  </div>
                </div>
              </div>
              
              {/* Metrics Grid */}
              <div className="grid grid-cols-2 gap-4">
                {state && (
                  <div className="bg-neutral-50 p-4 rounded-xl border border-neutral-200">
                    <div className="text-center">
                      <p className="text-2xl font-black text-neutral-900">{state.iteration_count}</p>
                      <p className="text-xs font-semibold text-neutral-600 uppercase tracking-wide">Iterations</p>
                    </div>
                  </div>
                )}
                
                <div className="bg-neutral-50 p-4 rounded-xl border border-neutral-200">
                  <div className="text-center">
                    <p className="text-2xl font-black text-neutral-900">{getActiveNodesCount(status)}</p>
                    <p className="text-xs font-semibold text-neutral-600 uppercase tracking-wide">Active Nodes</p>
                  </div>
                </div>
              </div>
              
              {/* Status Message */}
              <div className="bg-gradient-to-br from-neutral-50 to-neutral-100 p-4 rounded-xl border border-neutral-200">
                <div className="flex items-start space-x-3">
                  <div className="p-1 bg-neutral-200 rounded-lg mt-0.5">
                    <AlertCircle className="w-4 h-4 text-neutral-600" />
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-bold text-neutral-800 mb-1">Status Message</p>
                    <p className="text-sm text-neutral-700 leading-relaxed">{status.message}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Panel>

        {/* Enhanced Legend Panel */}
        <Panel position="bottom-right" className="bg-white/95 backdrop-blur-md p-0 rounded-2xl shadow-strong border border-neutral-200/80 overflow-hidden">
          <div className="min-w-[200px]">
            {/* Header */}
            <div className="bg-gradient-to-r from-neutral-50 to-neutral-100 p-4 border-b border-neutral-200/50">
              <h4 className="font-bold text-neutral-900 text-sm">Status Legend</h4>
            </div>
            
            {/* Content */}
            <div className="p-4 space-y-3">
              <div className="flex items-center space-x-3 group">
                <div className="relative">
                  <div className="w-4 h-4 bg-neutral-400 rounded-full shadow-soft transition-all duration-200 group-hover:scale-110"></div>
                </div>
                <span className="text-sm font-medium text-neutral-700 group-hover:text-neutral-900 transition-colors">Pending</span>
              </div>
              
              <div className="flex items-center space-x-3 group">
                <div className="relative">
                  <div className="w-4 h-4 bg-primary-500 rounded-full shadow-soft animate-pulse transition-all duration-200 group-hover:scale-110"></div>
                  <div className="absolute inset-0 w-4 h-4 bg-primary-400 rounded-full animate-ping opacity-30"></div>
                </div>
                <span className="text-sm font-medium text-neutral-700 group-hover:text-neutral-900 transition-colors">Running</span>
              </div>
              
              <div className="flex items-center space-x-3 group">
                <div className="relative">
                  <div className="w-4 h-4 bg-success-500 rounded-full shadow-soft transition-all duration-200 group-hover:scale-110"></div>
                  <div className="absolute inset-1 w-2 h-2 bg-white rounded-full"></div>
                </div>
                <span className="text-sm font-medium text-neutral-700 group-hover:text-neutral-900 transition-colors">Completed</span>
              </div>
              
              <div className="flex items-center space-x-3 group">
                <div className="relative">
                  <div className="w-4 h-4 bg-error-500 rounded-full shadow-soft transition-all duration-200 group-hover:scale-110"></div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <XCircle className="w-3 h-3 text-white" />
                  </div>
                </div>
                <span className="text-sm font-medium text-neutral-700 group-hover:text-neutral-900 transition-colors">Error</span>
              </div>
            </div>
          </div>
        </Panel>
      </ReactFlow>

      {/* Node Details Modal */}
      {showDetails && selectedNode && (
        <WorkflowNodeDetails
          nodeId={selectedNode}
          nodeName={formatNodeName(selectedNode)}
          status={getNodeStatus(selectedNode, status, state)}
          state={state}
          onClose={() => setShowDetails(false)}
        />
      )}
    </div>
  );
};

// Helper functions
function getNodeStatus(nodeId: string, status: WorkflowStatus, state: TTDRState | null): 'pending' | 'running' | 'completed' | 'error' {
  if (status.status === 'error') return 'error';
  if (status.current_node === nodeId) return 'running';
  
  // Determine if node has been completed based on workflow progress
  const nodeOrder = WORKFLOW_NODES.findIndex(n => n.id === nodeId);
  const currentNodeOrder = WORKFLOW_NODES.findIndex(n => n.id === status.current_node);
  
  if (currentNodeOrder > nodeOrder) return 'completed';
  if (status.status === 'completed') return 'completed';
  
  return 'pending';
}

function getNodeColor(status: string): string {
  switch (status) {
    case 'running': return '#3b82f6';
    case 'completed': return '#10b981';
    case 'error': return '#ef4444';
    default: return '#9ca3af';
  }
}

function getEdgeColor(edge: any, status: WorkflowStatus, state: TTDRState | null): string {
  const sourceStatus = getNodeStatus(edge.source, status, state);
  const targetStatus = getNodeStatus(edge.target, status, state);
  
  if (sourceStatus === 'completed' && targetStatus === 'running') return '#3b82f6';
  if (sourceStatus === 'completed' && targetStatus === 'completed') return '#10b981';
  if (status.status === 'error') return '#ef4444';
  
  return '#9ca3af';
}

function isEdgeActive(edge: any, status: WorkflowStatus): boolean {
  return status.current_node === edge.target || 
         (status.current_node === edge.source && status.status === 'running');
}

function formatNodeName(nodeId: string): string {
  return nodeId
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'running':
      return <Play className="w-5 h-5 text-primary-500" />;
    case 'completed':
      return <CheckCircle className="w-5 h-5 text-success-500" />;
    case 'error':
      return <XCircle className="w-5 h-5 text-error-500" />;
    case 'idle':
      return <Pause className="w-5 h-5 text-neutral-500" />;
    default:
      return <Clock className="w-5 h-5 text-neutral-500" />;
  }
}

function getStatusDotColor(status: string): string {
  switch (status) {
    case 'running':
      return 'bg-primary-500';
    case 'completed':
      return 'bg-success-500';
    case 'error':
      return 'bg-error-500';
    case 'idle':
      return 'bg-neutral-400';
    default:
      return 'bg-neutral-400';
  }
}

function getActiveNodesCount(status: WorkflowStatus): number {
  return status.current_node ? 1 : 0;
}

function getEdgeLabelColor(edge: any, status: WorkflowStatus, state: TTDRState | null): string {
  const sourceStatus = getNodeStatus(edge.source, status, state);
  const targetStatus = getNodeStatus(edge.target, status, state);
  
  if (sourceStatus === 'completed' && targetStatus === 'running') return '#1d4ed8';
  if (sourceStatus === 'completed' && targetStatus === 'completed') return '#15803d';
  if (status.status === 'error') return '#b91c1c';
  
  return '#525252';
}