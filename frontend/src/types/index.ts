// Core data types for TTD-DR Framework Frontend

export enum GapType {
  CONTENT = "content",
  EVIDENCE = "evidence",
  CITATION = "citation",
  ANALYSIS = "analysis"
}

export enum Priority {
  LOW = "low",
  MEDIUM = "medium",
  HIGH = "high",
  CRITICAL = "critical"
}

export enum ComplexityLevel {
  BASIC = "basic",
  INTERMEDIATE = "intermediate",
  ADVANCED = "advanced",
  EXPERT = "expert"
}

export enum ResearchDomain {
  TECHNOLOGY = "technology",
  SCIENCE = "science",
  BUSINESS = "business",
  ACADEMIC = "academic",
  GENERAL = "general"
}

export interface Source {
  url: string;
  title: string;
  domain: string;
  credibility_score: number;
  last_accessed: string;
}

export interface Section {
  id: string;
  title: string;
  content: string;
  subsections: Section[];
  estimated_length: number;
}

export interface ResearchStructure {
  sections: Section[];
  relationships: any[];
  estimated_length: number;
  complexity_level: ComplexityLevel;
  domain: ResearchDomain;
}

export interface Draft {
  id: string;
  topic: string;
  structure: ResearchStructure;
  content: Record<string, string>;
  metadata: {
    created_at: string;
    updated_at: string;
    author: string;
    version: string;
    word_count: number;
  };
  quality_score: number;
  iteration: number;
}

export interface InformationGap {
  id: string;
  section_id: string;
  gap_type: GapType;
  description: string;
  priority: Priority;
  search_queries: any[];
}

export interface QualityMetrics {
  completeness: number;
  coherence: number;
  accuracy: number;
  citation_quality: number;
  overall_score: number;
}

export interface ResearchRequirements {
  domain: ResearchDomain;
  complexity_level: ComplexityLevel;
  max_iterations: number;
  quality_threshold: number;
  max_sources: number;
  preferred_source_types: string[];
}

export interface TTDRState {
  topic: string;
  requirements: ResearchRequirements;
  current_draft: Draft | null;
  information_gaps: InformationGap[];
  retrieved_info: any[];
  iteration_count: number;
  quality_metrics: QualityMetrics | null;
  evolution_history: any[];
  final_report: string | null;
  error_log: string[];
}

// API Response types
export interface ApiResponse<T> {
  data: T;
  message?: string;
  error?: string;
}

export interface WorkflowStatus {
  status: 'idle' | 'running' | 'completed' | 'error';
  current_node: string | null;
  progress: number;
  message: string;
}