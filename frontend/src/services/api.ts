// API client for TTD-DR Framework

import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';
import { TTDRState, ResearchRequirements, ApiResponse, WorkflowStatus } from '../types';

export interface ApiError {
  message: string;
  code?: string;
  status?: number;
  details?: any;
}

class TTDRApiClient {
  private client: AxiosInstance;
  private retryAttempts = 3;
  private retryDelay = 1000;
  private token: string | null = null;

  constructor(baseURL: string = '/api') {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        if (this.token) {
          config.headers.Authorization = `Bearer ${this.token}`;
        }
        console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
        if (config.data) {
          console.log('Request payload:', config.data);
        }
        return config;
      },
      (error) => {
        console.error('Request error:', error);
        return Promise.reject(this.handleError(error));
      }
    );

    // Response interceptor with retry logic
    this.client.interceptors.response.use(
      (response: AxiosResponse) => {
        return response;
      },
      async (error: AxiosError) => {
        const originalRequest = error.config as any;
        
        // Retry logic for network errors and 5xx errors
        if (
          !originalRequest._retry &&
          originalRequest._retryCount < this.retryAttempts &&
          (error.code === 'NETWORK_ERROR' || 
           error.code === 'ECONNABORTED' ||
           (error.response?.status && error.response.status >= 500))
        ) {
          originalRequest._retry = true;
          originalRequest._retryCount = (originalRequest._retryCount || 0) + 1;
          
          // Wait before retrying
          await new Promise(resolve => 
            setTimeout(resolve, this.retryDelay * originalRequest._retryCount)
          );
          
          return this.client(originalRequest);
        }

        return Promise.reject(this.handleError(error));
      }
    );
  }

  private handleError(error: any): ApiError {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError;
      
      if (axiosError.response) {
        // Server responded with error status
        const status = axiosError.response.status;
        const data = axiosError.response.data as any;
        
        console.error('Server error response:', {
          status,
          data,
          url: axiosError.config?.url,
          method: axiosError.config?.method,
          payload: axiosError.config?.data
        });
        
        return {
          message: data?.message || data?.error || this.getStatusMessage(status),
          code: data?.code || `HTTP_${status}`,
          status,
          details: data?.details || data,
        };
      } else if (axiosError.request) {
        // Network error
        return {
          message: 'Network error - please check your connection',
          code: 'NETWORK_ERROR',
          details: axiosError.message,
        };
      } else {
        // Request setup error
        return {
          message: 'Request configuration error',
          code: 'REQUEST_ERROR',
          details: axiosError.message,
        };
      }
    }
    
    // Generic error
    return {
      message: error?.message || 'An unexpected error occurred',
      code: 'UNKNOWN_ERROR',
      details: error,
    };
  }

  private getStatusMessage(status: number): string {
    switch (status) {
      case 400:
        return 'Invalid request - please check your input';
      case 401:
        return 'Authentication required';
      case 403:
        return 'Access denied';
      case 404:
        return 'Resource not found';
      case 408:
        return 'Request timeout - please try again';
      case 429:
        return 'Too many requests - please wait and try again';
      case 500:
        return 'Server error - please try again later';
      case 502:
        return 'Service temporarily unavailable';
      case 503:
        return 'Service maintenance in progress';
      case 504:
        return 'Request timeout - please try again';
      default:
        return `Server error (${status})`;
    }
  }

  // Health check
  async healthCheck(): Promise<ApiResponse<{ status: string }>> {
    const response = await this.client.get('/health');
    return response.data;
  }

  // Authenticate and get token
  async login(username: string, password: string):Promise<void> {
    try {
      const params = new URLSearchParams();
      params.append('username', username);
      params.append('password', password);

      const response = await this.client.post('/api/v1/auth/login', params, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      });
      this.token = response.data.access_token;
      console.log('Login successful, token stored.');
    } catch (error) {
      console.error('Login failed:', error);
      throw this.handleError(error);
    }
  }

  // Start research workflow
  async startResearch(
    topic: string,
    requirements: ResearchRequirements
  ): Promise<ApiResponse<{ execution_id: string }>> {
    const payload = {
      topic,
      domain: requirements.domain,
      complexity_level: requirements.complexity_level,
      max_iterations: requirements.max_iterations,
      quality_threshold: requirements.quality_threshold,
      max_sources: requirements.max_sources,
      preferred_source_types: requirements.preferred_source_types,
    };
    console.log('Sending payload:', payload);
    const response = await this.client.post('/api/v1/research/initiate', payload);
    return response.data;
  }

  // Get workflow status
  async getWorkflowStatus(workflowId: string): Promise<ApiResponse<WorkflowStatus>> {
    const response = await this.client.get(`/api/v1/research/status/${workflowId}`);
    return response.data;
  }

  // Get current state
  async getCurrentState(workflowId: string): Promise<ApiResponse<TTDRState>> {
    const response = await this.client.get(`/api/v1/research/status/${workflowId}`);
    return response.data;
  }

  // Get final report
  async getFinalReport(workflowId: string): Promise<ApiResponse<{ report: string }>> {
    const response = await this.client.get(`/api/v1/research/result/${workflowId}`);
    return response.data;
  }

  // Cancel workflow
  async stopWorkflow(workflowId: string): Promise<ApiResponse<{ message: string }>> {
    const response = await this.client.post(`/api/v1/research/cancel/${workflowId}`);
    return response.data;
  }
}

// Create singleton instance
export const apiClient = new TTDRApiClient();

// Auto-login for development
if (process.env.NODE_ENV === 'development') {
  apiClient.login('dev_user', 'dev_password').catch(err => {
    console.error('Auto-login failed:', err);
  });
}

// WebSocket client for real-time updates
export class TTDRWebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  constructor(url: string = 'ws://localhost:8000/api/v1/research/ws') {
    this.url = url;
  }

  connect(workflowId: string, onMessage: (data: any) => void, onError?: (error: Event) => void): void {
    try {
      this.ws = new WebSocket(`${this.url}/${workflowId}`);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.attemptReconnect(workflowId, onMessage, onError);
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        if (onError) {
          onError(error);
        }
      };
    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
      if (onError) {
        onError(error as Event);
      }
    }
  }

  private attemptReconnect(workflowId: string, onMessage: (data: any) => void, onError?: (error: Event) => void): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
      
      setTimeout(() => {
        this.connect(workflowId, onMessage, onError);
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      console.error('Max reconnection attempts reached');
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  send(data: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.error('WebSocket is not connected');
    }
  }
}

export const wsClient = new TTDRWebSocketClient();