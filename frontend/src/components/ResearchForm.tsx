import React from 'react';
import { useForm } from 'react-hook-form';
import { ResearchRequirements, ResearchDomain, ComplexityLevel } from '../types';
import { LoadingButton } from './LoadingSystem';
import { useNotifications } from './NotificationSystem';

interface ResearchFormProps {
  onSubmit: (topic: string, requirements: ResearchRequirements) => void;
  isLoading: boolean;
}

interface FormData {
  topic: string;
  domain: ResearchDomain;
  complexity_level: ComplexityLevel;
  max_iterations: number;
  quality_threshold: number;
  max_sources: number;
  preferred_source_types: string[];
}

export const ResearchForm: React.FC<ResearchFormProps> = ({ onSubmit, isLoading }) => {
  const { showError, showWarning } = useNotifications();
  
  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
    setValue
  } = useForm<FormData>({
    defaultValues: {
      topic: 'AI在教育领域的应用研究',
      domain: ResearchDomain.GENERAL,
      complexity_level: ComplexityLevel.INTERMEDIATE,
      max_iterations: 5,
      quality_threshold: 0.8,
      max_sources: 20,
      preferred_source_types: ['academic', 'news', 'official']
    }
  });

  const selectedSourceTypes = watch('preferred_source_types');

  const handleSourceTypeChange = (sourceType: string, checked: boolean) => {
    const current = selectedSourceTypes || [];
    if (checked) {
      setValue('preferred_source_types', [...current, sourceType]);
    } else {
      setValue('preferred_source_types', current.filter(type => type !== sourceType));
    }
  };

  const onFormSubmit = (data: FormData) => {
    // Validate preferred source types
    if (!data.preferred_source_types || data.preferred_source_types.length === 0) {
      showWarning('Validation Error', 'Please select at least one preferred source type.');
      return;
    }

    // Validate topic length
    if (!data.topic || data.topic.trim().length < 10) {
      showError('Validation Error', 'Research topic must be at least 10 characters long.');
      return;
    }

    const requirements: ResearchRequirements = {
      domain: data.domain,
      complexity_level: data.complexity_level,
      max_iterations: data.max_iterations,
      quality_threshold: data.quality_threshold,
      max_sources: data.max_sources,
      preferred_source_types: data.preferred_source_types
    };
    
    onSubmit(data.topic, requirements);
  };

  const sourceTypeOptions = [
    { value: 'academic', label: 'Academic Papers' },
    { value: 'news', label: 'News Articles' },
    { value: 'official', label: 'Official Documents' },
    { value: 'blog', label: 'Blog Posts' },
    { value: 'wiki', label: 'Wikipedia' },
    { value: 'forum', label: 'Forums & Discussions' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-gray-800 font-sans">
      <div className="max-w-6xl mx-auto px-4 py-12 md:py-20">
        <header className="text-center mb-12 md:mb-20">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-3xl mb-8 shadow-lg transform hover:scale-110 transition-transform duration-300">
            <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <h1 className="text-5xl md:text-6xl lg:text-7xl font-extrabold bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-400 bg-clip-text text-transparent mb-6 tracking-tight">
            AI研究助手
          </h1>
          <p className="text-xl md:text-2xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
            将您的研究问题转化为全面、有据可查的报告，
            通过智能AI分析和迭代优化实现。
          </p>
        </header>
        
        <main className="max-w-4xl mx-auto">
          <div className="bg-gray-800/80 backdrop-blur-xl rounded-3xl shadow-2xl border border-gray-700/20 p-8 md:p-12 transform hover:-translate-y-2 transition-transform duration-300">
            <div className="text-center mb-10">
              <h2 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-gray-100 to-gray-300 bg-clip-text text-transparent mb-3">
                开始您的研究
              </h2>
              <p className="text-lg text-gray-400">
                输入您的研究主题并配置高级参数。
              </p>
            </div>
            
            <form onSubmit={handleSubmit(onFormSubmit)} className="space-y-6 md:space-y-8">
              {/* Research Topic */}
              <div className="space-y-3">
                <label htmlFor="topic" className="block text-lg font-semibold text-gray-100">
                  您想研究什么？ <span className="text-red-400">*</span>
                </label>
                <div className="relative group">
                  <textarea
                    id="topic"
                    data-testid="research-topic"
                    {...register('topic', { 
                      required: 'Research topic is required',
                      minLength: { value: 10, message: 'Topic must be at least 10 characters long' }
                    })}
                    className={`w-full px-5 py-4 text-lg rounded-2xl border-2 transition-all duration-300 focus:outline-none focus:ring-4 ${
                      errors.topic 
                        ? 'border-red-400 bg-red-900/50 text-red-100 focus:ring-red-200 focus:border-red-500' 
                        : 'border-gray-600 bg-gray-700 text-gray-100 focus:ring-blue-200 focus:border-blue-500 hover:border-gray-500 group-hover:border-blue-400'
                    }`}
                    rows={4}
                    placeholder="例如：'人工智能对现代医疗保健和患者结果的影响'"
                    disabled={isLoading}
                    aria-required="true"
                    autoFocus
                  />
                  <div className="absolute top-4 right-4 text-slate-400 group-focus-within:text-blue-500 transition-colors duration-300">
                    ✨
                  </div>
                </div>
                {errors.topic && (
                  <p className="text-sm text-red-600 font-medium flex items-center"><svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" /></svg>{errors.topic.message}</p>
                )}
                <p className="text-sm text-gray-400 flex items-start bg-gray-700/50 p-3 rounded-lg">
                  <span className="mr-2 text-lg">💡</span>
                  <span><strong>专业提示：</strong>要具体！不要只说"医疗AI"，试试"AI诊断工具如何降低农村医院的误诊率"</span>
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Research Domain */}
                <div className="space-y-2">
                  <label htmlFor="domain" className="block text-base font-semibold text-gray-200">
                    研究领域
                  </label>
                  <select
                    id="domain"
                    data-testid="research-domain"
                    {...register('domain')}
                    className="w-full px-4 py-3 rounded-xl border-2 border-gray-600 bg-gray-700 text-gray-100 transition-all duration-300 focus:outline-none focus:ring-4 focus:ring-blue-200 focus:border-blue-500 hover:border-gray-500 text-base"
                    disabled={isLoading}
                  >
                    <option value={ResearchDomain.GENERAL}>🌍 通用</option>
                    <option value={ResearchDomain.TECHNOLOGY}>💻 技术</option>
                    <option value={ResearchDomain.SCIENCE}>🔬 科学</option>
                    <option value={ResearchDomain.BUSINESS}>💼 商业</option>
                    <option value={ResearchDomain.ACADEMIC}>🎓 学术</option>
                  </select>
                </div>

                {/* Complexity Level */}
                <div className="space-y-2">
                  <label htmlFor="complexity_level" className="block text-base font-semibold text-gray-200">
                    复杂程度
                  </label>
                  <select
                    id="complexity_level"
                    data-testid="complexity-level"
                    {...register('complexity_level')}
                    className="w-full px-4 py-3 rounded-xl border-2 border-slate-200 bg-white text-slate-900 transition-all duration-300 focus:outline-none focus:ring-4 focus:ring-blue-200 focus:border-blue-500 hover:border-slate-300 text-base"
                    disabled={isLoading}
                  >
                    <option value={ComplexityLevel.BASIC}>🟢 基础</option>
                    <option value={ComplexityLevel.INTERMEDIATE}>🟡 中级</option>
                    <option value={ComplexityLevel.ADVANCED}>🟠 高级</option>
                    <option value={ComplexityLevel.EXPERT}>🔴 专家</option>
                  </select>
                </div>
              </div>

              {/* Advanced Configuration */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <label htmlFor="max_iterations" className="block text-sm font-semibold text-gray-200">
                    最大迭代次数
                  </label>
                  <input
                    type="number"
                    id="max_iterations"
                    {...register('max_iterations', { 
                      min: { value: 1, message: 'Must be at least 1' },
                      max: { value: 20, message: 'Cannot exceed 20' }
                    })}
                    className={`w-full px-4 py-3 rounded-xl border-2 transition-all duration-200 focus:outline-none focus:ring-4 ${
                      errors.max_iterations 
                        ? 'border-red-400 bg-red-900/50 text-red-100 focus:ring-red-200' 
                        : 'border-gray-600 bg-gray-700 text-gray-100 focus:ring-blue-200 focus:border-blue-500 hover:border-gray-500'
                    }`}
                    disabled={isLoading}
                  />
                  {errors.max_iterations && (
                    <p className="text-sm text-red-600 font-medium">{errors.max_iterations.message}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <label htmlFor="max_sources" className="block text-sm font-semibold text-gray-200">
                    Max Sources
                  </label>
                  <input
                    type="number"
                    id="max_sources"
                    {...register('max_sources', { 
                      min: { value: 5, message: 'Must be at least 5' },
                      max: { value: 100, message: 'Cannot exceed 100' }
                    })}
                    className={`w-full px-4 py-3 rounded-xl border-2 transition-all duration-200 focus:outline-none focus:ring-4 ${
                      errors.max_sources 
                        ? 'border-red-400 bg-red-50 text-red-900 focus:ring-red-200' 
                        : 'border-slate-200 bg-white text-slate-900 focus:ring-blue-200 focus:border-blue-500 hover:border-slate-300'
                    }`}
                    disabled={isLoading}
                  />
                  {errors.max_sources && (
                    <p className="text-sm text-red-600 font-medium">{errors.max_sources.message}</p>
                  )}
                </div>
              </div>

              {/* Quality Threshold */}
              <div className="space-y-3">
                <label htmlFor="quality_threshold" className="block text-sm font-semibold text-gray-200">
                    质量阈值: <span className="text-blue-400 font-bold">{watch('quality_threshold')}</span>
                  </label>
                <div className="relative">
                  <input
                    type="range"
                    id="quality_threshold"
                    {...register('quality_threshold')}
                    min="0.5"
                    max="1.0"
                    step="0.05"
                    className="w-full h-3 rounded-full appearance-none cursor-pointer bg-gradient-to-r from-blue-200 via-purple-200 to-indigo-200 focus:outline-none focus:ring-4 focus:ring-blue-200"
                    disabled={isLoading}
                  />
                  <div className="absolute -top-1 left-0 right-0 flex justify-between text-xs text-gray-400">
                    <span className="bg-white px-2 rounded">🟢 标准</span>
                    <span className="bg-white px-2 rounded">🟡 高</span>
                    <span className="bg-white px-2 rounded">🔴 优质</span>
                  </div>
                </div>
              </div>

              {/* Preferred Source Types */}
              <div className="space-y-3">
                <label className="block text-sm font-semibold text-gray-200">
                    首选来源类型
                  </label>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {sourceTypeOptions.map((option) => (
                    <label key={option.value} className="flex items-center space-x-3 p-4 rounded-2xl border-2 border-gray-600 bg-gray-700 hover:border-blue-500 hover:bg-gray-600 transition-all duration-200 cursor-pointer group">
                      <input
                        type="checkbox"
                        checked={selectedSourceTypes?.includes(option.value) || false}
                        onChange={(e) => handleSourceTypeChange(option.value, e.target.checked)}
                        className="w-5 h-5 text-blue-600 bg-white border-2 border-slate-300 rounded focus:ring-blue-500 focus:ring-2 transition-all duration-200"
                        disabled={isLoading}
                      />
                      <span className="text-sm font-medium text-gray-200 group-hover:text-blue-300">{option.label}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Submit Button */}
              <div className="pt-6">
                <LoadingButton
                  type="submit"
                  loading={isLoading}
                  loadingText="🔄 Starting Research..."
                  className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white text-lg font-bold py-4 px-6 rounded-2xl shadow-lg hover:shadow-xl transform hover:scale-[1.02] transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <span className="flex items-center justify-center space-x-2">
                    <span>🚀</span>
                    <span>立即开始研究</span>
                  </span>
                </LoadingButton>
              </div>
            </form>
          </div>
        </main>
      </div>
    </div>
  );
};