"""
Tests for domain evaluation system.
This module tests the evaluation and metrics for domain adaptation effectiveness.
"""

import pytest
from unittest.mock import Mock, patch
import json
from datetime import datetime

from backend.models.core import ResearchDomain, QualityMetrics
from backend.services.domain_adapter import DomainAdapter
from backend.services.domain_evaluation import (
    DomainEvaluator, DomainAdaptationMetrics, DomainBenchmark,
    create_domain_evaluation_report
)
from backend.services.kimi_k2_client import KimiK2Client


class TestDomainAdaptationMetrics:
    """Test cases for DomainAdaptationMetrics model"""
    
    def test_metrics_creation(self):
        """Test creation of domain adaptation metrics"""
        metrics = DomainAdaptationMetrics(
            domain=ResearchDomain.TECHNOLOGY,
            detection_accuracy=0.9,
            adaptation_effectiveness=0.85,
            terminology_consistency=0.8,
            format_compliance=0.9,
            quality_improvement=0.1,
            processing_time=2.5
        )
        
        assert metrics.domain == ResearchDomain.TECHNOLOGY
        assert metrics.detection_accuracy == 0.9
        assert metrics.overall_score() > 0.8
    
    def test_overall_score_calculation(self):
        """Test overall score calculation"""
        metrics = DomainAdaptationMetrics(
            domain=ResearchDomain.SCIENCE,
            detection_accuracy=0.8,
            adaptation_effectiveness=0.7,
            terminology_consistency=0.9,
            format_compliance=0.8,
            quality_improvement=0.2,
            processing_time=1.0
        )
        
        expected_score = (0.8 + 0.7 + 0.9 + 0.8 + 0.2) / 5
        assert abs(metrics.overall_score() - expected_score) < 0.01
    
    def test_negative_quality_improvement_handling(self):
        """Test handling of negative quality improvement"""
        metrics = DomainAdaptationMetrics(
            domain=ResearchDomain.BUSINESS,
            detection_accuracy=0.8,
            adaptation_effectiveness=0.7,
            terminology_consistency=0.9,
            format_compliance=0.8,
            quality_improvement=-0.1,  # Negative improvement
            processing_time=1.0
        )
        
        # Should not penalize overall score for negative improvement
        overall_score = metrics.overall_score()
        assert overall_score > 0.0
        assert overall_score == (0.8 + 0.7 + 0.9 + 0.8 + 0.0) / 5


class TestDomainEvaluator:
    """Test cases for DomainEvaluator class"""
    
    @pytest.fixture
    def mock_kimi_client(self):
        """Mock Kimi K2 client for testing"""
        return Mock(spec=KimiK2Client)
    
    @pytest.fixture
    def domain_adapter(self, mock_kimi_client):
        """Create DomainAdapter instance with mocked client"""
        return DomainAdapter(mock_kimi_client)
    
    @pytest.fixture
    def domain_evaluator(self, domain_adapter):
        """Create DomainEvaluator instance"""
        return DomainEvaluator(domain_adapter)
    
    def test_detection_accuracy_evaluation(self, domain_evaluator):
        """Test domain detection accuracy evaluation"""
        test_cases = [
            ("Machine Learning Algorithms", ResearchDomain.TECHNOLOGY),
            ("Clinical Trial Results", ResearchDomain.SCIENCE),
            ("Market Analysis Report", ResearchDomain.BUSINESS),
            ("Academic Research Paper", ResearchDomain.ACADEMIC),
            ("General Information Overview", ResearchDomain.GENERAL)
        ]
        
        accuracy = domain_evaluator.evaluate_detection_accuracy(test_cases)
        
        assert 0.0 <= accuracy <= 1.0
        assert isinstance(accuracy, float)
    
    def test_confidence_calibration_evaluation(self, domain_evaluator):
        """Test confidence calibration evaluation"""
        test_cases = [
            ("Machine Learning", ResearchDomain.TECHNOLOGY, 0.9),
            ("Clinical Study", ResearchDomain.SCIENCE, 0.8),
            ("Business Analysis", ResearchDomain.BUSINESS, 0.7)
        ]
        
        calibration_metrics = domain_evaluator.evaluate_confidence_calibration(test_cases)
        
        assert "calibration_error" in calibration_metrics
        assert "confidence_accuracy_correlation" in calibration_metrics
        assert "mean_confidence" in calibration_metrics
        assert "mean_accuracy" in calibration_metrics
        
        assert 0.0 <= calibration_metrics["calibration_error"] <= 1.0
        assert -1.0 <= calibration_metrics["confidence_accuracy_correlation"] <= 1.0
    
    def test_adaptation_effectiveness_evaluation(self, domain_evaluator):
        """Test adaptation effectiveness evaluation"""
        from backend.models.core import ResearchRequirements, ComplexityLevel
        
        original_requirements = ResearchRequirements(
            domain=ResearchDomain.GENERAL,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            quality_threshold=0.8,
            preferred_source_types=["general", "news"]
        )
        
        adapted_requirements = ResearchRequirements(
            domain=ResearchDomain.TECHNOLOGY,
            complexity_level=ComplexityLevel.INTERMEDIATE,
            quality_threshold=0.85,
            preferred_source_types=["tech_blogs", "documentation", "github"]
        )
        
        effectiveness = domain_evaluator.evaluate_adaptation_effectiveness(
            "Machine Learning Topic",
            original_requirements,
            adapted_requirements
        )
        
        assert 0.0 <= effectiveness <= 1.0
        assert effectiveness > 0.5  # Should be effective adaptation
    
    def test_terminology_consistency_evaluation(self, domain_evaluator):
        """Test terminology consistency evaluation"""
        # Technology domain content with consistent terminology
        tech_content = "The API framework uses AI (Artificial Intelligence) for ML processing. The SDK provides comprehensive API documentation."
        
        consistency = domain_evaluator.evaluate_terminology_consistency(
            tech_content, ResearchDomain.TECHNOLOGY
        )
        
        assert 0.0 <= consistency <= 1.0
        assert isinstance(consistency, float)
    
    def test_format_compliance_evaluation(self, domain_evaluator):
        """Test format compliance evaluation"""
        # Technology domain content with proper formatting
        tech_content = "The `API` framework provides robust functionality. The system architecture includes multiple components."
        
        compliance = domain_evaluator.evaluate_format_compliance(
            tech_content, ResearchDomain.TECHNOLOGY
        )
        
        assert 0.0 <= compliance <= 1.0
        assert isinstance(compliance, float)
    
    def test_comprehensive_evaluation(self, domain_evaluator):
        """Test comprehensive evaluation"""
        test_cases = [
            ("Machine Learning", ResearchDomain.TECHNOLOGY),
            ("Clinical Research", ResearchDomain.SCIENCE)
        ]
        
        results = domain_evaluator.run_comprehensive_evaluation(test_cases)
        
        assert "evaluation_timestamp" in results
        assert "total_test_cases" in results
        assert "metrics" in results
        assert "detection_accuracy" in results["metrics"]
        assert "confidence_calibration" in results["metrics"]
        assert "domain_specific" in results["metrics"]
        assert "overall_score" in results["metrics"]
        
        assert results["total_test_cases"] == len(test_cases)
        assert 0.0 <= results["metrics"]["overall_score"] <= 1.0
    
    def test_empty_test_cases_handling(self, domain_evaluator):
        """Test handling of empty test cases"""
        accuracy = domain_evaluator.evaluate_detection_accuracy([])
        assert accuracy == 0.0
        
        calibration = domain_evaluator.evaluate_confidence_calibration([])
        assert calibration["calibration_error"] == 1.0
        assert calibration["confidence_accuracy_correlation"] == 0.0
    
    def test_benchmark_initialization(self, domain_evaluator):
        """Test benchmark initialization"""
        benchmarks = domain_evaluator.benchmarks
        
        assert len(benchmarks) > 0
        assert all(isinstance(b, DomainBenchmark) for b in benchmarks)
        
        # Check that all domains are represented
        benchmark_domains = {b.expected_domain for b in benchmarks}
        assert len(benchmark_domains) == len(ResearchDomain)


class TestDomainBenchmark:
    """Test cases for DomainBenchmark model"""
    
    def test_benchmark_creation(self):
        """Test creation of domain benchmark"""
        benchmark = DomainBenchmark(
            topic="Test Topic",
            expected_domain=ResearchDomain.TECHNOLOGY,
            expected_confidence=0.9,
            expected_keywords=["test", "topic"],
            quality_baseline=QualityMetrics(
                completeness=0.8, coherence=0.8, accuracy=0.8, citation_quality=0.8
            )
        )
        
        assert benchmark.topic == "Test Topic"
        assert benchmark.expected_domain == ResearchDomain.TECHNOLOGY
        assert benchmark.expected_confidence == 0.9
        assert len(benchmark.expected_keywords) == 2


class TestDomainEvaluationReport:
    """Test cases for domain evaluation report generation"""
    
    @pytest.fixture
    def mock_kimi_client(self):
        """Mock Kimi K2 client for testing"""
        return Mock(spec=KimiK2Client)
    
    @pytest.fixture
    def domain_adapter(self, mock_kimi_client):
        """Create DomainAdapter instance with mocked client"""
        return DomainAdapter(mock_kimi_client)
    
    def test_evaluation_report_creation(self, domain_adapter):
        """Test creation of evaluation report"""
        report = create_domain_evaluation_report(domain_adapter)
        
        assert "evaluation_timestamp" in report
        assert "total_test_cases" in report
        assert "metrics" in report
        assert "system_info" in report
        assert "recommendations" in report
        
        # Check system info
        system_info = report["system_info"]
        assert "domains_supported" in system_info
        assert "total_strategies" in system_info
        assert "total_terminology_handlers" in system_info
        assert "evaluation_version" in system_info
        
        assert len(system_info["domains_supported"]) == len(ResearchDomain)
    
    def test_recommendations_based_on_score(self, domain_adapter):
        """Test that recommendations are appropriate for different scores"""
        report = create_domain_evaluation_report(domain_adapter)
        
        overall_score = report["metrics"]["overall_score"]
        recommendations = report["recommendations"]
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Recommendations should be strings
        assert all(isinstance(rec, str) for rec in recommendations)
    
    @patch('builtins.open', create=True)
    def test_report_file_output(self, mock_open, domain_adapter):
        """Test saving report to file"""
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        report = create_domain_evaluation_report(domain_adapter, "test_report.json")
        
        # Verify file was opened for writing
        mock_open.assert_called_once_with("test_report.json", 'w', encoding='utf-8')
        
        # Verify JSON was written to file
        mock_file.write.assert_called()


class TestDomainEvaluationIntegration:
    """Integration tests for domain evaluation system"""
    
    @pytest.fixture
    def mock_kimi_client(self):
        """Mock Kimi K2 client for integration testing"""
        return Mock(spec=KimiK2Client)
    
    @pytest.fixture
    def domain_adapter(self, mock_kimi_client):
        """Create DomainAdapter for integration testing"""
        return DomainAdapter(mock_kimi_client)
    
    def test_end_to_end_evaluation(self, domain_adapter):
        """Test complete evaluation workflow"""
        evaluator = DomainEvaluator(domain_adapter)
        
        # Test with a variety of topics
        test_cases = [
            ("Python Programming Tutorial", ResearchDomain.TECHNOLOGY),
            ("COVID-19 Vaccine Efficacy Study", ResearchDomain.SCIENCE),
            ("Quarterly Financial Report Analysis", ResearchDomain.BUSINESS),
            ("Cognitive Psychology Research Methods", ResearchDomain.ACADEMIC),
            ("Climate Change Overview", ResearchDomain.GENERAL)
        ]
        
        # Run comprehensive evaluation
        results = evaluator.run_comprehensive_evaluation(test_cases)
        
        # Verify results structure
        assert isinstance(results, dict)
        assert "metrics" in results
        assert "evaluation_timestamp" in results
        
        # Verify metrics are reasonable
        metrics = results["metrics"]
        assert 0.0 <= metrics["detection_accuracy"] <= 1.0
        assert 0.0 <= metrics["overall_score"] <= 1.0
        
        # Verify domain-specific results
        domain_specific = metrics["domain_specific"]
        assert isinstance(domain_specific, dict)
        
        for domain_name, domain_metrics in domain_specific.items():
            assert "accuracy" in domain_metrics
            assert "test_cases" in domain_metrics
            assert 0.0 <= domain_metrics["accuracy"] <= 1.0
            assert domain_metrics["test_cases"] > 0
    
    def test_evaluation_with_different_domains(self, domain_adapter):
        """Test evaluation across different research domains"""
        evaluator = DomainEvaluator(domain_adapter)
        
        # Test each domain individually
        for domain in ResearchDomain:
            test_cases = [(f"Sample {domain.value} topic", domain)]
            
            accuracy = evaluator.evaluate_detection_accuracy(test_cases)
            assert 0.0 <= accuracy <= 1.0
            
            # Test terminology consistency for domain-specific content
            sample_content = f"This is a {domain.value} research topic with relevant content."
            consistency = evaluator.evaluate_terminology_consistency(sample_content, domain)
            assert 0.0 <= consistency <= 1.0
            
            # Test format compliance
            compliance = evaluator.evaluate_format_compliance(sample_content, domain)
            assert 0.0 <= compliance <= 1.0
    
    def test_metrics_aggregation(self, domain_adapter):
        """Test aggregation of evaluation metrics"""
        evaluator = DomainEvaluator(domain_adapter)
        
        # Create metrics for different domains
        all_metrics = []
        for domain in ResearchDomain:
            metrics = DomainAdaptationMetrics(
                domain=domain,
                detection_accuracy=0.8,
                adaptation_effectiveness=0.7,
                terminology_consistency=0.9,
                format_compliance=0.8,
                quality_improvement=0.1,
                processing_time=1.0
            )
            all_metrics.append(metrics)
        
        # Verify all metrics are valid
        for metrics in all_metrics:
            assert 0.0 <= metrics.overall_score() <= 1.0
            assert metrics.domain in ResearchDomain
        
        # Calculate average performance across domains
        avg_score = sum(m.overall_score() for m in all_metrics) / len(all_metrics)
        assert 0.0 <= avg_score <= 1.0