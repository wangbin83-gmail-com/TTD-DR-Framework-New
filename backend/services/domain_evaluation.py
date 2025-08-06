"""
Domain adaptation evaluation and metrics system.
This module provides tools for evaluating the effectiveness of domain adaptation.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
import statistics
import json

from models.core import ResearchDomain, Draft, QualityMetrics
from services.domain_adapter import DomainAdapter, DomainDetectionResult


class DomainAdaptationMetrics(BaseModel):
    """Comprehensive metrics for domain adaptation evaluation"""
    domain: ResearchDomain
    detection_accuracy: float = Field(ge=0.0, le=1.0)
    adaptation_effectiveness: float = Field(ge=0.0, le=1.0)
    terminology_consistency: float = Field(ge=0.0, le=1.0)
    format_compliance: float = Field(ge=0.0, le=1.0)
    quality_improvement: float = Field(ge=-1.0, le=1.0)  # Can be negative
    processing_time: float = Field(ge=0.0)  # seconds
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def overall_score(self) -> float:
        """Calculate overall adaptation score"""
        metrics = [
            self.detection_accuracy,
            self.adaptation_effectiveness,
            self.terminology_consistency,
            self.format_compliance,
            max(0.0, self.quality_improvement)  # Don't penalize negative improvement
        ]
        return sum(metrics) / len(metrics)


class DomainBenchmark(BaseModel):
    """Benchmark data for domain adaptation evaluation"""
    topic: str
    expected_domain: ResearchDomain
    expected_confidence: float
    expected_keywords: List[str]
    quality_baseline: QualityMetrics
    adaptation_requirements: Dict[str, Any] = {}


class DomainEvaluator:
    """Evaluator for domain adaptation system performance"""
    
    def __init__(self, domain_adapter: DomainAdapter):
        self.domain_adapter = domain_adapter
        self.benchmarks = self._initialize_benchmarks()
    
    def evaluate_detection_accuracy(
        self, 
        test_cases: List[Tuple[str, ResearchDomain]]
    ) -> float:
        """
        Evaluate domain detection accuracy against known test cases.
        
        Args:
            test_cases: List of (topic, expected_domain) tuples
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if not test_cases:
            return 0.0
        
        correct_predictions = 0
        total_predictions = len(test_cases)
        
        for topic, expected_domain in test_cases:
            try:
                result = self.domain_adapter.detect_domain(topic)
                if result.primary_domain == expected_domain:
                    correct_predictions += 1
            except Exception as e:
                # Count failures as incorrect predictions
                continue
        
        return correct_predictions / total_predictions
    
    def evaluate_confidence_calibration(
        self, 
        test_cases: List[Tuple[str, ResearchDomain, float]]
    ) -> Dict[str, float]:
        """
        Evaluate how well confidence scores match actual accuracy.
        
        Args:
            test_cases: List of (topic, expected_domain, expected_confidence) tuples
            
        Returns:
            Dictionary with calibration metrics
        """
        predictions = []
        confidences = []
        accuracies = []
        
        for topic, expected_domain, expected_confidence in test_cases:
            try:
                result = self.domain_adapter.detect_domain(topic)
                predictions.append(result)
                confidences.append(result.confidence)
                accuracies.append(1.0 if result.primary_domain == expected_domain else 0.0)
            except Exception:
                continue
        
        if not predictions:
            return {"calibration_error": 1.0, "confidence_accuracy_correlation": 0.0}
        
        # Calculate calibration error (difference between confidence and accuracy)
        calibration_errors = [abs(conf - acc) for conf, acc in zip(confidences, accuracies)]
        mean_calibration_error = statistics.mean(calibration_errors)
        
        # Calculate correlation between confidence and accuracy
        if len(confidences) > 1:
            try:
                # Simple correlation calculation for Python 3.9 compatibility
                mean_conf = statistics.mean(confidences)
                mean_acc = statistics.mean(accuracies)
                
                numerator = sum((c - mean_conf) * (a - mean_acc) for c, a in zip(confidences, accuracies))
                denom_conf = sum((c - mean_conf) ** 2 for c in confidences)
                denom_acc = sum((a - mean_acc) ** 2 for a in accuracies)
                
                if denom_conf > 0 and denom_acc > 0:
                    correlation = numerator / (denom_conf * denom_acc) ** 0.5
                else:
                    correlation = 0.0
            except (ZeroDivisionError, ValueError):
                correlation = 0.0
        else:
            correlation = 0.0
        
        return {
            "calibration_error": mean_calibration_error,
            "confidence_accuracy_correlation": correlation,
            "mean_confidence": statistics.mean(confidences),
            "mean_accuracy": statistics.mean(accuracies)
        }
    
    def evaluate_adaptation_effectiveness(
        self, 
        topic: str, 
        original_requirements: Any,
        adapted_requirements: Any
    ) -> float:
        """
        Evaluate how effectively requirements were adapted for the domain.
        
        Args:
            topic: Research topic
            original_requirements: Original research requirements
            adapted_requirements: Domain-adapted requirements
            
        Returns:
            Effectiveness score (0.0 to 1.0)
        """
        effectiveness_score = 0.0
        total_criteria = 0
        
        # Check if domain was properly set
        if hasattr(adapted_requirements, 'domain') and adapted_requirements.domain != ResearchDomain.GENERAL:
            effectiveness_score += 1.0
        total_criteria += 1
        
        # Check if quality threshold was adjusted appropriately
        if (hasattr(original_requirements, 'quality_threshold') and 
            hasattr(adapted_requirements, 'quality_threshold')):
            if adapted_requirements.quality_threshold >= original_requirements.quality_threshold:
                effectiveness_score += 1.0
        total_criteria += 1
        
        # Check if source types were specialized
        if (hasattr(original_requirements, 'preferred_source_types') and 
            hasattr(adapted_requirements, 'preferred_source_types')):
            original_sources = set(original_requirements.preferred_source_types)
            adapted_sources = set(adapted_requirements.preferred_source_types)
            if adapted_sources != original_sources and len(adapted_sources) > 0:
                effectiveness_score += 1.0
        total_criteria += 1
        
        return effectiveness_score / total_criteria if total_criteria > 0 else 0.0
    
    def evaluate_terminology_consistency(
        self, 
        content: str, 
        domain: ResearchDomain
    ) -> float:
        """
        Evaluate consistency of domain-specific terminology usage.
        
        Args:
            content: Text content to evaluate
            domain: Research domain
            
        Returns:
            Consistency score (0.0 to 1.0)
        """
        if not content or domain not in self.domain_adapter.terminology_handlers:
            return 0.0
        
        terminology_handler = self.domain_adapter.terminology_handlers[domain]
        consistency_score = 0.0
        total_checks = 0
        
        # Check abbreviation consistency
        for abbrev, full_form in terminology_handler.abbreviations.items():
            if abbrev in content:
                total_checks += 1
                # Check if abbreviation is properly defined or used consistently
                if full_form in content or content.count(abbrev) > 1:
                    consistency_score += 1.0
        
        # Check terminology mapping consistency
        for general_term, domain_term in terminology_handler.terminology_map.items():
            if general_term in content.lower():
                total_checks += 1
                # Prefer domain-specific term
                if domain_term in content.lower():
                    consistency_score += 1.0
                else:
                    consistency_score += 0.5  # Partial credit for using general term
        
        return consistency_score / total_checks if total_checks > 0 else 1.0
    
    def evaluate_format_compliance(
        self, 
        content: str, 
        domain: ResearchDomain
    ) -> float:
        """
        Evaluate compliance with domain-specific formatting rules.
        
        Args:
            content: Text content to evaluate
            domain: Research domain
            
        Returns:
            Compliance score (0.0 to 1.0)
        """
        if not content:
            return 0.0
        
        compliance_score = 0.0
        total_checks = 0
        
        # Domain-specific format checks
        if domain == ResearchDomain.TECHNOLOGY:
            # Check for code formatting
            total_checks += 1
            if '`' in content or 'API' in content or 'framework' in content:
                compliance_score += 1.0
        
        elif domain == ResearchDomain.SCIENCE:
            # Check for statistical notation
            total_checks += 1
            if '*p*' in content or 'p =' in content or '*n*' in content:
                compliance_score += 1.0
        
        elif domain == ResearchDomain.BUSINESS:
            # Check for business formatting
            total_checks += 1
            if '$' in content or '%' in content or 'ROI' in content:
                compliance_score += 1.0
        
        elif domain == ResearchDomain.ACADEMIC:
            # Check for academic formatting
            total_checks += 1
            if 'et al.' in content or 'ibid.' in content or '(' in content:
                compliance_score += 1.0
        
        # General formatting checks
        total_checks += 1
        if len(content.split('.')) > 2:  # Has multiple sentences
            compliance_score += 1.0
        
        return compliance_score / total_checks if total_checks > 0 else 0.0
    
    def run_comprehensive_evaluation(
        self, 
        test_cases: Optional[List[Tuple[str, ResearchDomain]]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of domain adaptation system.
        
        Args:
            test_cases: Optional test cases, uses benchmarks if not provided
            
        Returns:
            Comprehensive evaluation results
        """
        if test_cases is None:
            test_cases = [(b.topic, b.expected_domain) for b in self.benchmarks]
        
        results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_test_cases": len(test_cases),
            "metrics": {}
        }
        
        # Detection accuracy
        detection_accuracy = self.evaluate_detection_accuracy(test_cases)
        results["metrics"]["detection_accuracy"] = detection_accuracy
        
        # Confidence calibration
        confidence_test_cases = [
            (topic, domain, 0.8)  # Assume expected confidence of 0.8
            for topic, domain in test_cases
        ]
        calibration_metrics = self.evaluate_confidence_calibration(confidence_test_cases)
        results["metrics"]["confidence_calibration"] = calibration_metrics
        
        # Domain-specific evaluations
        domain_results = {}
        for domain in ResearchDomain:
            domain_test_cases = [(topic, d) for topic, d in test_cases if d == domain]
            if domain_test_cases:
                domain_accuracy = self.evaluate_detection_accuracy(domain_test_cases)
                domain_results[domain.value] = {
                    "accuracy": domain_accuracy,
                    "test_cases": len(domain_test_cases)
                }
        
        results["metrics"]["domain_specific"] = domain_results
        
        # Overall score
        overall_score = detection_accuracy * 0.6 + calibration_metrics["confidence_accuracy_correlation"] * 0.4
        results["metrics"]["overall_score"] = max(0.0, overall_score)
        
        return results
    
    def _initialize_benchmarks(self) -> List[DomainBenchmark]:
        """Initialize benchmark test cases for evaluation"""
        return [
            DomainBenchmark(
                topic="Machine Learning Algorithms for Software Development",
                expected_domain=ResearchDomain.TECHNOLOGY,
                expected_confidence=0.9,
                expected_keywords=["machine learning", "algorithms", "software", "development"],
                quality_baseline=QualityMetrics(
                    completeness=0.8, coherence=0.8, accuracy=0.9, citation_quality=0.7
                )
            ),
            DomainBenchmark(
                topic="Clinical Trial Results for Cancer Treatment",
                expected_domain=ResearchDomain.SCIENCE,
                expected_confidence=0.95,
                expected_keywords=["clinical", "trial", "cancer", "treatment"],
                quality_baseline=QualityMetrics(
                    completeness=0.9, coherence=0.9, accuracy=0.95, citation_quality=0.9
                )
            ),
            DomainBenchmark(
                topic="Market Analysis for E-commerce Platforms",
                expected_domain=ResearchDomain.BUSINESS,
                expected_confidence=0.85,
                expected_keywords=["market", "analysis", "e-commerce", "platforms"],
                quality_baseline=QualityMetrics(
                    completeness=0.8, coherence=0.8, accuracy=0.8, citation_quality=0.7
                )
            ),
            DomainBenchmark(
                topic="Theoretical Framework for Educational Psychology",
                expected_domain=ResearchDomain.ACADEMIC,
                expected_confidence=0.9,
                expected_keywords=["theoretical", "framework", "educational", "psychology"],
                quality_baseline=QualityMetrics(
                    completeness=0.9, coherence=0.9, accuracy=0.9, citation_quality=0.95
                )
            ),
            DomainBenchmark(
                topic="Overview of Climate Change Impacts",
                expected_domain=ResearchDomain.GENERAL,
                expected_confidence=0.7,
                expected_keywords=["overview", "climate", "change", "impacts"],
                quality_baseline=QualityMetrics(
                    completeness=0.8, coherence=0.8, accuracy=0.8, citation_quality=0.8
                )
            )
        ]


def create_domain_evaluation_report(
    domain_adapter: DomainAdapter,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive domain adaptation evaluation report.
    
    Args:
        domain_adapter: Domain adapter instance to evaluate
        output_file: Optional file path to save the report
        
    Returns:
        Evaluation report dictionary
    """
    evaluator = DomainEvaluator(domain_adapter)
    report = evaluator.run_comprehensive_evaluation()
    
    # Add system information
    report["system_info"] = {
        "domains_supported": [domain.value for domain in ResearchDomain],
        "total_strategies": len(domain_adapter.research_strategies),
        "total_terminology_handlers": len(domain_adapter.terminology_handlers),
        "evaluation_version": "1.0"
    }
    
    # Add recommendations
    overall_score = report["metrics"]["overall_score"]
    if overall_score < 0.7:
        report["recommendations"] = [
            "Consider improving domain detection algorithms",
            "Review and expand domain-specific keywords",
            "Enhance confidence calibration mechanisms"
        ]
    elif overall_score < 0.85:
        report["recommendations"] = [
            "Fine-tune domain-specific quality thresholds",
            "Expand terminology mappings for better consistency"
        ]
    else:
        report["recommendations"] = [
            "System performing well, consider adding more specialized domains"
        ]
    
    # Save report if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    return report