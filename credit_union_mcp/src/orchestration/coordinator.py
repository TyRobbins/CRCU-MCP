"""
Agent Coordinator for Credit Union Analytics MCP

Manages and orchestrates interactions between multiple specialized agents,
routing requests to appropriate agents and coordinating multi-agent analyses.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from loguru import logger

from ..database.connection import DatabaseManager
from ..agents.base_agent import AnalysisContext, AnalysisResult
from ..agents.financial_performance import FinancialPerformanceAgent
from ..agents.portfolio_risk import PortfolioRiskAgent
from ..agents.member_analytics import MemberAnalyticsAgent
from ..agents.compliance import ComplianceAgent
from ..agents.operations import OperationsAgent
from .classifier import RequestClassifier


class AgentCoordinator:
    """
    Coordinates interactions between multiple specialized agents.
    
    Provides intelligent routing of analysis requests to appropriate agents,
    supports multi-agent orchestration, and aggregates results.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize the agent coordinator.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.logger = logger.bind(component="AgentCoordinator")
        
        # Initialize all agents
        self.agents = {
            'financial_performance': FinancialPerformanceAgent(db_manager),
            'portfolio_risk': PortfolioRiskAgent(db_manager),
            'member_analytics': MemberAnalyticsAgent(db_manager),
            'compliance': ComplianceAgent(db_manager),
            'operations': OperationsAgent(db_manager)
        }
        
        # Initialize request classifier
        self.classifier = RequestClassifier()
        
        self.logger.info(f"Initialized AgentCoordinator with {len(self.agents)} agents")
    
    async def route_request(self, context: AnalysisContext) -> AnalysisResult:
        """
        Route analysis request to appropriate agent(s).
        
        Args:
            context: Analysis context with request details
            
        Returns:
            AnalysisResult from the appropriate agent(s)
        """
        try:
            # Extract agent specification from context
            requested_agent = context.agent_type
            
            if requested_agent and requested_agent in self.agents:
                # Direct agent request
                return await self._execute_single_agent(requested_agent, context)
            elif requested_agent == 'multi_agent' or not requested_agent:
                # Multi-agent analysis or auto-routing
                return await self._execute_multi_agent_analysis(context)
            else:
                # Try to classify the request
                classified_agents = self.classifier.classify_request(context)
                
                if len(classified_agents) == 1:
                    return await self._execute_single_agent(classified_agents[0], context)
                else:
                    return await self._execute_multi_agent_analysis(context, classified_agents)
                    
        except Exception as e:
            self.logger.error(f"Error routing request: {e}")
            return AnalysisResult(
                agent="AgentCoordinator",
                analysis_type="routing_error",
                timestamp=datetime.now().isoformat(),
                success=False,
                errors=[str(e)],
                data={},
                metrics={},
                warnings=[],
                metadata={'error_type': type(e).__name__}
            )
    
    async def _execute_single_agent(self, agent_name: str, context: AnalysisContext) -> AnalysisResult:
        """
        Execute analysis with a single agent.
        
        Args:
            agent_name: Name of the agent to execute
            context: Analysis context
            
        Returns:
            AnalysisResult from the agent
        """
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        agent = self.agents[agent_name]
        self.logger.info(f"Executing single agent analysis: {agent_name}")
        
        # Execute analysis in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, agent.analyze, context)
        
        return result
    
    async def _execute_multi_agent_analysis(self, context: AnalysisContext, 
                                          agent_list: Optional[List[str]] = None) -> AnalysisResult:
        """
        Execute analysis with multiple agents in parallel.
        
        Args:
            context: Analysis context
            agent_list: Optional list of specific agents to execute
            
        Returns:
            Aggregated AnalysisResult from multiple agents
        """
        # Determine which agents to execute
        if agent_list:
            agents_to_execute = agent_list
        else:
            agents_to_execute = list(self.agents.keys())
        
        self.logger.info(f"Executing multi-agent analysis with: {agents_to_execute}")
        
        # Create tasks for parallel execution
        tasks = []
        for agent_name in agents_to_execute:
            if agent_name in self.agents:
                task = self._execute_single_agent(agent_name, context)
                tasks.append((agent_name, task))
        
        # Execute all agents in parallel
        results = {}
        errors = []
        
        for agent_name, task in tasks:
            try:
                result = await task
                results[agent_name] = result
            except Exception as e:
                self.logger.error(f"Error executing agent {agent_name}: {e}")
                errors.append(f"Agent {agent_name}: {str(e)}")
        
        # Aggregate results
        return self._aggregate_results(results, errors, context)
    
    def _aggregate_results(self, results: Dict[str, AnalysisResult], 
                          errors: List[str], context: AnalysisContext) -> AnalysisResult:
        """
        Aggregate results from multiple agents into a single response.
        
        Args:
            results: Dictionary of agent results
            errors: List of execution errors
            context: Original analysis context
            
        Returns:
            Aggregated AnalysisResult
        """
        # Prepare aggregated data
        aggregated_data = {
            'analysis_summary': {
                'total_agents_executed': len(results),
                'successful_analyses': len([r for r in results.values() if r.success]),
                'failed_analyses': len([r for r in results.values() if not r.success]),
                'execution_errors': len(errors)
            },
            'agent_results': {}
        }
        
        # Collect metrics from all agents
        aggregated_metrics = {}
        all_warnings = []
        all_errors = errors.copy()
        
        # Process each agent's results
        for agent_name, result in results.items():
            # Store individual agent results
            aggregated_data['agent_results'][agent_name] = {
                'success': result.success,
                'analysis_type': result.analysis_type,
                'data': result.data,
                'metrics': result.metrics,
                'warnings': result.warnings,
                'errors': result.errors
            }
            
            # Aggregate metrics with agent prefix
            for metric_name, metric_value in result.metrics.items():
                prefixed_name = f"{agent_name}_{metric_name}"
                aggregated_metrics[prefixed_name] = metric_value
            
            # Collect warnings and errors
            all_warnings.extend([f"{agent_name}: {w}" for w in result.warnings])
            all_errors.extend([f"{agent_name}: {e}" for e in result.errors])
        
        # Generate insights and recommendations
        insights = self._generate_cross_agent_insights(results)
        recommendations = self._generate_cross_agent_recommendations(results)
        
        aggregated_data['insights'] = insights
        aggregated_data['recommendations'] = recommendations
        
        # Determine overall success
        overall_success = len(results) > 0 and any(r.success for r in results.values())
        
        return AnalysisResult(
            agent="MultiAgent",
            analysis_type="comprehensive_analysis",
            timestamp=datetime.now().isoformat(),
            success=overall_success,
            data=aggregated_data,
            metrics=aggregated_metrics,
            warnings=all_warnings,
            errors=all_errors,
            metadata={
                'agents_executed': list(results.keys()),
                'analysis_date': context.as_of_date,
                'database': context.database
            }
        )
    
    def _generate_cross_agent_insights(self, results: Dict[str, AnalysisResult]) -> List[str]:
        """
        Generate insights by analyzing results across multiple agents.
        
        Args:
            results: Dictionary of agent results
            
        Returns:
            List of cross-agent insights
        """
        insights = []
        
        # Extract key metrics from each agent
        financial_metrics = {}
        risk_metrics = {}
        member_metrics = {}
        compliance_metrics = {}
        operations_metrics = {}
        
        if 'financial_performance' in results and results['financial_performance'].success:
            financial_metrics = results['financial_performance'].metrics
        
        if 'portfolio_risk' in results and results['portfolio_risk'].success:
            risk_metrics = results['portfolio_risk'].metrics
        
        if 'member_analytics' in results and results['member_analytics'].success:
            member_metrics = results['member_analytics'].metrics
        
        if 'compliance' in results and results['compliance'].success:
            compliance_metrics = results['compliance'].metrics
        
        if 'operations' in results and results['operations'].success:
            operations_metrics = results['operations'].metrics
        
        # Generate cross-agent insights
        
        # Financial performance vs risk correlation
        roa = financial_metrics.get('roa', 0)
        risk_score = risk_metrics.get('overall_risk_score', 0)
        
        if roa > 0 and risk_score > 0:
            if roa > 0.75 and risk_score < 40:
                insights.append("Strong financial performance with well-managed risk profile")
            elif roa < 0.5 and risk_score > 60:
                insights.append("Below-average profitability coupled with elevated risk levels")
        
        # Member growth vs operational efficiency
        member_growth = member_metrics.get('new_members_last_12m', 0)
        ops_efficiency = operations_metrics.get('overall_efficiency_score', 0)
        
        if member_growth > 0 and ops_efficiency > 0:
            if member_growth > 1000 and ops_efficiency > 80:
                insights.append("Healthy member growth supported by efficient operations")
            elif member_growth < 500 and ops_efficiency < 60:
                insights.append("Limited member growth may be hindered by operational inefficiencies")
        
        # Compliance vs financial performance
        compliance_score = compliance_metrics.get('compliance_score', 0)
        net_worth_ratio = financial_metrics.get('net_worth_ratio', 0)
        
        if compliance_score > 0 and net_worth_ratio > 0:
            if compliance_score > 90 and net_worth_ratio > 10:
                insights.append("Excellent compliance posture with strong capital position")
            elif compliance_score < 70 or net_worth_ratio < 8:
                insights.append("Compliance or capital adequacy concerns requiring attention")
        
        # Digital adoption vs member satisfaction
        digital_adoption = operations_metrics.get('channel_digital_adoption_rate', 0)
        member_satisfaction = member_metrics.get('avg_customer_satisfaction', 0)
        
        if digital_adoption > 0 and member_satisfaction > 0:
            if digital_adoption > 70 and member_satisfaction > 4.0:
                insights.append("High digital adoption correlates with strong member satisfaction")
            elif digital_adoption < 50:
                insights.append("Low digital adoption may impact member experience and efficiency")
        
        return insights if insights else ["Multiple analytics completed - review individual agent results for detailed insights"]
    
    def _generate_cross_agent_recommendations(self, results: Dict[str, AnalysisResult]) -> List[str]:
        """
        Generate recommendations based on cross-agent analysis.
        
        Args:
            results: Dictionary of agent results
            
        Returns:
            List of strategic recommendations
        """
        recommendations = []
        
        # Collect individual agent recommendations
        agent_recommendations = {}
        for agent_name, result in results.items():
            if result.success and result.data:
                agent_recs = result.data.get('recommendations', [])
                if agent_recs:
                    agent_recommendations[agent_name] = agent_recs
        
        # Generate strategic recommendations based on combined insights
        
        # Priority 1: Compliance and Risk Management
        compliance_issues = []
        risk_issues = []
        
        if 'compliance' in results and results['compliance'].success:
            compliance_data = results['compliance'].data
            if compliance_data.get('overall_status', {}).get('requires_immediate_attention', False):
                compliance_issues.append("Address critical compliance issues immediately")
        
        if 'portfolio_risk' in results and results['portfolio_risk'].success:
            risk_data = results['portfolio_risk'].data
            overall_assessment = risk_data.get('overall_assessment', {})
            if overall_assessment.get('risk_rating') in ['High Risk', 'Moderate-High Risk']:
                risk_issues.append("Implement risk mitigation strategies for portfolio")
        
        # Priority 2: Financial Performance Optimization
        financial_recommendations = []
        if 'financial_performance' in results and results['financial_performance'].success:
            financial_data = results['financial_performance'].data
            summary = financial_data.get('summary', {})
            if summary.get('overall_rating') in ['Needs Improvement', 'Satisfactory']:
                financial_recommendations.append("Focus on improving profitability and efficiency metrics")
        
        # Priority 3: Operational Excellence
        operational_recommendations = []
        if 'operations' in results and results['operations'].success:
            ops_data = results['operations'].data
            overall_perf = ops_data.get('overall_performance', {})
            if overall_perf.get('efficiency_rating') in ['Needs Improvement', 'Poor']:
                operational_recommendations.append("Implement operational efficiency improvements")
        
        # Priority 4: Member Experience Enhancement
        member_recommendations = []
        if 'member_analytics' in results and results['member_analytics'].success:
            member_data = results['member_analytics'].data
            if member_data.get('engagement', {}).get('online_banking_adoption', 0) < 60:
                member_recommendations.append("Enhance digital banking adoption and member engagement")
        
        # Combine recommendations by priority
        recommendations.extend(compliance_issues)
        recommendations.extend(risk_issues)
        recommendations.extend(financial_recommendations)
        recommendations.extend(operational_recommendations)
        recommendations.extend(member_recommendations)
        
        # Add strategic recommendations
        if len(agent_recommendations) >= 3:
            recommendations.append("Consider comprehensive strategic planning incorporating insights from all analytical areas")
        
        return recommendations if recommendations else ["Continue monitoring key performance indicators across all operational areas"]
    
    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """
        Get capabilities of all available agents.
        
        Returns:
            Dictionary mapping agent names to their capabilities
        """
        capabilities = {}
        for agent_name, agent in self.agents.items():
            capabilities[agent_name] = agent.get_capabilities()
        
        return capabilities
    
    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all agents.
        
        Returns:
            Dictionary with agent status information
        """
        status = {}
        for agent_name, agent in self.agents.items():
            status[agent_name] = {
                'name': agent.name,
                'capabilities_count': len(agent.get_capabilities()),
                'cache_entries': len(agent._cache) if hasattr(agent, '_cache') else 0,
                'available': True
            }
        
        return status
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all agents and coordinator.
        
        Returns:
            Dictionary with health check results
        """
        health_status = {
            'coordinator': 'healthy',
            'database_connection': 'unknown',
            'agents': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Test database connection
        try:
            db_status = self.db_manager.test_connection()
            health_status['database_connection'] = 'healthy' if db_status else 'unhealthy'
        except Exception as e:
            health_status['database_connection'] = f'error: {str(e)}'
        
        # Check each agent
        for agent_name, agent in self.agents.items():
            try:
                # Simple capability check
                capabilities = agent.get_capabilities()
                health_status['agents'][agent_name] = {
                    'status': 'healthy',
                    'capabilities_count': len(capabilities)
                }
            except Exception as e:
                health_status['agents'][agent_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        return health_status
    
    def clear_agent_caches(self) -> Dict[str, bool]:
        """
        Clear caches for all agents.
        
        Returns:
            Dictionary indicating cache clear success for each agent
        """
        results = {}
        
        for agent_name, agent in self.agents.items():
            try:
                if hasattr(agent, 'clear_cache'):
                    agent.clear_cache()
                results[agent_name] = True
            except Exception as e:
                self.logger.error(f"Failed to clear cache for {agent_name}: {e}")
                results[agent_name] = False
        
        return results
