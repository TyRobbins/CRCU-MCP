"""
Request Classifier for Credit Union Analytics MCP

Intelligently classifies incoming analysis requests to determine which
agent(s) should handle the request based on content analysis and keywords.
"""

import re
from typing import Dict, Any, List, Optional
from ..agents.base_agent import AnalysisContext


class RequestClassifier:
    """
    Classifies analysis requests and routes them to appropriate agents.
    
    Uses keyword matching, pattern recognition, and context analysis
    to determine the best agent(s) for handling a request.
    """
    
    def __init__(self):
        """Initialize the request classifier with agent patterns."""
        self.agent_patterns = {
            'financial_performance': {
                'keywords': [
                    'roa', 'roe', 'nim', 'profitability', 'income', 'revenue', 'expenses',
                    'financial performance', 'earnings', 'profit', 'loss', 'ncua', 'call report',
                    'efficiency ratio', 'net worth', 'capital ratio', 'asset quality',
                    'yield', 'spread', 'margin', 'cost of funds'
                ],
                'patterns': [
                    r'return on (assets?|equity)',
                    r'net interest margin',
                    r'operating efficiency',
                    r'financial (performance|analysis|metrics)',
                    r'profit(ability)?.*analysis',
                    r'income.*statement',
                    r'(quarterly|annual).*results'
                ]
            },
            'portfolio_risk': {
                'keywords': [
                    'risk', 'concentration', 'hhi', 'delinquency', 'charge.off', 'loss',
                    'default', 'credit risk', 'stress test', 'monte carlo', 'var',
                    'portfolio risk', 'loan risk', 'vintage', 'migration', 'provision',
                    'allowance', 'impairment', 'recovery'
                ],
                'patterns': [
                    r'portfolio.*risk',
                    r'concentration.*analysis',
                    r'stress.*test(ing)?',
                    r'delinquency.*(trend|forecast)',
                    r'credit.*quality',
                    r'loss.*(rate|analysis)',
                    r'risk.*assessment'
                ]
            },
            'member_analytics': {
                'keywords': [
                    'member', 'customer', 'rfm', 'segmentation', 'clustering', 'ltv',
                    'lifetime value', 'churn', 'retention', 'cross.sell', 'demographics',
                    'behavior', 'engagement', 'satisfaction', 'loyalty', 'acquisition'
                ],
                'patterns': [
                    r'member.*analytics',
                    r'customer.*(segmentation|analysis|behavior)',
                    r'rfm.*analysis',
                    r'lifetime.*value',
                    r'churn.*(prediction|analysis)',
                    r'member.*(retention|acquisition)',
                    r'cross.*sell'
                ]
            },
            'compliance': {
                'keywords': [
                    'compliance', 'regulation', 'bsa', 'aml', 'suspicious', 'ctr', 'sar',
                    'capital adequacy', 'ncua', 'regulatory', 'audit', 'examination',
                    'mbl', 'member business loan', 'cecl', 'allowance', 'provision'
                ],
                'patterns': [
                    r'bsa.*aml',
                    r'suspicious.*activity',
                    r'capital.*adequacy',
                    r'regulatory.*compliance',
                    r'member.*business.*loan',
                    r'cecl.*(calculation|analysis)',
                    r'compliance.*(monitoring|check)'
                ]
            },
            'operations': {
                'keywords': [
                    'operations', 'efficiency', 'branch', 'staff', 'productivity', 'cost',
                    'channel', 'digital', 'atm', 'online banking', 'mobile banking',
                    'process', 'workflow', 'performance', 'utilization', 'capacity'
                ],
                'patterns': [
                    r'operational.*efficiency',
                    r'branch.*performance',
                    r'staff.*productivity',
                    r'channel.*analysis',
                    r'digital.*adoption',
                    r'cost.*(analysis|efficiency)',
                    r'process.*optimization'
                ]
            }
        }
        
        # Multi-agent trigger patterns
        self.multi_agent_patterns = [
            r'comprehensive.*analysis',
            r'overall.*performance',
            r'dashboard',
            r'executive.*summary',
            r'complete.*assessment',
            r'full.*analysis',
            r'enterprise.*view'
        ]
    
    def classify_request(self, context: AnalysisContext) -> List[str]:
        """
        Classify an analysis request to determine appropriate agent(s).
        
        Args:
            context: Analysis context containing request details
            
        Returns:
            List of agent names that should handle the request
        """
        # If agent is explicitly specified, use that
        if context.agent_type and context.agent_type in self.agent_patterns:
            return [context.agent_type]
        
        # Extract text to analyze from context
        analysis_text = self._extract_analysis_text(context)
        
        if not analysis_text:
            # Default to comprehensive analysis if no text to analyze
            return list(self.agent_patterns.keys())
        
        # Check for multi-agent patterns first
        if self._matches_multi_agent_patterns(analysis_text):
            return list(self.agent_patterns.keys())
        
        # Score each agent based on keyword and pattern matches
        agent_scores = self._score_agents(analysis_text)
        
        # Determine which agents to activate based on scores
        selected_agents = self._select_agents(agent_scores)
        
        return selected_agents if selected_agents else ['financial_performance']  # Default fallback
    
    def _extract_analysis_text(self, context: AnalysisContext) -> str:
        """
        Extract text content from context for analysis.
        
        Args:
            context: Analysis context
            
        Returns:
            Combined text for analysis
        """
        text_parts = []
        
        # Add agent type if specified
        if context.agent_type:
            text_parts.append(context.agent_type)
        
        # Add analysis type if specified
        if hasattr(context, 'analysis_type') and context.analysis_type:
            text_parts.append(context.analysis_type)
        
        # Add parameters as text
        if context.parameters:
            for key, value in context.parameters.items():
                text_parts.append(f"{key} {value}")
        
        # Add any description or query text
        if hasattr(context, 'description') and context.description:
            text_parts.append(context.description)
        
        if hasattr(context, 'query') and context.query:
            text_parts.append(context.query)
        
        return ' '.join(text_parts).lower()
    
    def _matches_multi_agent_patterns(self, text: str) -> bool:
        """
        Check if text matches multi-agent analysis patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if multi-agent analysis is indicated
        """
        for pattern in self.multi_agent_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _score_agents(self, text: str) -> Dict[str, float]:
        """
        Score each agent based on keyword and pattern matches.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary mapping agent names to scores
        """
        scores = {}
        
        for agent_name, patterns in self.agent_patterns.items():
            score = 0.0
            
            # Score based on keyword matches
            keyword_matches = 0
            for keyword in patterns['keywords']:
                # Use word boundaries for more precise matching
                pattern = r'\b' + re.escape(keyword.replace('.', r'\.')) + r'\b'
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                keyword_matches += matches
            
            # Weight keyword matches
            score += keyword_matches * 1.0
            
            # Score based on regex pattern matches
            pattern_matches = 0
            for pattern in patterns['patterns']:
                if re.search(pattern, text, re.IGNORECASE):
                    pattern_matches += 1
            
            # Weight pattern matches higher
            score += pattern_matches * 2.0
            
            scores[agent_name] = score
        
        return scores
    
    def _select_agents(self, scores: Dict[str, float], threshold: float = 1.0) -> List[str]:
        """
        Select agents based on scores.
        
        Args:
            scores: Agent scores
            threshold: Minimum score threshold
            
        Returns:
            List of selected agent names
        """
        # Sort agents by score
        sorted_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select agents above threshold
        selected = []
        max_score = sorted_agents[0][1] if sorted_agents else 0
        
        for agent_name, score in sorted_agents:
            # Include agent if:
            # 1. Score is above threshold, OR
            # 2. Score is within 50% of max score and max score > 0
            if score >= threshold or (max_score > 0 and score >= max_score * 0.5):
                selected.append(agent_name)
        
        # If no agents selected but we have scores, take the top scorer
        if not selected and sorted_agents and sorted_agents[0][1] > 0:
            selected.append(sorted_agents[0][0])
        
        return selected
    
    def get_agent_keywords(self, agent_name: str) -> List[str]:
        """
        Get keywords associated with a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List of keywords for the agent
        """
        if agent_name in self.agent_patterns:
            return self.agent_patterns[agent_name]['keywords']
        return []
    
    def get_all_keywords(self) -> Dict[str, List[str]]:
        """
        Get all keywords for all agents.
        
        Returns:
            Dictionary mapping agent names to their keywords
        """
        return {
            agent: patterns['keywords'] 
            for agent, patterns in self.agent_patterns.items()
        }
    
    def suggest_agent(self, query: str) -> str:
        """
        Suggest the best agent for a simple text query.
        
        Args:
            query: Text query
            
        Returns:
            Suggested agent name
        """
        # Create a minimal context for classification
        from ..agents.base_agent import AnalysisContext
        
        context = AnalysisContext(
            agent_type=None,
            database="TEMENOS",
            parameters={'query': query}
        )
        
        agents = self.classify_request(context)
        return agents[0] if agents else 'financial_performance'
    
    def explain_classification(self, context: AnalysisContext) -> Dict[str, Any]:
        """
        Explain why specific agents were selected for a request.
        
        Args:
            context: Analysis context
            
        Returns:
            Dictionary with classification explanation
        """
        text = self._extract_analysis_text(context)
        scores = self._score_agents(text)
        selected = self._select_agents(scores)
        
        # Find matching keywords and patterns for each agent
        explanations = {}
        
        for agent_name, score in scores.items():
            matched_keywords = []
            matched_patterns = []
            
            # Find matching keywords
            patterns = self.agent_patterns[agent_name]
            for keyword in patterns['keywords']:
                pattern = r'\b' + re.escape(keyword.replace('.', r'\.')) + r'\b'
                if re.search(pattern, text, re.IGNORECASE):
                    matched_keywords.append(keyword)
            
            # Find matching patterns
            for pattern in patterns['patterns']:
                if re.search(pattern, text, re.IGNORECASE):
                    matched_patterns.append(pattern)
            
            explanations[agent_name] = {
                'score': score,
                'selected': agent_name in selected,
                'matched_keywords': matched_keywords,
                'matched_patterns': matched_patterns
            }
        
        return {
            'analyzed_text': text,
            'selected_agents': selected,
            'agent_explanations': explanations,
            'multi_agent_triggered': self._matches_multi_agent_patterns(text)
        }
