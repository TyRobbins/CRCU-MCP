"""
Continuity Management System for Credit Union MCP Server

Monitors conversation context limits and manages conversation handoffs
to ensure task completion across multiple conversation sessions.
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class ConversationState:
    """Represents the current state of a conversation."""
    conversation_id: str
    start_time: datetime
    last_update: datetime
    total_tokens: int
    context_limit: int
    task_description: str
    progress_checklist: List[Dict[str, Any]]
    key_findings: List[str]
    current_phase: str
    critical_context: Dict[str, Any]
    handoff_count: int = 0
    completion_status: str = "in_progress"  # in_progress, completed, failed


@dataclass
class ConversationHandoff:
    """Represents a conversation handoff with continuation context."""
    source_conversation_id: str
    target_conversation_id: str
    handoff_time: datetime
    continuation_prompt: str
    preserved_context: Dict[str, Any]
    handoff_reason: str


class TokenEstimator:
    """Estimates token usage for conversation monitoring."""
    
    # Rough estimates based on common tokenization patterns
    CHARS_PER_TOKEN = 4
    WORDS_PER_TOKEN = 0.75
    
    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """
        Estimate token count for given text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
            
        # Use character-based estimation as baseline
        char_estimate = len(text) / cls.CHARS_PER_TOKEN
        
        # Use word-based estimation as alternative
        word_count = len(text.split())
        word_estimate = word_count / cls.WORDS_PER_TOKEN
        
        # Take the higher estimate for safety
        return int(max(char_estimate, word_estimate))
    
    @classmethod
    def estimate_conversation_tokens(cls, conversation_history: List[Dict[str, Any]]) -> int:
        """
        Estimate total tokens for entire conversation history.
        
        Args:
            conversation_history: List of conversation messages
            
        Returns:
            Estimated total token count
        """
        total_tokens = 0
        
        for message in conversation_history:
            if isinstance(message, dict):
                if 'content' in message:
                    total_tokens += cls.estimate_tokens(str(message['content']))
                if 'role' in message:
                    total_tokens += cls.estimate_tokens(str(message['role']))
            elif isinstance(message, str):
                total_tokens += cls.estimate_tokens(message)
                
        return total_tokens


class ConversationStateManager:
    """Manages conversation state persistence and retrieval."""
    
    def __init__(self, state_dir: Path):
        """
        Initialize state manager.
        
        Args:
            state_dir: Directory to store conversation states
        """
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        
    def _get_state_file(self, conversation_id: str) -> Path:
        """Get state file path for conversation ID."""
        safe_id = hashlib.md5(conversation_id.encode()).hexdigest()
        return self.state_dir / f"conversation_{safe_id}.pkl"
    
    def save_state(self, state: ConversationState) -> None:
        """
        Save conversation state to disk.
        
        Args:
            state: Conversation state to save
        """
        with self._lock:
            try:
                state_file = self._get_state_file(state.conversation_id)
                with open(state_file, 'wb') as f:
                    pickle.dump(state, f)
                logger.info(f"Saved conversation state for {state.conversation_id}")
            except Exception as e:
                logger.error(f"Failed to save conversation state: {e}")
    
    def load_state(self, conversation_id: str) -> Optional[ConversationState]:
        """
        Load conversation state from disk.
        
        Args:
            conversation_id: ID of conversation to load
            
        Returns:
            Loaded conversation state or None if not found
        """
        with self._lock:
            try:
                state_file = self._get_state_file(conversation_id)
                if state_file.exists():
                    with open(state_file, 'rb') as f:
                        state = pickle.load(f)
                    logger.info(f"Loaded conversation state for {conversation_id}")
                    return state
            except Exception as e:
                logger.error(f"Failed to load conversation state: {e}")
        return None
    
    def list_active_conversations(self, max_age_hours: int = 24) -> List[ConversationState]:
        """
        List all active conversations within specified age.
        
        Args:
            max_age_hours: Maximum age in hours for active conversations
            
        Returns:
            List of active conversation states
        """
        active_conversations = []
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        try:
            for state_file in self.state_dir.glob("conversation_*.pkl"):
                try:
                    with open(state_file, 'rb') as f:
                        state = pickle.load(f)
                    
                    if (state.last_update >= cutoff_time and 
                        state.completion_status == "in_progress"):
                        active_conversations.append(state)
                        
                except Exception as e:
                    logger.warning(f"Failed to load state file {state_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to list conversation states: {e}")
            
        return active_conversations
    
    def cleanup_old_states(self, max_age_days: int = 30) -> int:
        """
        Clean up old conversation state files.
        
        Args:
            max_age_days: Maximum age in days before cleanup
            
        Returns:
            Number of files cleaned up
        """
        cleanup_count = 0
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        try:
            for state_file in self.state_dir.glob("conversation_*.pkl"):
                try:
                    # Check file modification time
                    if datetime.fromtimestamp(state_file.stat().st_mtime) < cutoff_time:
                        state_file.unlink()
                        cleanup_count += 1
                        logger.info(f"Cleaned up old state file: {state_file}")
                        
                except Exception as e:
                    logger.warning(f"Failed to cleanup state file {state_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup conversation states: {e}")
            
        return cleanup_count


class PromptGenerator:
    """Generates continuation prompts for conversation handoffs."""
    
    def __init__(self, business_rules: Dict[str, Any]):
        """
        Initialize prompt generator.
        
        Args:
            business_rules: Business rules configuration for context
        """
        self.business_rules = business_rules
    
    def generate_continuation_prompt(self, 
                                   state: ConversationState,
                                   handoff_reason: str) -> str:
        """
        Generate continuation prompt for conversation handoff.
        
        Args:
            state: Current conversation state
            handoff_reason: Reason for the handoff
            
        Returns:
            Formatted continuation prompt
        """
        prompt_sections = []
        
        # Header
        prompt_sections.append("# CONTINUATION CONTEXT")
        prompt_sections.append(f"**Previous Conversation ID:** {state.conversation_id}")
        prompt_sections.append(f"**Handoff Reason:** {handoff_reason}")
        prompt_sections.append(f"**Handoff Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        prompt_sections.append("")
        
        # Task Context
        prompt_sections.append("## ORIGINAL TASK")
        prompt_sections.append(state.task_description)
        prompt_sections.append("")
        
        # Progress Status
        prompt_sections.append("## PROGRESS STATUS")
        prompt_sections.append(f"**Current Phase:** {state.current_phase}")
        prompt_sections.append(f"**Completion Status:** {state.completion_status}")
        
        if state.progress_checklist:
            prompt_sections.append("**Progress Checklist:**")
            for item in state.progress_checklist:
                status = "✓" if item.get('completed', False) else "◯"
                prompt_sections.append(f"- {status} {item.get('description', 'Unknown task')}")
        prompt_sections.append("")
        
        # Key Findings
        if state.key_findings:
            prompt_sections.append("## KEY FINDINGS FROM PREVIOUS CONVERSATION")
            for i, finding in enumerate(state.key_findings[-10:], 1):  # Last 10 findings
                prompt_sections.append(f"{i}. {finding}")
            prompt_sections.append("")
        
        # Critical Context
        if state.critical_context:
            prompt_sections.append("## CRITICAL CONTEXT TO PRESERVE")
            for key, value in state.critical_context.items():
                prompt_sections.append(f"**{key}:** {value}")
            prompt_sections.append("")
        
        # Technical Context
        prompt_sections.append("## TECHNICAL CONTEXT")
        prompt_sections.append("**MCP Server Type:** Credit Union MCP with Business Rules Enforcement")
        prompt_sections.append("**Database Connections:** ARCUSYM000, TEMENOS")
        prompt_sections.append("**Key Business Rules:**")
        
        if self.business_rules:
            rules_summary = self.business_rules.get('business_rules', {})
            if 'process_date_requirements' in rules_summary:
                prompt_sections.append("- ProcessDate filtering is MANDATORY for all queries")
            if 'active_record_indicators' in rules_summary:
                prompt_sections.append("- Active records: (CloseDate IS NULL OR CloseDate = '1900-01-01')")
            if 'charged_off_indicators' in rules_summary:
                prompt_sections.append("- Not charged off: (ChargeOffDate IS NULL OR ChargeOffDate = '1900-01-01')")
        
        prompt_sections.append("")
        
        # Next Steps
        prompt_sections.append("## IMMEDIATE NEXT STEPS")
        if state.progress_checklist:
            incomplete_items = [item for item in state.progress_checklist 
                              if not item.get('completed', False)]
            if incomplete_items:
                prompt_sections.append("Continue with the following uncompleted tasks:")
                for item in incomplete_items[:5]:  # Next 5 tasks
                    prompt_sections.append(f"- {item.get('description', 'Unknown task')}")
            else:
                prompt_sections.append("- Review completed work and finalize results")
        else:
            prompt_sections.append("- Continue with the original task requirements")
        
        prompt_sections.append("")
        prompt_sections.append("## CONTINUATION INSTRUCTIONS")
        prompt_sections.append("1. Acknowledge this continuation context")
        prompt_sections.append("2. Resume work from where the previous conversation left off")
        prompt_sections.append("3. Update progress tracking as you complete tasks")
        prompt_sections.append("4. Preserve critical findings and context for potential future handoffs")
        
        return "\n".join(prompt_sections)
    
    def generate_task_summary(self, state: ConversationState) -> str:
        """
        Generate a concise task summary for quick reference.
        
        Args:
            state: Conversation state to summarize
            
        Returns:
            Concise task summary
        """
        summary_parts = []
        
        summary_parts.append(f"**Task:** {state.task_description}")
        summary_parts.append(f"**Phase:** {state.current_phase}")
        summary_parts.append(f"**Status:** {state.completion_status}")
        
        if state.progress_checklist:
            completed = sum(1 for item in state.progress_checklist 
                          if item.get('completed', False))
            total = len(state.progress_checklist)
            summary_parts.append(f"**Progress:** {completed}/{total} tasks completed")
        
        if state.key_findings:
            summary_parts.append(f"**Key Findings:** {len(state.key_findings)} discoveries")
        
        summary_parts.append(f"**Handoffs:** {state.handoff_count}")
        
        return " | ".join(summary_parts)


class ContinuityManager:
    """Main continuity management system."""
    
    def __init__(self, 
                 context_limit: int = 180000,  # 180K tokens (90% of 200K)
                 warning_threshold: float = 0.75,  # Warn at 75%
                 state_dir: Optional[Path] = None,
                 business_rules: Optional[Dict[str, Any]] = None):
        """
        Initialize continuity manager.
        
        Args:
            context_limit: Maximum context tokens before handoff
            warning_threshold: Threshold (0.0-1.0) for warnings
            state_dir: Directory for state persistence
            business_rules: Business rules configuration
        """
        self.context_limit = context_limit
        self.warning_threshold = warning_threshold
        self.warning_limit = int(context_limit * warning_threshold)
        
        # Set up state directory
        if state_dir is None:
            state_dir = Path(__file__).parent.parent.parent / "conversation_states"
        self.state_manager = ConversationStateManager(state_dir)
        
        # Initialize prompt generator
        self.prompt_generator = PromptGenerator(business_rules or {})
        
        # Current conversation tracking
        self.current_conversation: Optional[ConversationState] = None
        self._conversation_history: List[Dict[str, Any]] = []
        
        logger.info(f"Continuity Manager initialized with {context_limit} token limit")
    
    def start_conversation(self, 
                         conversation_id: str,
                         task_description: str,
                         initial_context: Optional[Dict[str, Any]] = None) -> ConversationState:
        """
        Start tracking a new conversation.
        
        Args:
            conversation_id: Unique identifier for conversation
            task_description: Description of the task being worked on
            initial_context: Initial context to preserve
            
        Returns:
            New conversation state
        """
        # Check if conversation already exists
        existing_state = self.state_manager.load_state(conversation_id)
        if existing_state:
            logger.info(f"Resuming existing conversation: {conversation_id}")
            self.current_conversation = existing_state
            return existing_state
        
        # Create new conversation state
        state = ConversationState(
            conversation_id=conversation_id,
            start_time=datetime.now(),
            last_update=datetime.now(),
            total_tokens=0,
            context_limit=self.context_limit,
            task_description=task_description,
            progress_checklist=[],
            key_findings=[],
            current_phase="initialization",
            critical_context=initial_context or {},
            handoff_count=0,
            completion_status="in_progress"
        )
        
        self.current_conversation = state
        self.state_manager.save_state(state)
        
        logger.info(f"Started new conversation tracking: {conversation_id}")
        return state
    
    def update_conversation(self,
                          message_content: str,
                          role: str = "user",
                          phase: Optional[str] = None,
                          progress_update: Optional[List[Dict[str, Any]]] = None,
                          key_finding: Optional[str] = None,
                          critical_context_update: Optional[Dict[str, Any]] = None) -> None:
        """
        Update conversation state with new message and context.
        
        Args:
            message_content: Content of the message
            role: Role of the message sender (user/assistant)
            phase: Current phase of work
            progress_update: Updated progress checklist
            key_finding: Important finding to preserve
            critical_context_update: Updates to critical context
        """
        if not self.current_conversation:
            logger.warning("No active conversation to update")
            return
        
        # Add message to history
        message = {
            'role': role,
            'content': message_content,
            'timestamp': datetime.now().isoformat()
        }
        self._conversation_history.append(message)
        
        # Estimate and update token count
        message_tokens = TokenEstimator.estimate_tokens(message_content)
        self.current_conversation.total_tokens += message_tokens
        self.current_conversation.last_update = datetime.now()
        
        # Update phase if provided
        if phase:
            self.current_conversation.current_phase = phase
        
        # Update progress if provided
        if progress_update:
            self.current_conversation.progress_checklist = progress_update
        
        # Add key finding if provided
        if key_finding:
            self.current_conversation.key_findings.append(key_finding)
        
        # Update critical context if provided
        if critical_context_update:
            self.current_conversation.critical_context.update(critical_context_update)
        
        # Save updated state
        self.state_manager.save_state(self.current_conversation)
        
        # Check for warnings or handoff needs
        self._check_context_limits()
    
    def _check_context_limits(self) -> None:
        """Check current context usage and trigger warnings/handoffs."""
        if not self.current_conversation:
            return
        
        current_tokens = self.current_conversation.total_tokens
        
        if current_tokens >= self.context_limit:
            logger.critical(f"Context limit exceeded: {current_tokens}/{self.context_limit} tokens")
            self._trigger_handoff("context_limit_exceeded")
        elif current_tokens >= self.warning_limit:
            logger.warning(f"Context limit warning: {current_tokens}/{self.context_limit} tokens "
                         f"({current_tokens/self.context_limit*100:.1f}%)")
    
    def _trigger_handoff(self, reason: str) -> ConversationHandoff:
        """
        Trigger a conversation handoff.
        
        Args:
            reason: Reason for the handoff
            
        Returns:
            Handoff information
        """
        if not self.current_conversation:
            raise ValueError("No active conversation for handoff")
        
        # Generate new conversation ID for continuation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_id = f"{self.current_conversation.conversation_id}_cont_{timestamp}"
        
        # Create handoff record
        handoff = ConversationHandoff(
            source_conversation_id=self.current_conversation.conversation_id,
            target_conversation_id=target_id,
            handoff_time=datetime.now(),
            continuation_prompt=self.prompt_generator.generate_continuation_prompt(
                self.current_conversation, reason),
            preserved_context=self.current_conversation.critical_context.copy(),
            handoff_reason=reason
        )
        
        # Update source conversation
        self.current_conversation.completion_status = "handed_off"
        self.current_conversation.handoff_count += 1
        self.state_manager.save_state(self.current_conversation)
        
        logger.info(f"Triggered conversation handoff: {self.current_conversation.conversation_id} -> {target_id}")
        
        return handoff
    
    def get_continuation_prompt(self) -> Optional[str]:
        """
        Get continuation prompt if handoff is needed.
        
        Returns:
            Continuation prompt or None if not needed
        """
        if not self.current_conversation:
            return None
        
        current_tokens = self.current_conversation.total_tokens
        
        if current_tokens >= self.context_limit:
            handoff = self._trigger_handoff("context_limit_exceeded")
            return handoff.continuation_prompt
        
        return None
    
    def get_context_status(self) -> Dict[str, Any]:
        """
        Get current context usage status.
        
        Returns:
            Context status information
        """
        if not self.current_conversation:
            return {
                'status': 'no_active_conversation',
                'current_tokens': 0,
                'limit': self.context_limit,
                'usage_percentage': 0.0,
                'warning_threshold_reached': False,
                'handoff_needed': False
            }
        
        current_tokens = self.current_conversation.total_tokens
        usage_percentage = (current_tokens / self.context_limit) * 100
        
        return {
            'status': 'active',
            'conversation_id': self.current_conversation.conversation_id,
            'current_tokens': current_tokens,
            'limit': self.context_limit,
            'usage_percentage': usage_percentage,
            'warning_threshold_reached': current_tokens >= self.warning_limit,
            'handoff_needed': current_tokens >= self.context_limit,
            'current_phase': self.current_conversation.current_phase,
            'completion_status': self.current_conversation.completion_status,
            'handoff_count': self.current_conversation.handoff_count,
            'last_update': self.current_conversation.last_update.isoformat()
        }
    
    def complete_conversation(self, 
                            completion_status: str = "completed",
                            final_summary: Optional[str] = None) -> None:
        """
        Mark conversation as completed.
        
        Args:
            completion_status: Final completion status
            final_summary: Optional final summary
        """
        if not self.current_conversation:
            logger.warning("No active conversation to complete")
            return
        
        self.current_conversation.completion_status = completion_status
        self.current_conversation.last_update = datetime.now()
        
        if final_summary:
            self.current_conversation.key_findings.append(f"FINAL: {final_summary}")
        
        self.state_manager.save_state(self.current_conversation)
        
        logger.info(f"Completed conversation {self.current_conversation.conversation_id} "
                   f"with status: {completion_status}")
    
    def get_conversation_summary(self) -> Optional[str]:
        """
        Get summary of current conversation.
        
        Returns:
            Conversation summary or None if no active conversation
        """
        if not self.current_conversation:
            return None
        
        return self.prompt_generator.generate_task_summary(self.current_conversation)
    
    def list_active_conversations(self) -> List[Dict[str, Any]]:
        """
        List all active conversations.
        
        Returns:
            List of active conversation summaries
        """
        active_states = self.state_manager.list_active_conversations()
        
        summaries = []
        for state in active_states:
            summary = {
                'conversation_id': state.conversation_id,
                'task_description': state.task_description,
                'current_phase': state.current_phase,
                'completion_status': state.completion_status,
                'total_tokens': state.total_tokens,
                'usage_percentage': (state.total_tokens / state.context_limit) * 100,
                'handoff_count': state.handoff_count,
                'last_update': state.last_update.isoformat(),
                'progress_summary': self.prompt_generator.generate_task_summary(state)
            }
            summaries.append(summary)
        
        return summaries
    
    def cleanup_old_conversations(self, max_age_days: int = 30) -> int:
        """
        Clean up old conversation files.
        
        Args:
            max_age_days: Maximum age in days before cleanup
            
        Returns:
            Number of files cleaned up
        """
        return self.state_manager.cleanup_old_states(max_age_days)
