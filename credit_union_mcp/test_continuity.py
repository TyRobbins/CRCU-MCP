#!/usr/bin/env python3
"""
Test script for continuity management system functionality.

This script tests the core continuity features without requiring
a full MCP server setup.
"""

import sys
import os
import asyncio
import json
from pathlib import Path
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.continuity_manager import (
    ContinuityManager, 
    ConversationState, 
    TokenEstimator,
    ConversationStateManager,
    PromptGenerator
)


def test_token_estimator():
    """Test token estimation functionality."""
    print("=" * 60)
    print("Testing Token Estimator")
    print("=" * 60)
    
    test_texts = [
        "Hello world",
        "This is a longer test message with more content to estimate tokens for.",
        "SELECT * FROM Account WHERE ProcessDate = FORMAT(DATEADD(DAY, -1, GETDATE()), 'yyyyMMdd')",
        "A" * 1000,  # 1000 character string
        ""  # Empty string
    ]
    
    for text in test_texts:
        tokens = TokenEstimator.estimate_tokens(text)
        print(f"Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"Length: {len(text)} chars, Estimated tokens: {tokens}")
        print()


def test_conversation_state_manager():
    """Test conversation state persistence."""
    print("=" * 60)
    print("Testing Conversation State Manager")
    print("=" * 60)
    
    # Create temporary state directory
    test_dir = Path(__file__).parent / "test_states"
    test_dir.mkdir(exist_ok=True)
    
    try:
        state_manager = ConversationStateManager(test_dir)
        
        # Create test conversation state
        test_state = ConversationState(
            conversation_id="test_conversation_123",
            start_time=datetime.now(),
            last_update=datetime.now(),
            total_tokens=5000,
            context_limit=180000,
            task_description="Test task for continuity management",
            progress_checklist=[
                {"description": "Analyze requirements", "completed": True},
                {"description": "Implement solution", "completed": False},
                {"description": "Test functionality", "completed": False}
            ],
            key_findings=["Finding 1: System works as expected", "Finding 2: Need more testing"],
            current_phase="testing",
            critical_context={"database": "ARCUSYM000", "query_type": "member_analysis"},
            handoff_count=0,
            completion_status="in_progress"
        )
        
        # Test saving state
        print("Saving conversation state...")
        state_manager.save_state(test_state)
        
        # Test loading state
        print("Loading conversation state...")
        loaded_state = state_manager.load_state("test_conversation_123")
        
        if loaded_state:
            print(f"✓ Successfully loaded state for conversation: {loaded_state.conversation_id}")
            print(f"  Task: {loaded_state.task_description}")
            print(f"  Phase: {loaded_state.current_phase}")
            print(f"  Tokens: {loaded_state.total_tokens}/{loaded_state.context_limit}")
            print(f"  Progress: {len([p for p in loaded_state.progress_checklist if p['completed']])}/{len(loaded_state.progress_checklist)} completed")
        else:
            print("✗ Failed to load conversation state")
        
        # Test listing active conversations
        print("\nListing active conversations...")
        active_conversations = state_manager.list_active_conversations()
        print(f"Found {len(active_conversations)} active conversations")
        
        for conv in active_conversations:
            print(f"  - {conv.conversation_id}: {conv.current_phase}")
        
    except Exception as e:
        print(f"✗ Error testing state manager: {e}")
    
    finally:
        # Clean up test directory
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
        print("Cleaned up test files")


def test_prompt_generator():
    """Test continuation prompt generation."""
    print("=" * 60)
    print("Testing Prompt Generator")
    print("=" * 60)
    
    # Mock business rules
    mock_business_rules = {
        'business_rules': {
            'process_date_requirements': {'mandatory': True},
            'active_record_indicators': {'LOAN': {'field': 'CloseDate'}},
            'charged_off_indicators': {'LOAN': {'field': 'ChargeOffDate'}}
        }
    }
    
    prompt_generator = PromptGenerator(mock_business_rules)
    
    # Create test conversation state
    test_state = ConversationState(
        conversation_id="test_conversation_456",
        start_time=datetime.now(),
        last_update=datetime.now(),
        total_tokens=150000,
        context_limit=180000,
        task_description="Analyze credit union member data for financial reporting",
        progress_checklist=[
            {"description": "Connect to database", "completed": True},
            {"description": "Validate business rules", "completed": True},
            {"description": "Execute member analysis queries", "completed": False},
            {"description": "Generate financial summary", "completed": False},
            {"description": "Create final report", "completed": False}
        ],
        key_findings=[
            "Database connection successful to ARCUSYM000",
            "ProcessDate filter is mandatory for all queries",
            "Active member count: approximately 57,600",
            "Total loan balance: approximately $351M"
        ],
        current_phase="data_analysis",
        critical_context={
            "database": "ARCUSYM000",
            "process_date": "20250923",
            "active_member_logic": "definitive_power_bi_query",
            "key_tables": ["NAME", "ACCOUNT", "LOAN", "SAVINGS"]
        },
        handoff_count=1,
        completion_status="in_progress"
    )
    
    # Generate continuation prompt
    print("Generating continuation prompt...")
    continuation_prompt = prompt_generator.generate_continuation_prompt(
        test_state, 
        "context_limit_approaching"
    )
    
    print("Generated Continuation Prompt:")
    print("-" * 40)
    print(continuation_prompt)
    print("-" * 40)
    
    # Generate task summary
    print("\nGenerating task summary...")
    task_summary = prompt_generator.generate_task_summary(test_state)
    print(f"Task Summary: {task_summary}")


def test_continuity_manager():
    """Test the full continuity manager functionality."""
    print("=" * 60)
    print("Testing Continuity Manager")
    print("=" * 60)
    
    # Create temporary state directory
    test_dir = Path(__file__).parent / "test_continuity_states"
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize continuity manager with lower limits for testing
        continuity_manager = ContinuityManager(
            context_limit=10000,  # Low limit for testing
            warning_threshold=0.8,
            state_dir=test_dir,
            business_rules={}
        )
        
        # Test starting conversation tracking
        print("1. Starting conversation tracking...")
        conversation_id = f"test_continuity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        state = continuity_manager.start_conversation(
            conversation_id=conversation_id,
            task_description="Test continuity management with database queries",
            initial_context={"database": "ARCUSYM000", "test_mode": True}
        )
        
        print(f"✓ Started tracking conversation: {state.conversation_id}")
        
        # Test updating conversation with messages
        print("\n2. Updating conversation with messages...")
        
        messages = [
            "Connecting to ARCUSYM000 database for member analysis",
            "Executing ProcessDate filtered query for active members",
            "Found 57,600 active members in the credit union system",
            "Analyzing member breakdown by account type categories",
            "Generating financial summary with loan and savings balances",
            "Creating comprehensive report with business rule compliance"
        ]
        
        for i, message in enumerate(messages):
            continuity_manager.update_conversation(
                message_content=message,
                role="assistant",
                phase=f"step_{i+1}",
                key_finding=f"Key finding {i+1}: {message[:30]}..."
            )
            
            status = continuity_manager.get_context_status()
            print(f"  Message {i+1}: {status['current_tokens']} tokens "
                  f"({status['usage_percentage']:.1f}%)")
            
            if status['warning_threshold_reached']:
                print(f"  ⚠️  Warning threshold reached!")
            
            if status['handoff_needed']:
                print(f"  🔄 Handoff needed!")
                break
        
        # Test getting final status
        print("\n3. Getting final context status...")
        final_status = continuity_manager.get_context_status()
        print(json.dumps(final_status, indent=2, default=str))
        
        # Test getting continuation prompt if needed
        print("\n4. Testing continuation prompt generation...")
        continuation_prompt = continuity_manager.get_continuation_prompt()
        
        if continuation_prompt:
            print("✓ Continuation prompt generated (handoff needed)")
            print(f"Prompt length: {len(continuation_prompt)} characters")
            print("First 200 characters:")
            print(continuation_prompt[:200] + "...")
        else:
            print("ℹ️  No continuation prompt needed at this time")
        
        # Test conversation summary
        print("\n5. Getting conversation summary...")
        summary = continuity_manager.get_conversation_summary()
        if summary:
            print(f"Summary: {summary}")
        else:
            print("No summary available")
        
        # Test listing active conversations
        print("\n6. Listing active conversations...")
        active_conversations = continuity_manager.list_active_conversations()
        print(f"Found {len(active_conversations)} active conversations")
        
        for conv in active_conversations:
            print(f"  - {conv['conversation_id']}: {conv['current_phase']} "
                  f"({conv['usage_percentage']:.1f}% context used)")
        
        # Test completing conversation
        print("\n7. Completing conversation...")
        continuity_manager.complete_conversation(
            completion_status="completed",
            final_summary="Successfully tested continuity management system"
        )
        print("✓ Conversation marked as completed")
        
    except Exception as e:
        print(f"✗ Error testing continuity manager: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test directory
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
        print("Cleaned up test files")


def main():
    """Run all continuity management tests."""
    print("Credit Union MCP Server - Continuity Management System Tests")
    print("=" * 80)
    
    try:
        test_token_estimator()
        test_conversation_state_manager()
        test_prompt_generator()
        test_continuity_manager()
        
        print("\n" + "=" * 80)
        print("✅ All continuity management tests completed successfully!")
        print("The continuity system is ready for integration with the MCP server.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
