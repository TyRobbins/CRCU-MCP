#!/usr/bin/env python3
"""
Test Credit Union MCP Server Initialization
Validates that the server can be imported and initialized without critical errors
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_server_initialization():
    """Test that the server can be imported and initialized"""
    try:
        print("Testing Credit Union MCP Server initialization...")
        
        # Test import
        from src.main import CreditUnionMCP
        print("✓ Successfully imported CreditUnionMCP")
        
        # Test instantiation 
        mcp = CreditUnionMCP()
        print("✓ Successfully created CreditUnionMCP instance")
        
        # Test setup (without running)
        await mcp.setup()
        print("✓ Successfully completed MCP server setup")
        
        # Verify tools registration
        if hasattr(mcp.server, '_tool_handlers'):
            print(f"✓ MCP tools registered: {len(mcp.server._tool_handlers)} handlers")
        else:
            print("✓ MCP server initialized (tools registration method may vary)")
        
        print("\n🎉 All initialization tests passed!")
        print("The server is ready for MCP protocol communication.")
        
        return True
        
    except Exception as e:
        print(f"❌ Server initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import asyncio
    
    success = asyncio.run(test_server_initialization())
    sys.exit(0 if success else 1)
