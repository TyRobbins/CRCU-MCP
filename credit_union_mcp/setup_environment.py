"""
Environment Setup Script for Credit Union Analytics MCP

This script helps configure the Windows environment for running the
Credit Union MCP server by setting up Python paths and dependencies.
"""

import os
import sys
import subprocess
import winreg
from pathlib import Path
from loguru import logger


class EnvironmentSetup:
    """Setup Windows environment for Credit Union MCP."""
    
    def __init__(self):
        """Initialize the environment setup."""
        self.python_path = sys.executable
        self.python_dir = str(Path(self.python_path).parent)
        self.scripts_dir = str(Path(self.python_path).parent / "Scripts")
        
    def check_python_in_path(self):
        """Check if Python is already in the system PATH."""
        try:
            result = subprocess.run(['python', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Python found in PATH: {result.stdout.strip()}")
                return True
            else:
                logger.warning("Python not found in PATH")
                return False
        except FileNotFoundError:
            logger.warning("Python not found in PATH")
            return False
    
    def get_current_path(self):
        """Get the current system PATH variable."""
        try:
            # Try to get system PATH first
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                              r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
                system_path, _ = winreg.QueryValueEx(key, "PATH")
                return system_path
        except Exception:
            # Fall back to user PATH
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                    user_path, _ = winreg.QueryValueEx(key, "PATH")
                    return user_path
            except Exception:
                return os.environ.get("PATH", "")
    
    def add_python_to_path(self, use_user_path=True):
        """Add Python to Windows PATH environment variable."""
        logger.info("Adding Python to Windows PATH...")
        
        current_path = self.get_current_path()
        
        # Check if Python directories are already in PATH
        paths_to_add = []
        if self.python_dir not in current_path:
            paths_to_add.append(self.python_dir)
        if self.scripts_dir not in current_path:
            paths_to_add.append(self.scripts_dir)
        
        if not paths_to_add:
            logger.info("Python is already in PATH")
            return True
        
        # Add paths
        new_path = current_path
        for path in paths_to_add:
            if new_path and not new_path.endswith(';'):
                new_path += ';'
            new_path += path
            logger.info(f"Adding to PATH: {path}")
        
        try:
            if use_user_path:
                # Add to user environment (doesn't require admin)
                key_path = "Environment"
                hkey = winreg.HKEY_CURRENT_USER
                logger.info("Adding to user PATH (no admin required)")
            else:
                # Add to system environment (requires admin)
                key_path = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
                hkey = winreg.HKEY_LOCAL_MACHINE
                logger.info("Adding to system PATH (requires admin)")
            
            with winreg.OpenKey(hkey, key_path, 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_path)
            
            logger.info("Python paths added to Windows PATH successfully!")
            logger.info("Please restart your command prompt or system for changes to take effect.")
            return True
            
        except PermissionError:
            logger.error("Permission denied. Try running as administrator or use user PATH.")
            return False
        except Exception as e:
            logger.error(f"Failed to update PATH: {e}")
            return False
    
    def create_batch_scripts(self):
        """Create convenient batch scripts for running the MCP server."""
        logger.info("Creating batch scripts...")
        
        # Create start script
        start_script = f"""@echo off
title Credit Union Analytics MCP Server
echo.
echo Credit Union Analytics MCP Server
echo ===================================
echo.
echo Python Path: {self.python_path}
echo Project Directory: {Path(__file__).parent}
echo.
echo Starting server...
echo.

cd /d "{Path(__file__).parent}"
"{self.python_path}" -m src.main

echo.
echo Server stopped. Press any key to exit...
pause > nul
"""
        
        with open("start_mcp_server.bat", "w") as f:
            f.write(start_script)
        
        # Create build script
        build_script = f"""@echo off
title Credit Union MCP Build
echo.
echo Credit Union Analytics MCP Build
echo =================================
echo.
echo Building standalone executable...
echo.

cd /d "{Path(__file__).parent}"
"{self.python_path}" build.py

echo.
echo Build complete. Press any key to exit...
pause > nul
"""
        
        with open("build_mcp.bat", "w") as f:
            f.write(build_script)
        
        # Create install dependencies script
        install_script = f"""@echo off
title Install MCP Dependencies
echo.
echo Credit Union Analytics MCP Dependencies
echo =======================================
echo.
echo Installing Python dependencies...
echo.

cd /d "{Path(__file__).parent}"
"{self.python_path}" -m pip install --upgrade pip
"{self.python_path}" -m pip install -r requirements.txt

echo.
echo Dependencies installed. Press any key to exit...
pause > nul
"""
        
        with open("install_dependencies.bat", "w") as f:
            f.write(install_script)
        
        logger.info("Created batch scripts:")
        logger.info("  - start_mcp_server.bat (start the MCP server)")
        logger.info("  - build_mcp.bat (build executable)")
        logger.info("  - install_dependencies.bat (install Python packages)")
    
    def verify_dependencies(self):
        """Verify that all required dependencies are installed."""
        logger.info("Verifying Python dependencies...")
        
        # Map package names to their import names
        package_imports = {
            'mcp': 'mcp',
            'pyodbc': 'pyodbc', 
            'sqlalchemy': 'sqlalchemy',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'scipy': 'scipy',
            'scikit-learn': 'sklearn',
            'pydantic': 'pydantic',
            'pyyaml': 'yaml',
            'loguru': 'loguru',
            'pyinstaller': 'PyInstaller'
        }
        
        missing_packages = []
        installed_packages = []
        
        for package_name, import_name in package_imports.items():
            try:
                __import__(import_name)
                installed_packages.append(package_name)
                logger.info(f"✓ {package_name}")
            except ImportError:
                missing_packages.append(package_name)
                logger.error(f"✗ {package_name}")
        
        if missing_packages:
            logger.warning(f"Missing packages: {missing_packages}")
            logger.info("Run 'install_dependencies.bat' or 'pip install -r requirements.txt'")
            return False
        else:
            logger.info(f"All {len(installed_packages)} required packages are installed!")
            return True
    
    def setup_all(self):
        """Run complete environment setup."""
        logger.info("=" * 60)
        logger.info("Credit Union MCP Environment Setup")
        logger.info("=" * 60)
        
        try:
            # Check current Python setup
            logger.info(f"Python executable: {self.python_path}")
            logger.info(f"Python version: {sys.version}")
            
            # Check if Python is in PATH
            python_in_path = self.check_python_in_path()
            
            # Add Python to PATH if needed
            if not python_in_path:
                logger.info("Python not found in PATH. Adding...")
                if not self.add_python_to_path():
                    logger.error("Failed to add Python to PATH")
                    return False
            
            # Create batch scripts
            self.create_batch_scripts()
            
            # Verify dependencies
            deps_ok = self.verify_dependencies()
            
            logger.info("=" * 60)
            if deps_ok:
                logger.info("ENVIRONMENT SETUP COMPLETED SUCCESSFULLY!")
                logger.info("=" * 60)
                logger.info("Next steps:")
                logger.info("1. Restart your command prompt (if PATH was updated)")
                logger.info("2. Run 'start_mcp_server.bat' to start the server")
                logger.info("3. Or run 'build_mcp.bat' to create standalone executable")
            else:
                logger.warning("ENVIRONMENT SETUP COMPLETED WITH WARNINGS!")
                logger.info("=" * 60)
                logger.info("Next steps:")
                logger.info("1. Run 'install_dependencies.bat' to install missing packages")
                logger.info("2. Re-run this setup script")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            return False


def main():
    """Main setup function."""
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", 
              format="{time:HH:mm:ss} | {level} | {message}")
    
    setup = EnvironmentSetup()
    success = setup.setup_all()
    
    if success:
        logger.info("Setup completed!")
    else:
        logger.error("Setup failed!")
    
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
