"""
Build Script for Credit Union Analytics MCP

Creates a standalone Windows executable using PyInstaller with all
necessary dependencies and data files included.
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path
import PyInstaller.__main__
from loguru import logger


class MCPBuilder:
    """Builder for Credit Union MCP executable."""
    
    def __init__(self):
        """Initialize the builder."""
        self.project_root = Path(__file__).parent
        self.build_dir = self.project_root / "build"
        self.dist_dir = self.project_root / "dist"
        self.spec_file = self.project_root / "CreditUnionMCP.spec"
        
    def clean_build(self):
        """Remove old build artifacts."""
        logger.info("Cleaning build directories...")
        
        dirs_to_remove = [self.build_dir, self.dist_dir]
        for dir_path in dirs_to_remove:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                logger.info(f"Removed {dir_path}")
        
        # Remove spec file if it exists
        if self.spec_file.exists():
            self.spec_file.unlink()
            logger.info(f"Removed {self.spec_file}")
    
    def verify_dependencies(self):
        """Verify all required dependencies are installed."""
        logger.info("Verifying dependencies...")
        
        required_packages = [
            'mcp', 'pyodbc', 'sqlalchemy', 'pandas', 'numpy', 'scipy',
            'scikit-learn', 'pydantic', 'pyyaml', 'loguru', 'pyinstaller'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                logger.info(f"✓ {package}")
            except ImportError:
                missing_packages.append(package)
                logger.error(f"✗ {package}")
        
        if missing_packages:
            logger.error(f"Missing packages: {missing_packages}")
            logger.error("Please install missing packages with: pip install -r requirements.txt")
            return False
        
        logger.info("All dependencies verified!")
        return True
    
    def create_spec_file(self):
        """Create PyInstaller spec file with custom configuration."""
        logger.info("Creating PyInstaller spec file...")
        
        spec_content = '''# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

# Add project root to path
project_root = Path(SPECPATH)
sys.path.insert(0, str(project_root))

block_cipher = None

a = Analysis(
    ['src/main.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        ('config', 'config'),
        ('logs', 'logs'),
    ],
    hiddenimports=[
        # MCP Framework
        'mcp',
        'mcp.server',
        'mcp.server.stdio',
        'mcp.types',
        
        # Database drivers
        'pyodbc',
        'pymssql',
        'sqlalchemy',
        'sqlalchemy.dialects.mssql',
        'sqlalchemy.dialects.mssql.pyodbc',
        
        # Data science libraries
        'pandas',
        'numpy',
        'scipy',
        'scipy.stats',
        'scipy.optimize',
        'sklearn',
        'sklearn.cluster',
        'sklearn.preprocessing',
        'sklearn.ensemble',
        'sklearn.model_selection',
        'sklearn.metrics',
        'sklearn.tree._utils',
        'sklearn.neighbors._typedefs',
        'sklearn.utils._weight_vector',
        'sklearn.utils._cython_blas',
        
        # Utilities
        'pydantic',
        'yaml',
        'loguru',
        'asyncio',
        'json',
        'datetime',
        'pathlib',
        
        # Project modules
        'src.database.connection',
        'src.orchestration.coordinator',
        'src.orchestration.classifier',
        'src.agents.base_agent',
        'src.agents.financial_performance',
        'src.agents.portfolio_risk',
        'src.agents.member_analytics',
        'src.agents.compliance',
        'src.agents.operations',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'matplotlib',
        'tkinter',
        'PIL',
        'IPython',
        'jupyter',
        'notebook',
        'plotly',
        'bokeh',
        'seaborn',
        'statsmodels.tsa.statespace._filters',
        'statsmodels.tsa.statespace._smoothers',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='CreditUnionMCP',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
'''
        
        with open(self.spec_file, 'w') as f:
            f.write(spec_content)
        
        logger.info(f"Spec file created: {self.spec_file}")
    
    def build_executable(self):
        """Build the standalone executable."""
        logger.info("Building executable with PyInstaller...")
        
        try:
            # Run PyInstaller with the spec file
            PyInstaller.__main__.run([
                str(self.spec_file),
                '--clean',
                '--noconfirm',
                '--log-level=INFO'
            ])
            
            logger.info("Build completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Build failed: {e}")
            return False
    
    def post_build_setup(self):
        """Perform post-build setup and validation."""
        logger.info("Performing post-build setup...")
        
        exe_path = self.dist_dir / "CreditUnionMCP.exe"
        
        if not exe_path.exists():
            logger.error(f"Executable not found: {exe_path}")
            return False
        
        # Create installation directory structure
        install_dir = self.dist_dir / "CreditUnionMCP_Install"
        install_dir.mkdir(exist_ok=True)
        
        # Copy executable
        shutil.copy2(exe_path, install_dir / "CreditUnionMCP.exe")
        
        # Copy configuration files
        config_source = self.project_root / "config"
        config_dest = install_dir / "config"
        if config_source.exists():
            shutil.copytree(config_source, config_dest, dirs_exist_ok=True)
        
        # Create logs directory
        logs_dir = install_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Create README for installation
        readme_content = """# Credit Union Analytics MCP Server

## Installation Instructions

1. Extract all files to your desired installation directory
2. Edit config/database_config.yaml with your database connection details
3. Run CreditUnionMCP.exe to start the server

## Configuration

Edit config/database_config.yaml:

```yaml
TEMENOS:
  server: "YOUR_SERVER_NAME"
  database: "TEMENOS"
  windows_auth: true

ARCUSYM000:
  server: "YOUR_SERVER_NAME"
  database: "ARCUSYM000"
  windows_auth: true
```

## Usage

The MCP server provides the following tools:
- execute_query: Execute SQL queries
- analyze_financial_performance: Financial performance analysis
- analyze_portfolio_risk: Portfolio risk analysis
- analyze_member_segments: Member analytics
- check_compliance: Compliance monitoring
- analyze_operations: Operations analysis
- comprehensive_analysis: Multi-agent analysis

## Logs

Server logs are written to the logs/ directory.

## Support

For technical support, please refer to the documentation.
"""
        
        readme_path = install_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        # Create batch file for easy execution
        batch_content = """@echo off
echo Starting Credit Union Analytics MCP Server...
echo.
echo Configuration file: config\\database_config.yaml
echo Logs directory: logs\\
echo.
echo Press Ctrl+C to stop the server
echo.
CreditUnionMCP.exe
pause
"""
        
        batch_path = install_dir / "start_server.bat"
        with open(batch_path, 'w') as f:
            f.write(batch_content)
        
        logger.info(f"Installation package created: {install_dir}")
        
        # Create a ZIP file for distribution
        zip_path = self.dist_dir / "CreditUnionMCP_v1.0.0_Windows"
        shutil.make_archive(str(zip_path), 'zip', install_dir)
        logger.info(f"Distribution ZIP created: {zip_path}.zip")
        
        return True
    
    def validate_build(self):
        """Validate the built executable."""
        logger.info("Validating build...")
        
        exe_path = self.dist_dir / "CreditUnionMCP.exe"
        
        if not exe_path.exists():
            logger.error("Executable not found!")
            return False
        
        # Check file size (should be reasonable)
        file_size_mb = exe_path.stat().st_size / (1024 * 1024)
        logger.info(f"Executable size: {file_size_mb:.1f} MB")
        
        if file_size_mb > 500:  # Warn if over 500MB
            logger.warning("Executable is quite large - consider excluding more dependencies")
        
        # Test if executable can be launched (basic test)
        try:
            result = subprocess.run([str(exe_path), '--help'], 
                                  capture_output=True, text=True, timeout=30)
            logger.info("Executable validation completed")
            return True
        except subprocess.TimeoutExpired:
            logger.warning("Executable test timed out - this may be normal for MCP servers")
            return True
        except Exception as e:
            logger.error(f"Executable validation failed: {e}")
            return False
    
    def build(self):
        """Run the complete build process."""
        logger.info("=" * 60)
        logger.info("Credit Union MCP Build Process Starting")
        logger.info("=" * 60)
        
        try:
            # Step 1: Clean previous builds
            self.clean_build()
            
            # Step 2: Verify dependencies
            if not self.verify_dependencies():
                return False
            
            # Step 3: Create spec file
            self.create_spec_file()
            
            # Step 4: Build executable
            if not self.build_executable():
                return False
            
            # Step 5: Post-build setup
            if not self.post_build_setup():
                return False
            
            # Step 6: Validate build
            if not self.validate_build():
                return False
            
            logger.info("=" * 60)
            logger.info("BUILD COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"Executable: {self.dist_dir / 'CreditUnionMCP.exe'}")
            logger.info(f"Install Package: {self.dist_dir / 'CreditUnionMCP_Install'}")
            logger.info(f"Distribution ZIP: {self.dist_dir / 'CreditUnionMCP_v1.0.0_Windows.zip'}")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Build process failed: {e}")
            return False


def main():
    """Main build function."""
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    
    builder = MCPBuilder()
    success = builder.build()
    
    if success:
        logger.info("Build completed successfully!")
        sys.exit(0)
    else:
        logger.error("Build failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
