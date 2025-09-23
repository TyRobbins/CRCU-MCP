# Credit Union Analytics MCP Server

Multi-Agentic Credit Union Analytics MCP Server providing comprehensive financial analysis capabilities through specialized AI agents.

## 🚀 Quick Start

### Prerequisites
- Windows 10/11
- Python 3.11 or higher
- Access to TEMENOS and ARCUSYM000 databases

### 1. Environment Setup

**Option A: Automatic Setup (Recommended)**
```bash
python setup_environment.py
```
This will:
- Check if Python is in your system PATH
- Add Python to PATH if needed
- Create convenient batch scripts
- Verify all dependencies are installed

**Option B: Manual Setup**

1. **Add Python to Windows PATH:**
   - Open Windows Settings → System → About → Advanced System Settings
   - Click "Environment Variables"
   - In System Variables, find and select "Path" → Click "Edit"
   - Click "New" and add your Python installation directory (e.g., `C:\Python311\`)
   - Click "New" and add your Python Scripts directory (e.g., `C:\Python311\Scripts\`)
   - Click "OK" to save changes
   - Restart Command Prompt

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### 2. Database Configuration

The database configuration is already set up for production:

**config/database_config.yaml:**
```yaml
TEMENOS:
  server: "decsql4"
  port: 1433
  database: "Temenos"
  windows_auth: false
  username: "svctrcacct"
  password: "e;Co_Bb2MD_Pbvh%f"
  
ARCUSYM000:
  server: "prodarcu"
  port: 1433
  database: "ARCUSYM000"
  windows_auth: false
  username: "svctrcacct"
  password: "e;Co_Bb2MD_Pbvh%f"
```

### 3. Running the Server

**Option A: Using Batch Scripts (after running setup)**
- Double-click `start_mcp_server.bat`

**Option B: Command Line**
```bash
python -m src.main
```

### 4. Building Standalone Executable

**Option A: Using Batch Script**
- Double-click `build_mcp.bat`

**Option B: Command Line**
```bash
python build.py
```

This creates a standalone Windows executable in the `dist/` folder.

## 🎯 Features

### 5 Specialized AI Agents

#### 1. Financial Performance Agent
- **Capabilities:** ROA, ROE, NIM calculations, NCUA 5300 metrics, profitability analysis
- **Tool:** `analyze_financial_performance`
- **Parameters:** 
  - `metric_type`: profitability, capital, asset_quality, comprehensive
  - `database`: TEMENOS, ARCUSYM000
  - `as_of_date`: YYYY-MM-DD format

#### 2. Portfolio Risk Agent
- **Capabilities:** HHI concentration, delinquency forecasting, Monte Carlo stress testing
- **Tool:** `analyze_portfolio_risk`
- **Parameters:**
  - `analysis_type`: concentration, delinquency, stress_test, comprehensive
  - `database`: TEMENOS, ARCUSYM000
  - `as_of_date`: YYYY-MM-DD format

#### 3. Member Analytics Agent
- **Capabilities:** RFM segmentation, K-means clustering, lifetime value, churn prediction
- **Tool:** `analyze_member_segments`
- **Parameters:**
  - `method`: rfm, clustering, lifetime_value, churn_prediction, comprehensive
  - `database`: TEMENOS, ARCUSYM000
  - `as_of_date`: YYYY-MM-DD format

#### 4. Compliance Agent
- **Capabilities:** BSA/AML monitoring, capital adequacy, CECL calculations, regulatory limits
- **Tool:** `check_compliance`
- **Parameters:**
  - `check_type`: capital, bsa_aml, lending_limits, cecl, all
  - `database`: TEMENOS, ARCUSYM000
  - `as_of_date`: YYYY-MM-DD format

#### 5. Operations Agent
- **Capabilities:** Branch performance, channel efficiency, staff productivity, cost analysis
- **Tool:** `analyze_operations`
- **Parameters:**
  - `focus_area`: branch, channel, staff, all
  - `database`: TEMENOS, ARCUSYM000
  - `as_of_date`: YYYY-MM-DD format

### Database Tools

#### Execute SQL Queries
- **Tool:** `execute_query`
- **Parameters:**
  - `query`: SQL SELECT statement
  - `database`: TEMENOS, ARCUSYM000
  - `max_rows`: Maximum rows to return (default: 10000)

#### Database Schema Information
- **Tool:** `get_tables` - List all tables
- **Tool:** `get_table_schema` - Get table structure
- **Parameters:**
  - `database`: TEMENOS, ARCUSYM000
  - `schema`: Database schema (default: dbo)

### Multi-Agent Analysis

#### Comprehensive Analysis
- **Tool:** `comprehensive_analysis`
- **Description:** Runs all agents in parallel for complete credit union analysis
- **Parameters:**
  - `database`: TEMENOS, ARCUSYM000
  - `as_of_date`: YYYY-MM-DD format
  - `agents`: Optional list of specific agents to run

### Utility Tools

- **Tool:** `test_connection` - Test database connectivity
- **Tool:** `health_check` - System health status
- **Tool:** `get_agent_capabilities` - List all agent capabilities

## 🔧 Technical Architecture

### Project Structure
```
credit_union_mcp/
├── src/
│   ├── main.py                 # MCP Server entry point
│   ├── database/
│   │   ├── connection.py       # Database connection management
│   │   └── query_builder.py    # SQL query utilities
│   ├── agents/
│   │   ├── base_agent.py       # Abstract base agent
│   │   ├── financial_performance.py
│   │   ├── portfolio_risk.py
│   │   ├── member_analytics.py
│   │   ├── compliance.py
│   │   └── operations.py
│   └── orchestration/
│       ├── coordinator.py      # Agent coordination
│       └── classifier.py       # Request routing
├── config/
│   └── database_config.yaml    # Database configuration
├── logs/                       # Application logs
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
├── build.py                   # Executable builder
├── setup_environment.py       # Environment setup
└── README.md                  # This file
```

### Key Components

#### Database Layer
- **Connection Pooling:** SQLAlchemy with configurable pool settings
- **Security:** Read-only query enforcement, SQL injection prevention
- **Authentication:** SQL Server authentication with encrypted passwords
- **Multi-Database:** Simultaneous connections to TEMENOS and ARCUSYM000

#### Agent Framework
- **Base Agent:** Standardized interface for all specialized agents
- **Async Support:** Non-blocking parallel execution
- **Caching:** Intelligent result caching for performance
- **Error Handling:** Comprehensive error recovery and logging

#### Orchestration System
- **Intelligent Routing:** Keyword-based request classification
- **Parallel Execution:** Multiple agents run simultaneously
- **Result Aggregation:** Cross-agent insights and recommendations
- **Health Monitoring:** Real-time system status tracking

## 📊 Analytics Capabilities

### Financial Metrics
- Return on Assets (ROA)
- Return on Equity (ROE)
- Net Interest Margin (NIM)
- Efficiency Ratio
- Capital Adequacy Ratios
- NCUA Call Report Metrics

### Risk Analytics
- Portfolio Concentration (HHI Index)
- Delinquency Trending & Forecasting
- Credit Loss Projections
- Monte Carlo Stress Testing
- Vintage Analysis
- Migration Matrices

### Member Intelligence
- RFM Segmentation (Recency, Frequency, Monetary)
- Customer Lifetime Value (CLV)
- Churn Prediction Models
- Cross-Sell Propensity
- Behavioral Clustering
- Demographic Analysis

### Compliance Monitoring
- BSA/AML Anomaly Detection
- Suspicious Activity Monitoring
- Capital Adequacy Assessment
- CECL Calculations
- Member Business Loan Limits
- Regulatory Ratio Monitoring

### Operational Analytics
- Branch Performance Ranking
- Channel Efficiency Analysis
- Staff Productivity Metrics
- Cost Per Member Calculations
- Digital Adoption Rates
- Process Optimization Insights

## 🛡️ Security Features

- **Read-Only Database Access:** Only SELECT queries permitted
- **SQL Injection Prevention:** Parameterized queries and validation
- **Connection Security:** Encrypted database connections
- **Audit Logging:** Comprehensive activity logging
- **Error Sanitization:** Sensitive information protection

## 📝 Deployment Options

### Option 1: Python Environment
1. Install Python 3.11+
2. Run `setup_environment.py`
3. Use `start_mcp_server.bat`

### Option 2: Standalone Executable
1. Run `build_mcp.bat` or `python build.py`
2. Deploy `dist/CreditUnionMCP_v1.0.0_Windows.zip`
3. Extract and run `CreditUnionMCP.exe`

### Option 3: Claude Desktop Integration
Add to Claude Desktop configuration:
```json
{
  "mcpServers": {
    "credit-union-analytics": {
      "command": "python",
      "args": ["-m", "src.main"],
      "cwd": "C:/path/to/credit_union_mcp"
    }
  }
}
```

## 🔍 Troubleshooting

### Common Issues

**Python not found in PATH**
- Run `setup_environment.py` as administrator
- Or manually add Python to Windows PATH
- Restart command prompt after changes

**Database connection failed**
- Verify server names and credentials in `config/database_config.yaml`
- Check network connectivity to database servers
- Ensure SQL Server allows remote connections

**Import errors**
- Run `install_dependencies.bat`
- Or `pip install -r requirements.txt`
- Check Python version compatibility (3.11+ required)

**Build errors**
- Ensure all dependencies are installed
- Run `python build.py` from project root
- Check disk space for large executable

### Logs
- Application logs: `logs/credit_union_mcp_{timestamp}.log`
- Build logs: Console output during build process
- Database logs: Included in application logs

## 📞 Support

For technical support or questions about the Credit Union Analytics MCP Server:

1. Check the logs in the `logs/` directory
2. Verify database connectivity with `test_connection` tool
3. Run `health_check` tool for system status
4. Review this documentation for configuration steps

## 🔄 Updates

### Version 1.0.0
- Initial release with 5 specialized agents
- Production database configuration
- Standalone executable support
- Comprehensive analytics capabilities
- Multi-agent orchestration
- Windows environment setup automation

---

*Credit Union Analytics MCP Server - Enterprise-Grade Financial Analytics*
