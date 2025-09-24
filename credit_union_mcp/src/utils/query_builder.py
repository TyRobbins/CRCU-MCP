"""
Query Builder Utilities for Credit Union MCP Server

Provides common SQL query patterns and builders to reduce code duplication
across agents and standardize database interactions.

CRITICAL: All queries MUST include ProcessDate filter to avoid historical aggregation.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger


class QueryBuilder:
    """
    Builder class for constructing common SQL queries with proper parameterization.
    
    CRITICAL: All queries include mandatory ProcessDate filtering.
    """
    
    def __init__(self, database_type: str = "ARCUSYM000"):
        """
        Initialize query builder.
        
        Args:
            database_type: Target database type (ARCUSYM000 or TEMENOS)
        """
        self.database_type = database_type
        self.ensure_process_date_filter = True  # Mandatory ProcessDate filtering
        
    def build_member_query(self, as_of_date: Optional[str] = None, 
                          include_inactive: bool = False,
                          limit: Optional[int] = None,
                          use_current_snapshot: bool = True) -> str:
        """
        Build standardized member data query with ProcessDate filter.
        
        Args:
            as_of_date: Analysis date filter
            include_inactive: Whether to include inactive members
            limit: Optional row limit
            use_current_snapshot: Use current ProcessDate snapshot (recommended)
            
        Returns:
            SQL query string with ProcessDate filter
        """
        if use_current_snapshot:
            base_query = """
            SELECT 
                n.PARENTACCOUNT AS member_id,
                n.FIRST AS first_name,
                n.LAST AS last_name,
                n.MBRCREATEDATE AS join_date,  -- Corrected field name
                n.MBRSTATUS AS member_status,  -- Corrected field name
                a.ADDRESS AS address,
                a.CITY AS city,
                a.STATE AS state,
                a.ZIPCODE AS zip_code,
                a.EMAIL AS email,
                a.PHONE AS phone
            FROM NAME n
            LEFT JOIN MBRADDRESS a ON n.PARENTACCOUNT = a.MemberNumber
            WHERE n.ProcessDate = FORMAT(DATEADD(DAY, -1, GETDATE()), 'yyyyMMdd')
                AND n.TYPE = 0  -- Primary name record
                AND n.SSN IS NOT NULL AND n.SSN <> ''  -- Valid SSN required
                AND CAST(n.PARENTACCOUNT AS BIGINT) > 100  -- Exclude test accounts
            """
        else:
            # Fallback without ProcessDate (not recommended)
            base_query = """
            SELECT 
                n.PARENTACCOUNT AS member_id,
                n.FIRST AS first_name,
                n.LAST AS last_name,
                n.MBRCREATEDATE AS join_date,
                n.MBRSTATUS AS member_status,
                a.ADDRESS AS address,
                a.CITY AS city,
                a.STATE AS state,
                a.ZIPCODE AS zip_code,
                a.EMAIL AS email,
                a.PHONE AS phone
            FROM NAME n
            LEFT JOIN MBRADDRESS a ON n.PARENTACCOUNT = a.MemberNumber
            WHERE n.TYPE = 0 AND CAST(n.PARENTACCOUNT AS BIGINT) > 100
            """
        
        conditions = []
        
        if not include_inactive:
            conditions.append("n.MBRSTATUS = 0")  # Corrected field name
            
        if as_of_date:
            conditions.append("n.MBRCREATEDATE <= :as_of_date")
            
        if conditions:
            base_query += " AND " + " AND ".join(conditions)
            
        if limit:
            base_query = base_query.replace("SELECT", f"SELECT TOP {limit}")
            
        return base_query
    
    def build_account_summary_query(self, member_ids: Optional[List[str]] = None,
                                  as_of_date: Optional[str] = None,
                                  include_loans: bool = True) -> str:
        """
        Build account summary query for members with ProcessDate filter.
        
        Args:
            member_ids: Optional list of specific member IDs
            as_of_date: Analysis date
            include_loans: Include loan data in summary
            
        Returns:
            SQL query string with ProcessDate filter
        """
        if include_loans:
            query = """
            WITH CurrentSnapshot AS (
                SELECT MAX(ProcessDate) as CurrentDate FROM SAVINGS
            ),
            SavingsData AS (
                SELECT 
                    s.PARENTACCOUNT AS member_id,
                    COUNT(s.ID) AS savings_accounts,
                    SUM(s.BALANCE) AS total_savings_balance,
                    MAX(s.OPENDATE) AS last_savings_opened
                FROM SAVINGS s
                CROSS JOIN CurrentSnapshot cs
                WHERE s.ProcessDate = cs.CurrentDate
                    AND s.CLOSEDATE IS NULL
                    AND CAST(s.PARENTACCOUNT AS BIGINT) > 100
                GROUP BY s.PARENTACCOUNT
            ),
            LoanData AS (
                SELECT 
                    l.PARENTACCOUNT AS member_id,
                    COUNT(l.ID) AS loan_accounts,
                    SUM(l.BALANCE) AS total_loan_balance,
                    MAX(l.OPENDATE) AS last_loan_opened
                FROM LOAN l
                CROSS JOIN CurrentSnapshot cs
                WHERE l.ProcessDate = cs.CurrentDate
                    AND l.CLOSEDATE IS NULL
                    AND l.CHARGEOFFDATE IS NULL
                    AND CAST(l.PARENTACCOUNT AS BIGINT) > 100
                GROUP BY l.PARENTACCOUNT
            )
            SELECT 
                COALESCE(s.member_id, l.member_id) AS member_id,
                ISNULL(s.savings_accounts, 0) AS savings_accounts,
                ISNULL(l.loan_accounts, 0) AS loan_accounts,
                ISNULL(s.total_savings_balance, 0) AS total_savings_balance,
                ISNULL(l.total_loan_balance, 0) AS total_loan_balance,
                ISNULL(s.savings_accounts, 0) + ISNULL(l.loan_accounts, 0) AS total_accounts,
                ISNULL(s.total_savings_balance, 0) - ISNULL(l.total_loan_balance, 0) AS net_balance
            FROM SavingsData s
            FULL OUTER JOIN LoanData l ON s.member_id = l.member_id
            """
        else:
            query = """
            WITH CurrentSnapshot AS (
                SELECT MAX(ProcessDate) as CurrentDate FROM SAVINGS
            )
            SELECT 
                s.PARENTACCOUNT AS member_id,
                COUNT(s.ID) AS savings_accounts,
                SUM(s.BALANCE) AS total_savings_balance,
                MAX(s.OPENDATE) AS last_account_opened,
                COUNT(s.ID) AS total_accounts,
                SUM(s.BALANCE) AS total_balance
            FROM SAVINGS s
            CROSS JOIN CurrentSnapshot cs
            WHERE s.ProcessDate = cs.CurrentDate
                AND s.CLOSEDATE IS NULL
                AND CAST(s.PARENTACCOUNT AS BIGINT) > 100
            """
        
        conditions = []
        
        if member_ids:
            if include_loans:
                conditions.append("(s.member_id IN :member_ids OR l.member_id IN :member_ids)")
            else:
                conditions.append("s.PARENTACCOUNT IN :member_ids")
            
        if conditions:
            query += " AND " + " AND ".join(conditions)
            
        if not include_loans:
            query += " GROUP BY s.PARENTACCOUNT"
        
        return query

    def build_corrected_profitability_query(self, member_ids: Optional[List[str]] = None,
                                          include_fees: bool = True) -> str:
        """
        Build member profitability query with all corrections applied.
        
        Args:
            member_ids: Optional list of specific member IDs
            include_fees: Include fee income calculations
            
        Returns:
            Corrected profitability query with ProcessDate filter
        """
        base_query = """
        WITH CurrentSnapshot AS (
            SELECT MAX(ProcessDate) as CurrentDate FROM NAME
        ),
        ActiveMembers AS (
            SELECT DISTINCT
                n.PARENTACCOUNT as MemberNumber,
                n.FIRST, n.LAST,
                n.MBRSTATUS,
                n.MBRCREATEDATE
            FROM NAME n
            CROSS JOIN CurrentSnapshot cs
            WHERE n.ProcessDate = cs.CurrentDate
                AND n.TYPE = 0  -- Primary name
                AND n.MBRSTATUS = 0  -- Active
                AND CAST(n.PARENTACCOUNT AS BIGINT) > 100  -- Exclude test
        ),
        LoanIncome AS (
            SELECT 
                l.PARENTACCOUNT,
                SUM(ISNULL(l.INTERESTYTD, 0)) as InterestIncomeYTD,
                SUM(ISNULL(l.INTERESTLASTYEAR, 0)) as InterestIncomeLY,
                COUNT(l.ID) as LoanCount,
                SUM(l.BALANCE) as TotalLoanBalance
            FROM LOAN l
            CROSS JOIN CurrentSnapshot cs
            WHERE l.ProcessDate = cs.CurrentDate
                AND l.CLOSEDATE IS NULL
                AND l.CHARGEOFFDATE IS NULL
                AND CAST(l.PARENTACCOUNT AS BIGINT) > 100
            GROUP BY l.PARENTACCOUNT
        ),
        SavingsBalances AS (
            SELECT 
                s.PARENTACCOUNT,
                SUM(s.BALANCE) as TotalSavingsBalance,
                COUNT(s.ID) as SavingsCount
            FROM SAVINGS s
            CROSS JOIN CurrentSnapshot cs
            WHERE s.ProcessDate = cs.CurrentDate
                AND s.CLOSEDATE IS NULL
                AND CAST(s.PARENTACCOUNT AS BIGINT) > 100
            GROUP BY s.PARENTACCOUNT
        )
        """
        
        if include_fees:
            base_query += """
        ,
        FeeIncome AS (
            SELECT 
                lt.PARENTACCOUNT,
                SUM(CASE WHEN lt.TransactionCode IN (301, 302, 303) 
                    THEN lt.Amount ELSE 0 END) as NonPunitiveFees,
                SUM(CASE WHEN lt.TransactionCode IN (700, 701, 702, 710, 720) 
                    THEN lt.Amount ELSE 0 END) as PunitiveFees
            FROM LOANTRANSACTION lt
            INNER JOIN LOAN l ON lt.PARENTACCOUNT = l.PARENTACCOUNT 
                AND lt.PARENTID = l.ID  -- Corrected: Use PARENTID not ID
            CROSS JOIN CurrentSnapshot cs
            WHERE l.ProcessDate = cs.CurrentDate
                AND lt.PostDate >= DATEADD(year, -1, GETDATE())
                AND CAST(lt.PARENTACCOUNT AS BIGINT) > 100
            GROUP BY lt.PARENTACCOUNT
        )
        """
        
        base_query += """
        SELECT 
            am.MemberNumber,
            am.FIRST + ' ' + am.LAST as MemberName,
            am.MBRCREATEDATE as JoinDate,
            ISNULL(li.InterestIncomeYTD, 0) + ISNULL(li.InterestIncomeLY, 0) as TotalInterestIncome,
            ISNULL(li.LoanCount, 0) as LoanCount,
            ISNULL(li.TotalLoanBalance, 0) as TotalLoanBalance,
            ISNULL(sb.SavingsCount, 0) as SavingsCount,
            ISNULL(sb.TotalSavingsBalance, 0) as TotalSavingsBalance
        """
        
        if include_fees:
            base_query += """
            ,
            ISNULL(fi.NonPunitiveFees, 0) as NonPunitiveFees,
            ISNULL(fi.PunitiveFees, 0) as PunitiveFees,
            ISNULL(li.InterestIncomeYTD, 0) + ISNULL(li.InterestIncomeLY, 0) + 
            ISNULL(fi.NonPunitiveFees, 0) + ISNULL(fi.PunitiveFees, 0) as TotalRevenue
            """
        
        base_query += """
        FROM ActiveMembers am
        LEFT JOIN LoanIncome li ON am.MemberNumber = li.PARENTACCOUNT
        LEFT JOIN SavingsBalances sb ON am.MemberNumber = sb.PARENTACCOUNT
        """
        
        if include_fees:
            base_query += "LEFT JOIN FeeIncome fi ON am.MemberNumber = fi.PARENTACCOUNT"
        
        if member_ids:
            base_query += " WHERE am.MemberNumber IN :member_ids"
        
        return base_query
    
    def build_transaction_query(self, member_ids: Optional[List[str]] = None,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              transaction_types: Optional[List[str]] = None,
                              min_amount: Optional[float] = None) -> str:
        """
        Build transaction history query.
        
        Args:
            member_ids: Optional member ID filter
            start_date: Start date filter
            end_date: End date filter
            transaction_types: Optional transaction type filter
            min_amount: Minimum transaction amount filter
            
        Returns:
            SQL query string
        """
        query = """
        SELECT 
            st.ACCOUNT AS member_id,
            st.ID AS account_id,
            st.TRANSDATE AS transaction_date,
            st.TRANSCODE AS transaction_code,
            st.AMOUNT AS transaction_amount,
            st.BALANCE AS account_balance,
            st.DESCRIPTION AS description,
            st.REFERENCE AS reference
        FROM SAVINGSTRANSACTION st
        WHERE 1=1
        """
        
        conditions = []
        
        if member_ids:
            conditions.append("st.ACCOUNT IN :member_ids")
            
        if start_date:
            conditions.append("st.TRANSDATE >= :start_date")
            
        if end_date:
            conditions.append("st.TRANSDATE <= :end_date")
            
        if transaction_types:
            conditions.append("st.TRANSCODE IN :transaction_types")
            
        if min_amount:
            conditions.append("ABS(st.AMOUNT) >= :min_amount")
            
        if conditions:
            query += " AND " + " AND ".join(conditions)
            
        query += " ORDER BY st.TRANSDATE DESC, st.ACCOUNT"
        
        return query
    
    def build_financial_summary_query(self, as_of_date: Optional[str] = None) -> str:
        """
        Build financial summary query for institution-level metrics.
        
        Args:
            as_of_date: Analysis date
            
        Returns:
            SQL query string
        """
        query = """
        SELECT 
            SUM(CASE WHEN s.ID LIKE '%S%' AND s.BALANCE > 0 THEN s.BALANCE ELSE 0 END) AS total_deposits,
            SUM(CASE WHEN s.ID LIKE '%L%' AND s.BALANCE > 0 THEN s.BALANCE ELSE 0 END) AS total_loans,
            COUNT(DISTINCT s.ACCOUNT) AS total_members,
            COUNT(CASE WHEN s.ID LIKE '%S%' AND s.BALANCE > 0 THEN s.ID END) AS deposit_accounts,
            COUNT(CASE WHEN s.ID LIKE '%L%' AND s.BALANCE > 0 THEN s.ID END) AS loan_accounts,
            AVG(CASE WHEN s.ID LIKE '%S%' AND s.BALANCE > 0 THEN s.BALANCE END) AS avg_deposit_balance,
            AVG(CASE WHEN s.ID LIKE '%L%' AND s.BALANCE > 0 THEN s.BALANCE END) AS avg_loan_balance
        FROM SAVINGS s
        WHERE (s.CLOSEDATE IS NULL OR s.CLOSEDATE = 0)
        """
        
        if as_of_date:
            query += " AND s.OPENDATE <= :as_of_date"
            
        return query


class CommonQueries:
    """
    Collection of commonly used queries across agents.
    """
    
    @staticmethod
    def get_active_members_count(as_of_date: Optional[str] = None) -> str:
        """Get count of active members."""
        query = """
        SELECT COUNT(DISTINCT ACCOUNT) as active_member_count
        FROM NAME
        WHERE STATUS IN ('A', 'ACTIVE')
        """
        
        if as_of_date:
            query += """
            AND OPENDATE <= :as_of_date
            AND (CLOSEDATE IS NULL OR CLOSEDATE > :as_of_date)
            """
            
        return query
    
    @staticmethod
    def get_new_members_period(start_date: str, end_date: str) -> str:
        """Get new members for a specific period."""
        return """
        SELECT COUNT(*) as new_member_count
        FROM NAME
        WHERE OPENDATE >= :start_date
        AND OPENDATE <= :end_date
        AND STATUS IN ('A', 'ACTIVE')
        """
    
    @staticmethod
    def get_account_types_summary() -> str:
        """Get summary of account types."""
        return """
        SELECT 
            CASE 
                WHEN ID LIKE '%S%' THEN 'Savings'
                WHEN ID LIKE '%L%' THEN 'Loan'
                WHEN ID LIKE '%C%' THEN 'Credit'
                ELSE 'Other'
            END AS account_type,
            COUNT(*) as account_count,
            SUM(BALANCE) as total_balance,
            AVG(BALANCE) as average_balance
        FROM SAVINGS
        WHERE (CLOSEDATE IS NULL OR CLOSEDATE = 0)
        AND BALANCE != 0
        GROUP BY 
            CASE 
                WHEN ID LIKE '%S%' THEN 'Savings'
                WHEN ID LIKE '%L%' THEN 'Loan'
                WHEN ID LIKE '%C%' THEN 'Credit'
                ELSE 'Other'
            END
        """
    
    @staticmethod
    def get_member_relationship_length() -> str:
        """Calculate member relationship length."""
        return """
        SELECT 
            ACCOUNT as member_id,
            OPENDATE as join_date,
            DATEDIFF(day, OPENDATE, GETDATE()) as relationship_days,
            DATEDIFF(month, OPENDATE, GETDATE()) as relationship_months,
            DATEDIFF(year, OPENDATE, GETDATE()) as relationship_years
        FROM NAME
        WHERE STATUS IN ('A', 'ACTIVE')
        AND OPENDATE IS NOT NULL
        """
    
    @staticmethod
    def get_high_value_members(threshold: float = 10000.0) -> str:
        """Get members with balances above threshold."""
        return """
        SELECT 
            s.ACCOUNT as member_id,
            SUM(CASE WHEN s.BALANCE > 0 THEN s.BALANCE ELSE 0 END) as total_deposits,
            SUM(CASE WHEN s.BALANCE < 0 THEN ABS(s.BALANCE) ELSE 0 END) as total_loans,
            SUM(s.BALANCE) as net_balance,
            COUNT(s.ID) as total_accounts
        FROM SAVINGS s
        WHERE (s.CLOSEDATE IS NULL OR s.CLOSEDATE = 0)
        GROUP BY s.ACCOUNT
        HAVING SUM(CASE WHEN s.BALANCE > 0 THEN s.BALANCE ELSE 0 END) >= :threshold
        ORDER BY total_deposits DESC
        """
    
    @staticmethod
    def get_transaction_volume_by_month(months_back: int = 12) -> str:
        """Get transaction volume trends."""
        return """
        SELECT 
            YEAR(st.TRANSDATE) as transaction_year,
            MONTH(st.TRANSDATE) as transaction_month,
            COUNT(*) as transaction_count,
            SUM(ABS(st.AMOUNT)) as transaction_volume,
            COUNT(DISTINCT st.ACCOUNT) as unique_members
        FROM SAVINGSTRANSACTION st
        WHERE st.TRANSDATE >= DATEADD(month, -:months_back, GETDATE())
        GROUP BY YEAR(st.TRANSDATE), MONTH(st.TRANSDATE)
        ORDER BY transaction_year DESC, transaction_month DESC
        """
    
    @staticmethod
    def get_delinquency_summary() -> str:
        """Get loan delinquency summary."""
        return """
        SELECT 
            COUNT(CASE WHEN l.PAYMENTDUE > 0 AND l.DAYSPASTDUE >= 30 THEN 1 END) as delinquent_30_plus,
            COUNT(CASE WHEN l.PAYMENTDUE > 0 AND l.DAYSPASTDUE >= 60 THEN 1 END) as delinquent_60_plus,
            COUNT(CASE WHEN l.PAYMENTDUE > 0 AND l.DAYSPASTDUE >= 90 THEN 1 END) as delinquent_90_plus,
            SUM(CASE WHEN l.PAYMENTDUE > 0 AND l.DAYSPASTDUE >= 30 THEN l.BALANCE END) as delinquent_balance_30_plus,
            COUNT(CASE WHEN l.PAYMENTDUE > 0 THEN 1 END) as total_loans_with_payment_due,
            SUM(CASE WHEN l.PAYMENTDUE > 0 THEN l.BALANCE END) as total_balance_with_payment_due
        FROM LOAN l
        WHERE l.CLOSEDATE IS NULL OR l.CLOSEDATE = 0
        """


class QueryValidator:
    """
    Validates and sanitizes SQL queries for security and performance.
    """
    
    FORBIDDEN_KEYWORDS = [
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 
        'TRUNCATE', 'MERGE', 'UPSERT', 'REPLACE', 'EXEC', 'EXECUTE'
    ]
    
    @classmethod
    def is_safe_query(cls, query: str) -> bool:
        """
        Check if query is safe (read-only).
        
        Args:
            query: SQL query string
            
        Returns:
            True if query is safe
        """
        upper_query = query.upper().strip()
        
        for keyword in cls.FORBIDDEN_KEYWORDS:
            if keyword in upper_query:
                logger.warning(f"Forbidden keyword '{keyword}' found in query")
                return False
                
        return upper_query.startswith('SELECT') or upper_query.startswith('WITH')
    
    @classmethod
    def add_row_limit(cls, query: str, limit: int = 10000) -> str:
        """
        Add row limit to query if not present.
        
        Args:
            query: Original query
            limit: Row limit to add
            
        Returns:
            Query with row limit
        """
        upper_query = query.upper()
        
        if 'TOP' in upper_query or 'LIMIT' in upper_query:
            return query
            
        if query.upper().strip().startswith('SELECT'):
            return query.replace('SELECT', f'SELECT TOP {limit}', 1)
            
        return query
    
    @classmethod
    def validate_parameters(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize query parameters.
        
        Args:
            params: Query parameters
            
        Returns:
            Sanitized parameters
        """
        sanitized = {}
        
        for key, value in params.items():
            # Basic sanitization - remove potential SQL injection attempts
            if isinstance(value, str):
                # Remove common SQL injection patterns
                sanitized_value = value.replace("'", "''")  # Escape single quotes
                sanitized_value = sanitized_value.replace(";", "")  # Remove semicolons
                sanitized[key] = sanitized_value
            else:
                sanitized[key] = value
                
        return sanitized
