"""
Credit Union Data Catalog

Comprehensive mapping of critical data locations across ARCUSYM000 and TEMENOS databases.
This catalog serves as the definitive guide for data location and routing decisions.
"""

from typing import Dict, List, Any
from enum import Enum

class DatabaseRole(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    SPECIALIZED = "specialized"

class DataCatalog:
    """
    Central catalog mapping all critical credit union data locations.
    
    ARCUSYM000: Primary operational core banking system
    TEMENOS: Specialized collections, origination, and workflow management
    """
    
    DATABASE_ROLES = {
        "ARCUSYM000": DatabaseRole.PRIMARY,
        "TEMENOS": DatabaseRole.SPECIALIZED
    }
    
    # Core Member Data (ARCUSYM000 Primary)
    MEMBER_DATA = {
        "primary_database": "ARCUSYM000",
        "tables": {
            "NAME": {
                "description": "Primary name records for all members",
                "key_fields": ["SSN", "LAST", "FIRST", "MIDDLE", "MemberNumber", "PARENTACCOUNT", "TYPE"],
                "business_rules": ["TYPE=0 for primary name record", "SSN is PII - requires protection"],
                "usage": "Member identification, PII management, account relationships"
            },
            "MEMBERREC": {
                "description": "Core member demographic and status information", 
                "key_fields": ["SSN", "MemberNumber", "BirthDate", "JoinDate", "MemberStatus"],
                "business_rules": ["One record per unique member", "Status determines membership validity"],
                "usage": "Member demographics, status tracking, NCUA reporting"
            },
            "MBRADDRESS": {
                "description": "Member addresses with type classifications",
                "key_fields": ["MemberNumber", "AddressType", "Street", "City", "State", "ZipCode"],
                "business_rules": ["Multiple addresses per member allowed", "Type 0 = Primary address"],
                "usage": "Mailing, correspondence, regulatory reporting"
            },
            "ACCOUNT": {
                "description": "Master account records - foundation of all relationships",
                "key_fields": ["ACCOUNTNUMBER", "TYPE", "OPENDATE", "CLOSEDATE", "BRANCH", "MEMBERSTATUS"],
                "business_rules": ["Account types 0-15, 87-99 are valid", "CLOSEDATE = '19000101' means open"],
                "usage": "Account management, member relationships, product categorization",
                "schema_notes": "100+ fields including warning codes, payment history, NSF tracking"
            }
        }
    }
    
    # Deposit/Savings Data (ARCUSYM000 Primary)
    DEPOSIT_DATA = {
        "primary_database": "ARCUSYM000", 
        "tables": {
            "SAVINGS": {
                "description": "All deposit/share accounts (checking, savings, CDs, IRAs)",
                "key_fields": ["PARENTACCOUNT", "ID", "TYPE", "BALANCE", "OPENDATE", "CLOSEDATE"],
                "business_rules": ["CLOSEDATE = '19000101' means active", "Balance tracking by period"],
                "usage": "Deposit balances, interest calculation, maturity tracking"
            },
            "SAVINGSTRANSACTION": {
                "description": "All deposit account transactions",
                "key_fields": ["PARENTACCOUNT", "ID", "TransactionCode", "Amount", "PostDate"],
                "business_rules": ["Real-time transaction posting", "Audit trail maintained"],
                "usage": "Transaction history, account reconciliation, member statements"
            },
            "SAVINGSHOLD": {
                "description": "Holds and restrictions on deposit accounts",
                "key_fields": ["PARENTACCOUNT", "ID", "HoldType", "Amount", "ExpirationDate"],
                "business_rules": ["Multiple holds per account allowed", "Affects available balance"],
                "usage": "Fund availability, compliance holds, collateral management"
            }
        }
    }
    
    # Loan Data (ARCUSYM000 Primary with TEMENOS for Collections)
    LOAN_DATA = {
        "primary_database": "ARCUSYM000",
        "secondary_database": "TEMENOS", 
        "tables": {
            "ARCUSYM000": {
                "LOAN": {
                    "description": "Comprehensive loan records - 500+ fields covering all loan aspects",
                    "key_fields": ["PARENTACCOUNT", "ID", "TYPE", "BALANCE", "ORIGINALBALANCE", "DUEDATE", "CHARGEOFFDATE"],
                    "business_rules": ["CHARGEOFFDATE = '19000101' means not charged off", "CLOSEDATE = '19000101' means active"],
                    "usage": "Loan balances, payment tracking, interest accrual, regulatory reporting",
                    "schema_notes": "Includes payment buckets, credit card tracking, ARM management, escrow"
                },
                "LOANTRANSACTION": {
                    "description": "All loan transaction history",
                    "key_fields": ["PARENTACCOUNT", "ID", "TransactionCode", "Amount", "PostDate"],
                    "business_rules": ["Complete audit trail", "Transaction codes define purpose"],
                    "usage": "Payment history, transaction analysis, member statements"
                },
                "LOANHOLD": {
                    "description": "Loan holds and restrictions",
                    "key_fields": ["PARENTACCOUNT", "ID", "HoldType", "Amount"],
                    "usage": "Payment restrictions, compliance holds"
                }
            },
            "TEMENOS": {
                "tblAccount": {
                    "description": "Collections and delinquency management for loans",
                    "key_fields": ["AccountNumber", "DelinquencyStatus", "CollectionStage"],
                    "business_rules": ["Focus on delinquent accounts", "Workflow-driven"],
                    "usage": "Collections processing, delinquency tracking, workout management"
                },
                "tblAccountChargeOff": {
                    "description": "Charge-off processing and management",
                    "key_fields": ["AccountNumber", "ChargeOffDate", "ChargeOffAmount", "RecoveryAmount"],
                    "usage": "Charge-off processing, recovery tracking, loss mitigation"
                }
            }
        },
        "routing_rules": {
            "current_loans": "ARCUSYM000",
            "delinquent_loans": "Both - ARCUSYM000 for data, TEMENOS for workflow",
            "charged_off_loans": "Both - ARCUSYM000 for records, TEMENOS for collections"
        }
    }
    
    # Card Data (ARCUSYM000 Primary)
    CARD_DATA = {
        "primary_database": "ARCUSYM000",
        "tables": {
            "CARD": {
                "description": "Debit and credit card records",
                "key_fields": ["PARENTACCOUNT", "ID", "CardNumber", "CardType", "Status", "ExpirationDate"],
                "business_rules": ["Card numbers are PII", "Status determines usability"],
                "usage": "Card management, transaction authorization, fraud monitoring"
            },
            "CARDACCESS": {
                "description": "Card access controls and permissions",
                "key_fields": ["PARENTACCOUNT", "ID", "AccessType", "DailyLimit", "MonthlyLimit"],
                "usage": "Transaction limits, access control, fraud prevention"
            },
            "CARDNAME": {
                "description": "Names authorized on cards", 
                "key_fields": ["PARENTACCOUNT", "ID", "NameType", "FirstName", "LastName"],
                "business_rules": ["Multiple authorized users allowed"],
                "usage": "Authorized user management, card embossing"
            }
        }
    }
    
    # Transaction Data (ARCUSYM000 Primary)
    TRANSACTION_DATA = {
        "primary_database": "ARCUSYM000",
        "tables": {
            "ACTIVITY": {
                "description": "Real-time transaction activity across all products",
                "key_fields": ["PARENTACCOUNT", "TransactionCode", "Amount", "PostDate", "Description"],
                "business_rules": ["Real-time posting", "Comprehensive audit trail"],
                "usage": "Member statements, transaction analysis, fraud monitoring",
                "performance_notes": "High-volume table - use date filters"
            },
            "EFT": {
                "description": "Electronic funds transfer records",
                "key_fields": ["PARENTACCOUNT", "EFTType", "Amount", "EffectiveDate", "Status"],
                "business_rules": ["ACH, wire, and electronic transfers", "Status tracking required"],
                "usage": "Electronic payment processing, ACH management"
            },
            "EFTTRANSFER": {
                "description": "Inter-account transfer records",
                "key_fields": ["FromAccount", "ToAccount", "Amount", "TransferDate", "TransferType"],
                "usage": "Member transfers, internal fund movements"
            }
        }
    }
    
    # Account Origination (TEMENOS Primary)
    ORIGINATION_DATA = {
        "primary_database": "TEMENOS",
        "tables": {
            "tblAccountWorkflow": {
                "description": "Account opening and origination workflows",
                "key_fields": ["AccountNumber", "WorkflowType", "Status", "CreateDate"],
                "business_rules": ["Workflow-driven processes", "Status determines stage"],
                "usage": "New account processing, approval workflows"
            },
            "tblAccountWorkflowStep": {
                "description": "Individual workflow steps and approvals",
                "key_fields": ["WorkflowID", "StepType", "Status", "CompletedDate", "UserID"],
                "usage": "Process tracking, approval audit trail"
            },
            "tblCreditBureau": {
                "description": "Credit bureau integration for loan origination",
                "key_fields": ["AccountNumber", "BureauType", "Score", "ReportDate"],
                "usage": "Credit decisions, risk assessment"
            }
        }
    }
    
    # Collections & Delinquency (TEMENOS Primary)
    COLLECTIONS_DATA = {
        "primary_database": "TEMENOS",
        "tables": {
            "tblQueue": {
                "description": "Collections queues and work management",
                "key_fields": ["AccountNumber", "QueueType", "Priority", "AssignedUser"],
                "usage": "Collections workflow, task management"
            },
            "tblPromise": {
                "description": "Payment promises and arrangements",
                "key_fields": ["AccountNumber", "PromiseDate", "PromiseAmount", "Status"],
                "usage": "Payment arrangements, broken promise tracking"
            },
            "tblAccountBankruptcy": {
                "description": "Bankruptcy case management",
                "key_fields": ["AccountNumber", "Chapter", "FileDate", "DischargeDate"],
                "usage": "Bankruptcy processing, legal compliance"
            }
        }
    }
    
    # Third-Party Integration Data
    THIRD_PARTY_DATA = {
        "locations": {
            "ARCUSYM000": [
                "ATMDIALOG - ATM transaction processing",
                "BATCHACHORIG - ACH origination",
                "CHECKS - Check processing and images",
                "CREDREP - Credit bureau reporting",
                "IRS - Tax reporting (1099-INT, etc.)"
            ],
            "TEMENOS": [
                "tblCreditBureau* - Credit bureau integration",
                "tblDialer* - Automated calling systems", 
                "tblSoaWebServicesLog - Web service integrations",
                "tblConnectorNACHA* - NACHA file processing"
            ]
        }
    }
    
    # Data Routing Rules
    ROUTING_RULES = {
        "member_analytics": {
            "primary": "ARCUSYM000",
            "tables": ["NAME", "MEMBERREC", "ACCOUNT", "SAVINGS", "LOAN"],
            "reason": "Complete member relationship data"
        },
        "deposit_operations": {
            "primary": "ARCUSYM000", 
            "tables": ["SAVINGS", "SAVINGSTRANSACTION", "ACTIVITY"],
            "reason": "Real-time deposit data and transactions"
        },
        "loan_servicing": {
            "primary": "ARCUSYM000",
            "tables": ["LOAN", "LOANTRANSACTION", "ACTIVITY"],
            "reason": "Complete loan portfolio data"
        },
        "collections": {
            "primary": "TEMENOS",
            "secondary": "ARCUSYM000",
            "reason": "TEMENOS for workflow, ARCUSYM000 for account data"
        },
        "account_origination": {
            "primary": "TEMENOS",
            "tables": ["tblAccountWorkflow*", "tblCreditBureau*"],
            "reason": "Workflow and approval processes"
        },
        "delinquency_management": {
            "primary": "TEMENOS",
            "secondary": "ARCUSYM000", 
            "reason": "TEMENOS for collections workflow, ARCUSYM000 for account status"
        },
        "card_management": {
            "primary": "ARCUSYM000",
            "tables": ["CARD", "CARDACCESS", "CARDNAME"],
            "reason": "Complete card management system"
        },
        "transaction_analysis": {
            "primary": "ARCUSYM000",
            "tables": ["ACTIVITY", "SAVINGSTRANSACTION", "LOANTRANSACTION"],
            "reason": "Complete transaction history"
        }
    }
    
    @classmethod
    def get_primary_database_for_analysis(cls, analysis_type: str) -> str:
        """Get the primary database for a specific type of analysis."""
        routing = cls.ROUTING_RULES.get(analysis_type, {})
        return routing.get("primary", "ARCUSYM000")  # Default to ARCUSYM000
    
    @classmethod
    def get_tables_for_analysis(cls, analysis_type: str) -> List[str]:
        """Get the relevant tables for a specific analysis type."""
        routing = cls.ROUTING_RULES.get(analysis_type, {})
        return routing.get("tables", [])
    
    @classmethod
    def should_use_secondary_database(cls, analysis_type: str) -> bool:
        """Check if secondary database should be used for analysis."""
        routing = cls.ROUTING_RULES.get(analysis_type, {})
        return "secondary" in routing
    
    @classmethod
    def get_data_location_summary(cls) -> Dict[str, Any]:
        """Get a summary of data locations for quick reference."""
        return {
            "databases": {
                "ARCUSYM000": {
                    "role": "Primary operational core banking system",
                    "primary_for": [
                        "Member data and relationships",
                        "Deposit/savings accounts", 
                        "Loan portfolio (current and historical)",
                        "Card management",
                        "Transaction processing",
                        "Member analytics and reporting"
                    ],
                    "key_tables": ["ACCOUNT", "LOAN", "SAVINGS", "NAME", "MEMBERREC", "CARD", "ACTIVITY"]
                },
                "TEMENOS": {
                    "role": "Specialized collections, origination, and workflow management",
                    "primary_for": [
                        "Account origination workflows",
                        "Collections and delinquency management",
                        "Charge-off processing",
                        "Credit bureau integration",
                        "Workflow management"
                    ],
                    "key_tables": ["tblAccount", "tblAccountWorkflow*", "tblAccountChargeOff", "tblQueue", "tblCreditBureau*"]
                }
            },
            "routing_strategy": "ARCUSYM000 primary for operational data, TEMENOS for specialized workflows"
        }
    
    @classmethod
    def get_analysis_plan(cls, analysis_type: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get comprehensive analysis plan from data dictionary BEFORE querying databases.
        This is the mandatory first step for all agent analyses.
        
        Args:
            analysis_type: Type of analysis to perform
            parameters: Additional analysis parameters
            
        Returns:
            Complete analysis plan with data sources, business rules, and query patterns
        """
        parameters = parameters or {}
        
        # Get primary database and routing
        primary_db = cls.get_primary_database_for_analysis(analysis_type)
        recommended_tables = cls.get_tables_for_analysis(analysis_type)
        use_secondary = cls.should_use_secondary_database(analysis_type)
        
        # Get relevant data categories
        data_categories = cls._get_data_categories_for_analysis(analysis_type)
        
        # Build comprehensive plan
        analysis_plan = {
            "analysis_type": analysis_type,
            "primary_database": primary_db,
            "use_secondary_database": use_secondary,
            "recommended_tables": recommended_tables,
            "data_categories": data_categories,
            "business_rules": cls._get_business_rules_for_analysis(analysis_type),
            "query_patterns": cls._get_query_patterns_for_analysis(analysis_type),
            "data_validation_rules": cls._get_data_validation_rules(analysis_type),
            "performance_hints": cls._get_performance_hints(analysis_type),
            "pii_considerations": cls._get_pii_considerations_for_analysis(analysis_type),
            "routing_explanation": cls.ROUTING_RULES.get(analysis_type, {}).get("reason", "Standard routing")
        }
        
        return analysis_plan
    
    @classmethod
    def _get_data_categories_for_analysis(cls, analysis_type: str) -> Dict[str, Any]:
        """Get relevant data categories for analysis type."""
        category_mapping = {
            "financial_performance": {
                "primary_data": cls.DEPOSIT_DATA,
                "secondary_data": cls.LOAN_DATA,
                "supporting_data": cls.MEMBER_DATA
            },
            "member_analytics": {
                "primary_data": cls.MEMBER_DATA,
                "secondary_data": cls.DEPOSIT_DATA,
                "supporting_data": cls.TRANSACTION_DATA
            },
            "portfolio_risk": {
                "primary_data": cls.LOAN_DATA,
                "secondary_data": cls.COLLECTIONS_DATA,
                "supporting_data": cls.MEMBER_DATA
            },
            "loan_servicing": {
                "primary_data": cls.LOAN_DATA,
                "supporting_data": cls.MEMBER_DATA
            },
            "collections": {
                "primary_data": cls.COLLECTIONS_DATA,
                "secondary_data": cls.LOAN_DATA,
                "supporting_data": cls.MEMBER_DATA
            },
            "account_origination": {
                "primary_data": cls.ORIGINATION_DATA,
                "supporting_data": cls.MEMBER_DATA
            }
        }
        
        return category_mapping.get(analysis_type, {
            "primary_data": cls.MEMBER_DATA,
            "note": "Using default member data for unknown analysis type"
        })
    
    @classmethod
    def _get_business_rules_for_analysis(cls, analysis_type: str) -> List[str]:
        """Get business rules that must be applied for analysis type."""
        common_rules = [
            "Always filter out test accounts (ACCOUNT < 100 typically test)",
            "Respect PII protection requirements",
            "Use active status filters for current analysis",
            "Apply date range filters for performance"
        ]
        
        specific_rules = {
            "member_analytics": [
                "STATUS = 0 for active members only",
                "Use TYPE = 0 for primary name records",
                "CLOSEDATE = '19000101' means account is open"
            ],
            "financial_performance": [
                "CLOSEDATE = '19000101' for active accounts",
                "Sum balances across all product types",
                "Exclude charged-off accounts unless specifically analyzing losses"
            ],
            "portfolio_risk": [
                "CHARGEOFFDATE = '19000101' means not charged off",
                "Include delinquency status from both systems",
                "Use TEMENOS for collection workflow status"
            ],
            "collections": [
                "Focus on accounts with CHARGEOFFDATE != '19000101' OR delinquent status",
                "Use TEMENOS for workflow and queue management",
                "Cross-reference with ARCUSYM000 for account details"
            ]
        }
        
        rules = common_rules.copy()
        rules.extend(specific_rules.get(analysis_type, []))
        return rules
    
    @classmethod
    def _get_query_patterns_for_analysis(cls, analysis_type: str) -> Dict[str, str]:
        """Get recommended query patterns for analysis type."""
        patterns = {
            "member_analytics": {
                "active_members": """
                SELECT ACCOUNT, FIRST, LAST, JOINDATE, STATUS
                FROM NAME 
                WHERE TYPE = 0 AND STATUS = 0 AND ACCOUNT > 100
                """,
                "member_balances": """
                SELECT m.ACCOUNT, m.FIRST, m.LAST, 
                       COALESCE(SUM(s.BALANCE), 0) as TOTAL_DEPOSITS,
                       COALESCE(SUM(l.BALANCE), 0) as TOTAL_LOANS
                FROM NAME m
                LEFT JOIN SAVINGS s ON m.ACCOUNT = s.PARENTACCOUNT AND s.CLOSEDATE = '19000101'
                LEFT JOIN LOAN l ON m.ACCOUNT = l.PARENTACCOUNT AND l.CLOSEDATE = '19000101'
                WHERE m.TYPE = 0 AND m.STATUS = 0 AND m.ACCOUNT > 100
                GROUP BY m.ACCOUNT, m.FIRST, m.LAST
                """
            },
            "financial_performance": {
                "deposit_summary": """
                SELECT TYPE, COUNT(*) as ACCOUNT_COUNT, SUM(BALANCE) as TOTAL_BALANCE
                FROM SAVINGS 
                WHERE CLOSEDATE = '19000101' AND PARENTACCOUNT > 100
                GROUP BY TYPE
                """,
                "loan_summary": """
                SELECT TYPE, COUNT(*) as LOAN_COUNT, SUM(BALANCE) as TOTAL_BALANCE,
                       SUM(ORIGINALBALANCE) as TOTAL_ORIGINAL
                FROM LOAN
                WHERE CLOSEDATE = '19000101' AND CHARGEOFFDATE = '19000101' AND PARENTACCOUNT > 100
                GROUP BY TYPE
                """
            },
            "portfolio_risk": {
                "loan_portfolio": """
                SELECT TYPE, BALANCE, ORIGINALBALANCE, INTERESTRATE, DUEDATE,
                       CASE WHEN CHARGEOFFDATE != '19000101' THEN 'CHARGED_OFF' ELSE 'ACTIVE' END as STATUS
                FROM LOAN
                WHERE PARENTACCOUNT > 100 AND CLOSEDATE = '19000101'
                """
            }
        }
        
        return patterns.get(analysis_type, {
            "basic_query": "-- Use data dictionary to build appropriate queries for " + analysis_type
        })
    
    @classmethod
    def _get_data_validation_rules(cls, analysis_type: str) -> List[str]:
        """Get data validation rules for analysis type."""
        return [
            "Verify account numbers are positive integers",
            "Check date fields are in proper YYYYMMDD format",
            "Validate currency fields are numeric",
            "Ensure status codes are within valid ranges",
            "Cross-check totals across related tables",
            "Verify PII fields are properly protected"
        ]
    
    @classmethod
    def _get_performance_hints(cls, analysis_type: str) -> List[str]:
        """Get performance optimization hints for analysis type."""
        return [
            "Use appropriate date range filters to limit data volume",
            "Index on ACCOUNT/PARENTACCOUNT fields for joins",
            "Filter on CLOSEDATE early in WHERE clauses",
            "Use EXISTS instead of IN for large subqueries",
            "Consider using TOP clauses for large result sets",
            "Batch process large data sets rather than single queries"
        ]
    
    @classmethod
    def _get_pii_considerations_for_analysis(cls, analysis_type: str) -> Dict[str, Any]:
        """Get PII protection requirements for analysis type."""
        pii_fields = {
            "member_analytics": {
                "pii_fields": ["SSN", "FIRST", "LAST", "ADDRESS", "PHONE", "EMAIL", "BIRTHDATE"],
                "protection_level": "HIGH",
                "recommendations": [
                    "Mask or hash SSN in results",
                    "Use member IDs instead of names when possible",
                    "Aggregate data rather than individual records",
                    "Apply role-based access controls"
                ]
            },
            "financial_performance": {
                "pii_fields": ["SSN", "FIRST", "LAST"],
                "protection_level": "MEDIUM", 
                "recommendations": [
                    "Focus on aggregate numbers rather than individual accounts",
                    "Use account numbers instead of member names",
                    "Ensure results don't allow re-identification"
                ]
            },
            "portfolio_risk": {
                "pii_fields": ["SSN", "FIRST", "LAST", "ADDRESS"],
                "protection_level": "MEDIUM",
                "recommendations": [
                    "Use account identifiers rather than member names",
                    "Focus on risk categories and trends",
                    "Aggregate by loan types rather than individuals"
                ]
            }
        }
        
        return pii_fields.get(analysis_type, {
            "pii_fields": ["SSN", "FIRST", "LAST"],
            "protection_level": "STANDARD",
            "recommendations": ["Apply standard PII protection practices"]
        })
    
    @classmethod
    def validate_analysis_request(cls, analysis_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate analysis request against data dictionary before execution.
        
        Args:
            analysis_type: Type of analysis requested
            parameters: Analysis parameters
            
        Returns:
            Validation results with any issues or recommendations
        """
        validation = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check if analysis type is supported
        if analysis_type not in cls.ROUTING_RULES:
            validation["warnings"].append(f"Analysis type '{analysis_type}' not found in routing rules")
            validation["recommendations"].append("Consider using a standard analysis type for optimal data routing")
        
        # Validate database selection
        requested_db = parameters.get("database")
        recommended_db = cls.get_primary_database_for_analysis(analysis_type)
        
        if requested_db and requested_db != recommended_db:
            validation["warnings"].append(
                f"Requested database '{requested_db}' differs from recommended '{recommended_db}' for {analysis_type}"
            )
            validation["recommendations"].append(f"Consider using {recommended_db} for optimal results")
        
        # Check for required parameters
        if analysis_type == "member_analytics":
            method = parameters.get("method")
            if not method:
                validation["warnings"].append("Member analytics analysis should specify a method")
                validation["recommendations"].append("Use methods: active_members, demographics, balances, etc.")
        
        return validation
