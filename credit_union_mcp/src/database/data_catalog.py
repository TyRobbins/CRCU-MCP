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
