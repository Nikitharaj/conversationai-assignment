"""
SQL Integration for Financial Q&A System with Mixture of Experts.

This script demonstrates how the MoE system can be integrated with SQL databases
for financial data analysis and reporting.
"""

import os
import json
import logging
import sqlite3
from pathlib import Path
import pandas as pd
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the MoE fine-tuner
from src.fine_tuning.mixture_of_experts import create_moe_fine_tuner


class FinancialDatabaseManager:
    """Manages the financial database for the Q&A system."""

    def __init__(self, db_path="financial_data.db"):
        """Initialize the database manager."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def connect(self):
        """Connect to the database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database: {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def setup_tables(self):
        """Set up the financial database tables."""
        try:
            # Create income statement table
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS income_statement (
                id INTEGER PRIMARY KEY,
                year INTEGER NOT NULL,
                quarter INTEGER NOT NULL,
                revenue REAL NOT NULL,
                cost_of_goods_sold REAL NOT NULL,
                gross_profit REAL NOT NULL,
                operating_expenses REAL NOT NULL,
                operating_income REAL NOT NULL,
                net_income REAL NOT NULL,
                earnings_per_share REAL NOT NULL,
                UNIQUE(year, quarter)
            )
            """)

            # Create balance sheet table
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS balance_sheet (
                id INTEGER PRIMARY KEY,
                year INTEGER NOT NULL,
                quarter INTEGER NOT NULL,
                total_assets REAL NOT NULL,
                total_liabilities REAL NOT NULL,
                total_equity REAL NOT NULL,
                cash_and_equivalents REAL NOT NULL,
                accounts_receivable REAL NOT NULL,
                inventory REAL NOT NULL,
                long_term_debt REAL NOT NULL,
                UNIQUE(year, quarter)
            )
            """)

            # Create cash flow table
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS cash_flow (
                id INTEGER PRIMARY KEY,
                year INTEGER NOT NULL,
                quarter INTEGER NOT NULL,
                operating_cash_flow REAL NOT NULL,
                investing_cash_flow REAL NOT NULL,
                financing_cash_flow REAL NOT NULL,
                capital_expenditures REAL NOT NULL,
                free_cash_flow REAL NOT NULL,
                dividends_paid REAL NOT NULL,
                UNIQUE(year, quarter)
            )
            """)

            # Create financial metrics table
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS financial_metrics (
                id INTEGER PRIMARY KEY,
                year INTEGER NOT NULL,
                quarter INTEGER NOT NULL,
                profit_margin REAL NOT NULL,
                return_on_assets REAL NOT NULL,
                return_on_equity REAL NOT NULL,
                debt_to_equity REAL NOT NULL,
                current_ratio REAL NOT NULL,
                quick_ratio REAL NOT NULL,
                UNIQUE(year, quarter)
            )
            """)

            # Create queries log table
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                expert TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)

            self.conn.commit()
            logger.info("Database tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to set up tables: {e}")
            return False

    def insert_sample_data(self):
        """Insert sample financial data into the database."""
        try:
            # Sample income statement data
            income_data = [
                (2022, 4, 1162.0, 710.0, 452.0, 232.0, 220.0, 167.4, 1.67),
                (2023, 4, 1250.0, 750.0, 500.0, 250.0, 250.0, 187.5, 1.88),
            ]
            self.cursor.executemany(
                "INSERT OR REPLACE INTO income_statement (year, quarter, revenue, cost_of_goods_sold, gross_profit, operating_expenses, operating_income, net_income, earnings_per_share) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                income_data,
            )

            # Sample balance sheet data
            balance_data = [
                (2022, 4, 1350.0, 600.0, 750.0, 300.0, 250.0, 200.0, 350.0),
                (2023, 4, 1500.0, 650.0, 850.0, 350.0, 275.0, 225.0, 375.0),
            ]
            self.cursor.executemany(
                "INSERT OR REPLACE INTO balance_sheet (year, quarter, total_assets, total_liabilities, total_equity, cash_and_equivalents, accounts_receivable, inventory, long_term_debt) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                balance_data,
            )

            # Sample cash flow data
            cash_flow_data = [
                (2022, 4, 200.0, -150.0, -30.0, 150.0, 50.0, 20.0),
                (2023, 4, 220.0, -170.0, -25.0, 170.0, 50.0, 22.5),
            ]
            self.cursor.executemany(
                "INSERT OR REPLACE INTO cash_flow (year, quarter, operating_cash_flow, investing_cash_flow, financing_cash_flow, capital_expenditures, free_cash_flow, dividends_paid) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                cash_flow_data,
            )

            # Sample financial metrics data
            metrics_data = [
                (2022, 4, 0.144, 0.124, 0.223, 0.8, 1.5, 1.2),
                (2023, 4, 0.15, 0.125, 0.221, 0.765, 1.55, 1.25),
            ]
            self.cursor.executemany(
                "INSERT OR REPLACE INTO financial_metrics (year, quarter, profit_margin, return_on_assets, return_on_equity, debt_to_equity, current_ratio, quick_ratio) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                metrics_data,
            )

            self.conn.commit()
            logger.info("Sample data inserted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to insert sample data: {e}")
            return False

    def execute_query(self, query):
        """Execute a SQL query and return the results."""
        try:
            result = pd.read_sql_query(query, self.conn)
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return None

    def log_query(self, query, answer, expert, confidence):
        """Log a query and its response."""
        try:
            self.cursor.execute(
                "INSERT INTO query_log (query, answer, expert, confidence) VALUES (?, ?, ?, ?)",
                (query, answer, expert, confidence),
            )
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to log query: {e}")
            return False


class FinancialQueryProcessor:
    """Processes financial queries using the MoE system and SQL database."""

    def __init__(
        self, moe_model_dir="models/moe_fine_tuned", db_path="financial_data.db"
    ):
        """Initialize the query processor."""
        self.moe_tuner = create_moe_fine_tuner(
            model_name="distilgpt2", output_dir=moe_model_dir
        )
        self.db_manager = FinancialDatabaseManager(db_path)
        self.is_initialized = False

    def initialize(self, train_data_path):
        """Initialize the system with training data."""
        # Connect to the database
        if not self.db_manager.connect():
            return False

        # Set up database tables
        if not self.db_manager.setup_tables():
            return False

        # Insert sample data
        if not self.db_manager.insert_sample_data():
            return False

        # Load training data
        try:
            with open(train_data_path, "r", encoding="utf-8") as f:
                train_pairs = json.load(f)

            # Enrich with section information
            enriched_train_pairs = self._enrich_qa_pairs_with_sections(train_pairs)

            # Train the MoE system
            success = self.moe_tuner.train_moe(enriched_train_pairs, epochs=3)
            if success:
                logger.info("MoE system trained successfully")
                self.is_initialized = True
                return True
            else:
                logger.error("MoE system training failed")
                return False
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def _enrich_qa_pairs_with_sections(self, qa_pairs):
        """Enrich Q&A pairs with section information."""
        enriched_pairs = []

        # Define section keywords for classification
        section_keywords = {
            "income_statement": [
                "revenue",
                "sales",
                "income",
                "profit",
                "earnings",
                "ebitda",
                "margin",
                "cost",
                "expense",
                "operating",
                "gross",
                "net income",
                "eps",
            ],
            "balance_sheet": [
                "assets",
                "liabilities",
                "equity",
                "debt",
                "cash",
                "inventory",
                "receivables",
                "payables",
                "balance sheet",
                "book value",
            ],
            "cash_flow": [
                "cash flow",
                "operating cash",
                "free cash",
                "capex",
                "capital expenditure",
                "financing",
                "investing",
                "dividends",
                "share repurchase",
            ],
            "notes_mda": [
                "guidance",
                "outlook",
                "strategy",
                "risk",
                "management",
                "discussion",
                "analysis",
                "segment",
                "geographic",
                "employees",
                "environmental",
            ],
        }

        for pair in qa_pairs:
            question = pair["question"].lower()

            # Determine section based on keywords
            section_scores = {}
            for section, keywords in section_keywords.items():
                score = sum(1 for keyword in keywords if keyword in question)
                section_scores[section] = score

            # Assign section with highest score, default to income_statement
            section = (
                max(section_scores, key=section_scores.get)
                if any(section_scores.values())
                else "income_statement"
            )

            # Create enriched pair
            enriched_pair = pair.copy()
            enriched_pair["section"] = section
            enriched_pairs.append(enriched_pair)

        return enriched_pairs

    def process_query(self, query):
        """Process a financial query."""
        if not self.is_initialized:
            return {"error": "System not initialized"}

        # Process with MoE system
        moe_result = self.moe_tuner.process_query(query)

        # Try to generate SQL if it's a data-related query
        sql_query = self._generate_sql_query(query, moe_result["selected_expert"])

        # Execute SQL query if available
        sql_result = None
        if sql_query:
            sql_result = self.db_manager.execute_query(sql_query)

        # Log the query
        self.db_manager.log_query(
            query,
            moe_result["answer"],
            moe_result["selected_expert"],
            moe_result["confidence"],
        )

        # Prepare response
        response = {
            "query": query,
            "answer": moe_result["answer"],
            "confidence": moe_result["confidence"],
            "expert": moe_result["selected_expert"],
            "expert_weights": moe_result["expert_weights"],
            "sql_query": sql_query,
            "sql_result": sql_result.to_dict("records")
            if sql_result is not None
            else None,
            "timestamp": datetime.now().isoformat(),
        }

        return response

    def _generate_sql_query(self, query, expert):
        """Generate SQL query based on the user query and selected expert."""
        query_lower = query.lower()

        # Simple rule-based SQL generation
        if expert == "income_statement":
            if "revenue" in query_lower and "2023" in query_lower:
                return "SELECT year, revenue FROM income_statement WHERE year = 2023"
            elif "revenue" in query_lower:
                return "SELECT year, revenue FROM income_statement ORDER BY year DESC"
            elif "profit margin" in query_lower:
                return "SELECT year, profit_margin FROM financial_metrics ORDER BY year DESC"
            elif "net income" in query_lower or "profit" in query_lower:
                return (
                    "SELECT year, net_income FROM income_statement ORDER BY year DESC"
                )

        elif expert == "balance_sheet":
            if "assets" in query_lower:
                return "SELECT year, total_assets FROM balance_sheet ORDER BY year DESC"
            elif "liabilities" in query_lower:
                return "SELECT year, total_liabilities FROM balance_sheet ORDER BY year DESC"
            elif "equity" in query_lower:
                return "SELECT year, total_equity FROM balance_sheet ORDER BY year DESC"

        elif expert == "cash_flow":
            if "cash flow" in query_lower:
                return "SELECT year, operating_cash_flow, investing_cash_flow, financing_cash_flow FROM cash_flow ORDER BY year DESC"
            elif "free cash" in query_lower:
                return "SELECT year, free_cash_flow FROM cash_flow ORDER BY year DESC"

        # Default queries by expert
        if expert == "income_statement":
            return "SELECT * FROM income_statement ORDER BY year DESC, quarter DESC LIMIT 1"
        elif expert == "balance_sheet":
            return (
                "SELECT * FROM balance_sheet ORDER BY year DESC, quarter DESC LIMIT 1"
            )
        elif expert == "cash_flow":
            return "SELECT * FROM cash_flow ORDER BY year DESC, quarter DESC LIMIT 1"
        elif expert == "notes_mda":
            return "SELECT year, profit_margin, return_on_equity FROM financial_metrics ORDER BY year DESC"

        return None

    def close(self):
        """Close the system."""
        self.db_manager.close()


def main():
    """Main function to demonstrate the SQL integration."""
    # Define paths
    project_root = Path.cwd()
    data_dir = project_root / "data"
    qa_pairs_dir = data_dir / "qa_pairs"
    moe_model_dir = project_root / "models" / "moe_fine_tuned"
    db_path = project_root / "financial_data.db"

    # Create model directory if it doesn't exist
    moe_model_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the query processor
    processor = FinancialQueryProcessor(
        moe_model_dir=str(moe_model_dir), db_path=str(db_path)
    )

    # Initialize the system
    train_data_path = qa_pairs_dir / "financial_qa_train.json"
    if processor.initialize(train_data_path):
        logger.info("System initialized successfully")

        # Process sample queries
        sample_queries = [
            "What was the revenue in 2023?",
            "What are the total assets?",
            "How much free cash flow was generated?",
            "What is the company's profit margin?",
            "What is your favorite color?",  # Irrelevant query
        ]

        for query in sample_queries:
            logger.info(f"\nProcessing query: {query}")
            result = processor.process_query(query)

            logger.info(f"Answer: {result['answer']}")
            logger.info(
                f"Expert: {result['expert']} (confidence: {result['confidence']:.2f})"
            )

            if result["sql_query"]:
                logger.info(f"SQL Query: {result['sql_query']}")
                if result["sql_result"]:
                    logger.info(f"SQL Result: {result['sql_result']}")

        # Close the processor
        processor.close()
    else:
        logger.error("System initialization failed")


if __name__ == "__main__":
    main()
