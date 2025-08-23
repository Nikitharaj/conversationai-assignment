#!/usr/bin/env python
"""
MoE Financial Q&A System Interactive Demo.

This script provides an interactive demo of the Mixture of Experts (MoE)
Financial Q&A system with SQL integration.
"""

import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import time
import sqlite3
from datetime import datetime
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import project modules
from src.fine_tuning.mixture_of_experts import create_moe_fine_tuner


class FinancialQADemo:
    """Interactive demo for the MoE Financial Q&A system."""

    def __init__(self, model_dir="models/moe_fine_tuned", db_path="financial_data.db"):
        """Initialize the demo."""
        self.model_dir = Path(model_dir)
        self.db_path = Path(db_path)
        self.moe_model = None
        self.conn = None
        self.cursor = None
        self.is_initialized = False

    def initialize(self, train_data_path):
        """Initialize the MoE model and database."""
        logger.info("Initializing MoE Financial Q&A Demo...")

        # Initialize MoE model
        logger.info("Loading MoE model...")
        try:
            self.moe_model = create_moe_fine_tuner(
                model_name="distilgpt2", output_dir=str(self.model_dir)
            )

            # Create model directory if it doesn't exist
            self.model_dir.mkdir(parents=True, exist_ok=True)

            # Load and enrich training data
            with open(train_data_path, "r", encoding="utf-8") as f:
                train_pairs = json.load(f)

            enriched_train_pairs = self._enrich_qa_pairs(train_pairs)

            # Train MoE model
            logger.info("Training MoE model...")
            success = self.moe_model.train_moe(enriched_train_pairs, epochs=3)

            if not success:
                logger.error("Failed to train MoE model")
                return False

            logger.info("MoE model trained successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MoE model: {e}")
            return False

        # Initialize database
        logger.info("Connecting to database...")
        try:
            # Connect to database
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()

            # Check if tables exist
            self.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='income_statement'"
            )
            if not self.cursor.fetchone():
                logger.info("Creating database tables...")
                self._setup_database()
            else:
                logger.info("Database tables already exist")

            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False

        self.is_initialized = True
        logger.info("MoE Financial Q&A Demo initialized successfully")
        return True

    def _enrich_qa_pairs(self, qa_pairs):
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

    def _setup_database(self):
        """Set up the financial database."""
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

        # Insert sample data
        self._insert_sample_data()

        self.conn.commit()

    def _insert_sample_data(self):
        """Insert sample financial data into the database."""
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

    def process_query(self, query):
        """Process a financial query."""
        if not self.is_initialized:
            return {"error": "System not initialized"}

        # Process with MoE system
        moe_result = self.moe_model.process_query(query)

        # Try to generate SQL if it's a data-related query
        sql_query = self._generate_sql_query(query, moe_result["selected_expert"])

        # Execute SQL query if available
        sql_result = None
        if sql_query:
            try:
                sql_result = pd.read_sql_query(sql_query, self.conn)
            except Exception as e:
                logger.error(f"SQL query execution failed: {e}")

        # Log the query
        try:
            self.cursor.execute(
                "INSERT INTO query_log (query, answer, expert, confidence) VALUES (?, ?, ?, ?)",
                (
                    query,
                    moe_result["answer"],
                    moe_result["selected_expert"],
                    moe_result["confidence"],
                ),
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to log query: {e}")

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

    def run_interactive_demo(self):
        """Run the interactive demo."""
        print("\n===== MoE Financial Q&A System Demo =====")
        print("Type 'exit' to quit, 'help' for commands\n")

        while True:
            try:
                query = input("\nEnter your financial question: ")

                if query.lower() in ["exit", "quit", "q"]:
                    break

                if query.lower() in ["help", "h", "?"]:
                    self._print_help()
                    continue

                if query.lower() in ["stats", "statistics"]:
                    self._print_stats()
                    continue

                if query.lower() in ["history", "log"]:
                    self._print_history()
                    continue

                # Process the query
                start_time = time.time()
                result = self.process_query(query)
                elapsed_time = time.time() - start_time

                # Display the result
                print(f"\nAnswer: {result['answer']}")
                print(
                    f"Expert: {result['expert']} (confidence: {result['confidence']:.2f})"
                )
                print(f"Response time: {elapsed_time:.3f} seconds")

                if result["sql_query"]:
                    print(f"\nSQL Query: {result['sql_query']}")

                    if result["sql_result"]:
                        print("\nSQL Result:")
                        df = pd.DataFrame(result["sql_result"])
                        print(df.to_string(index=False))

            except KeyboardInterrupt:
                print("\nExiting...")
                break

            except Exception as e:
                print(f"Error: {e}")

    def _print_help(self):
        """Print help information."""
        print("\n===== Commands =====")
        print("help    - Show this help message")
        print("stats   - Show system statistics")
        print("history - Show query history")
        print("exit    - Exit the demo")
        print("\n===== Example Questions =====")
        print("1. What was the revenue in 2023?")
        print("2. What are the total assets?")
        print("3. How much free cash flow was generated?")
        print("4. What is the company's profit margin?")
        print("5. What is the outlook for the next fiscal year?")

    def _print_stats(self):
        """Print system statistics."""
        try:
            # Get expert stats
            expert_stats = self.moe_model.get_expert_stats()

            # Get query counts by expert
            self.cursor.execute(
                "SELECT expert, COUNT(*) as count FROM query_log GROUP BY expert"
            )
            expert_counts = {row[0]: row[1] for row in self.cursor.fetchall()}

            # Get average confidence by expert
            self.cursor.execute(
                "SELECT expert, AVG(confidence) as avg_confidence FROM query_log GROUP BY expert"
            )
            expert_confidences = {row[0]: row[1] for row in self.cursor.fetchall()}

            # Print stats
            print("\n===== System Statistics =====")
            print(f"Model: {expert_stats['model_name']}")
            print(f"Number of experts: {expert_stats['num_experts']}")
            print(f"Expert sections: {', '.join(expert_stats['expert_sections'])}")

            print("\n--- Expert Usage ---")
            for section in expert_stats["expert_sections"]:
                count = expert_counts.get(section, 0)
                confidence = expert_confidences.get(section, 0)
                print(f"{section}: {count} queries, {confidence:.2f} avg confidence")

            # Get total queries
            self.cursor.execute("SELECT COUNT(*) FROM query_log")
            total_queries = self.cursor.fetchone()[0]

            print(f"\nTotal queries: {total_queries}")

        except Exception as e:
            print(f"Error getting statistics: {e}")

    def _print_history(self):
        """Print query history."""
        try:
            # Get recent queries
            self.cursor.execute(
                "SELECT query, answer, expert, confidence, timestamp FROM query_log ORDER BY timestamp DESC LIMIT 10"
            )
            history = self.cursor.fetchall()

            if not history:
                print("No query history found")
                return

            print("\n===== Recent Queries =====")
            for i, (query, answer, expert, confidence, timestamp) in enumerate(history):
                print(f"\n{i + 1}. Query: {query}")
                print(f"   Answer: {answer}")
                print(f"   Expert: {expert} (confidence: {confidence:.2f})")
                print(f"   Time: {timestamp}")

        except Exception as e:
            print(f"Error getting history: {e}")

    def close(self):
        """Close the demo."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="MoE Financial Q&A System Interactive Demo"
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/moe_fine_tuned",
        help="Directory for the MoE model",
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default="financial_data.db",
        help="Path to the SQLite database",
    )

    parser.add_argument(
        "--train-data",
        type=str,
        default="data/qa_pairs/financial_qa_train.json",
        help="Path to the training data",
    )

    args = parser.parse_args()

    # Create demo
    demo = FinancialQADemo(model_dir=args.model_dir, db_path=args.db_path)

    # Initialize demo
    if demo.initialize(args.train_data):
        # Run interactive demo
        try:
            demo.run_interactive_demo()
        finally:
            demo.close()
    else:
        logger.error("Failed to initialize demo")


if __name__ == "__main__":
    main()
