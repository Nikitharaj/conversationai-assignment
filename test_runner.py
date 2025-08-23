#!/usr/bin/env python
"""
Unified Test Runner for Financial Q&A System

This script provides a simple way to run all tests from the terminal.
Usage:
    python test_runner.py                    # Run all tests
    python test_runner.py --unit            # Run only unit tests
    python test_runner.py --integration     # Run only integration tests
    python test_runner.py --moe             # Run only MoE tests
    python test_runner.py --verbose         # Verbose output
"""

import os
import sys
import argparse
from pathlib import Path

# Add scripts directory to path for shared utilities
sys.path.append(str(Path(__file__).parent / "scripts"))

from utils import setup_logging, run_command, print_summary, get_project_root

logger = setup_logging("test_runner")


class TestRunner:
    """Unified test runner for the Financial Q&A system."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.project_root = get_project_root()
        self.tests_dir = self.project_root / "tests"
        self.scripts_dir = self.project_root / "scripts"

    def run_unit_tests(self):
        """Run standard unit tests using pytest."""
        logger.info("=" * 60)
        logger.info(" RUNNING UNIT TESTS")
        logger.info("=" * 60)

        # List of unit test modules to run
        unit_test_dirs = [
            "tests/data_processing",
            "tests/evaluation",
            "tests/fine_tuning",
            "tests/rag_system",
            "tests/ui",
        ]

        results = []
        for test_dir in unit_test_dirs:
            if (self.project_root / test_dir).exists():
                success, _ = run_command(
                    f"python -m pytest {test_dir} -v",
                    f"Unit tests in {test_dir}",
                    cwd=self.project_root,
                    verbose=self.verbose,
                )
                results.append(success)

        return all(results)

    def run_integration_tests(self):
        """Run integration tests."""
        logger.info("=" * 60)
        logger.info("🔗 RUNNING INTEGRATION TESTS")
        logger.info("=" * 60)

        integration_tests = [
            ("tests/test_integration.py", "Integration tests"),
        ]

        results = []
        for test_file, description in integration_tests:
            test_path = self.project_root / test_file
            if test_path.exists():
                success, _ = run_command(
                    f"python -m pytest {test_file} -v",
                    description,
                    cwd=self.project_root,
                    verbose=self.verbose,
                )
                results.append(success)

        return all(results) if results else True

    def run_moe_tests(self):
        """Run MoE-specific tests."""
        logger.info("=" * 60)
        logger.info(" RUNNING MOE TESTS")
        logger.info("=" * 60)

        moe_tests = [
            ("tests/test_moe.py", "MoE Basic Functionality"),
            ("tests/test_router.py", "Router Component"),
            ("tests/test_specialized_experts.py", "Specialized Experts"),
            ("tests/test_comparative.py", "Comparative Analysis"),
        ]

        results = []
        for test_file, description in moe_tests:
            test_path = self.project_root / test_file
            if test_path.exists():
                success, _ = run_command(
                    f"python {test_file}",
                    description,
                    cwd=self.project_root,
                    verbose=self.verbose,
                )
                results.append(success)
            else:
                logger.warning(f"  Test file not found: {test_file}")

        return all(results) if results else True

    def run_system_tests(self):
        """Run end-to-end system tests."""
        logger.info("=" * 60)
        logger.info(" RUNNING SYSTEM TESTS")
        logger.info("=" * 60)

        # Test if the main app can be imported
        success, _ = run_command(
            "python -c \"import app; print(' Main app imports successfully')\"",
            "Main App Import Test",
            cwd=self.project_root,
            verbose=self.verbose,
        )

        return success

    def run_all_tests(self):
        """Run all test suites."""
        import time

        start_time = time.time()

        logger.info(" STARTING COMPREHENSIVE TEST SUITE")
        logger.info("=" * 60)

        # Run all test categories
        results = {
            "Unit Tests": self.run_unit_tests(),
            "Integration Tests": self.run_integration_tests(),
            "MoE Tests": self.run_moe_tests(),
            "System Tests": self.run_system_tests(),
        }

        # Summary
        end_time = time.time()
        duration = end_time - start_time

        # Use shared summary function
        success = print_summary(results, "TEST SUMMARY")

        logger.info(f"Duration: {duration:.2f} seconds")
        return success


def main():
    parser = argparse.ArgumentParser(description="Financial Q&A System Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument("--moe", action="store_true", help="Run MoE tests only")
    parser.add_argument("--system", action="store_true", help="Run system tests only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    runner = TestRunner(verbose=args.verbose)

    success = True

    if args.unit:
        success = runner.run_unit_tests()
    elif args.integration:
        success = runner.run_integration_tests()
    elif args.moe:
        success = runner.run_moe_tests()
    elif args.system:
        success = runner.run_system_tests()
    else:
        # Run all tests
        success = runner.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
