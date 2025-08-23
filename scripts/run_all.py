#!/usr/bin/env python
"""
Master script to run all demos of the Financial Q&A system.

This script runs all the demo components in sequence:
1. SQL integration demo
2. Interactive MoE demo

Note: For running tests, use the unified test runner:
    python test_runner.py
"""

import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_command(command, description, timeout=None):
    """Run a command and log the output."""
    logger.info(f"Running {description}...")
    try:
        start_time = time.time()
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        elapsed_time = time.time() - start_time
        logger.info(
            f" {description} completed successfully in {elapsed_time:.2f} seconds"
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f" {description} failed with error code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False, e.stderr
    except subprocess.TimeoutExpired:
        logger.error(f" {description} timed out after {timeout} seconds")
        return False, "Timed out"


def run_all_components(args):
    """Run all components in sequence."""
    # Create results directory
    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)

    # Define components
    components = [
        {
            "name": "Basic MoE Test",
            "command": "python test_moe.py",
            "description": "Basic MoE system functionality",
            "timeout": 300,  # 5 minutes
        },
        {
            "name": "SQL Integration",
            "command": "python sql_integration.py",
            "description": "SQL database integration",
            "timeout": 300,  # 5 minutes
        },
        {
            "name": "Comparative Analysis",
            "command": "python test_comparative.py",
            "description": "Comparison with standard fine-tuning",
            "timeout": 600,  # 10 minutes
        },
        {
            "name": "Specialized Experts Test",
            "command": "python test_specialized_experts.py",
            "description": "Evaluation of specialized experts",
            "timeout": 300,  # 5 minutes
        },
        {
            "name": "Router Evaluation",
            "command": "python test_router.py",
            "description": "Evaluation of the router component",
            "timeout": 300,  # 5 minutes
        },
        {
            "name": "Interactive Demo",
            "command": "python moe_demo.py",
            "description": "Interactive demo (will run for 30 seconds then terminate)",
            "timeout": 30,  # 30 seconds - just to show it works
        },
    ]

    # Run components
    results = {}
    overall_start_time = time.time()

    for component in components:
        if args.skip and component["name"].lower() in args.skip:
            logger.info(f"Skipping {component['name']}...")
            continue

        success, output = run_command(
            component["command"],
            component["description"],
            timeout=component["timeout"],
        )
        results[component["name"]] = {
            "success": success,
            "output": output,
        }

        # Save output to file
        output_file = (
            results_dir / f"{component['name'].lower().replace(' ', '_')}_output.txt"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)

    overall_elapsed_time = time.time() - overall_start_time

    # Print summary
    logger.info("\n===== SUMMARY =====")
    logger.info(f"Total time: {overall_elapsed_time:.2f} seconds")

    passed = sum(1 for result in results.values() if result["success"])
    failed = sum(1 for result in results.values() if not result["success"])

    logger.info(f"Components passed: {passed}")
    logger.info(f"Components failed: {failed}")

    for name, result in results.items():
        status = " PASSED" if result["success"] else " FAILED"
        logger.info(f"{name}: {status}")

    logger.info(f"\nDetailed outputs saved to {results_dir}/")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run all components of the Financial Q&A system"
    )

    parser.add_argument(
        "--skip",
        type=str,
        nargs="+",
        choices=["basic", "sql", "comparative", "specialized", "router", "demo"],
        help="Skip specific components",
    )

    args = parser.parse_args()

    run_all_components(args)


if __name__ == "__main__":
    main()
