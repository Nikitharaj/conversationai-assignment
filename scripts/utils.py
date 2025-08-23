#!/usr/bin/env python
"""
Shared utilities for scripts and test runners.
"""

import logging
import subprocess
import time
from pathlib import Path


def setup_logging(name="script"):
    """Set up standardized logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(name)


def run_command(command, description, cwd=None, timeout=None, verbose=False):
    """
    Run a command and return success status.

    Args:
        command: Command to run
        description: Description for logging
        cwd: Working directory (optional)
        timeout: Command timeout (optional)
        verbose: Whether to show output in real time

    Returns:
        tuple: (success: bool, output: str)
    """
    logger = logging.getLogger(__name__)
    logger.info(f" {description}")

    try:
        start_time = time.time()

        result = subprocess.run(
            command,
            shell=True,
            check=True,
            cwd=cwd,
            capture_output=not verbose,
            text=True,
            timeout=timeout,
        )

        elapsed_time = time.time() - start_time

        if verbose and result.stdout:
            print(result.stdout)

        logger.info(f" {description} - PASSED ({elapsed_time:.2f}s)")
        return True, result.stdout if result.stdout else ""

    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        logger.error(f" {description} - FAILED ({elapsed_time:.2f}s)")

        error_output = ""
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
            error_output += f"stdout: {e.stdout}\n"
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
            error_output += f"stderr: {e.stderr}\n"

        return False, error_output

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        logger.error(f" {description} - TIMEOUT ({elapsed_time:.2f}s)")
        return False, f"Command timed out after {timeout}s"


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def print_summary(results, title="SUMMARY"):
    """Print a formatted summary of test results."""
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info(f" {title}")
    logger.info("=" * 60)

    passed = 0
    total = len(results)

    for name, success in results.items():
        status = " PASSED" if success else " FAILED"
        logger.info(f"{name:30} {status}")
        if success:
            passed += 1

    logger.info("-" * 60)
    logger.info(f"Total: {passed}/{total} passed")

    if passed == total:
        logger.info(" ALL TESTS PASSED!")
        return True
    else:
        logger.error(f" {total - passed} test(s) failed!")
        return False
