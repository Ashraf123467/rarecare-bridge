"""
Convenience test runner script for RareCare Bridge test suite.
"""

import sys
import pytest

if __name__ == "__main__":
    exit_code = pytest.main(["tests/test_structuring.py", "tests/test_federated.py", "tests/test_privacy.py", "-v", "-p", "no:anyio"])
    sys.exit(exit_code)
