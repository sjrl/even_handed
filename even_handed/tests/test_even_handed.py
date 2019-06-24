"""
Unit and regression test for the even_handed package.
"""

# Import package, test suite, and other packages as needed
import even_handed
import pytest
import sys

def test_even_handed_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "even_handed" in sys.modules
