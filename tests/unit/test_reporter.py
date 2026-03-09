"""
Unit tests for reporter module.
"""

from io import StringIO
from unittest.mock import patch

import pytest

from model_compression.reporter.console_reporter import ConsoleReporter


class TestConsoleReporter:
    """Test ConsoleReporter class."""

    def test_report_output(self):
        """Test that report produces output."""
        reporter = ConsoleReporter()

        original_result = {
            "avg_latency_ms": 10.5,
            "min_latency_ms": 9.2,
            "max_latency_ms": 12.1,
            "total_params": 1000000,
            "param_size_mb": 4.0,
        }

        compressed_result = {
            "avg_latency_ms": 5.2,
            "min_latency_ms": 4.8,
            "max_latency_ms": 6.1,
            "total_params": 700000,
            "param_size_mb": 2.8,
        }

        # Capture stdout
        with patch("sys.stdout", new=StringIO()) as fake_out:
            reporter.report(original_result, compressed_result)
            output = fake_out.getvalue()

        # Check that output contains key information
        assert "Benchmark Results" in output
        assert "Original Model" in output
        assert "Compressed Model" in output
        assert "Improvement" in output
        assert "10.5" in output  # original latency
        assert "5.2" in output  # compressed latency

    def test_print_result(self):
        """Test _print_result method."""
        reporter = ConsoleReporter()

        result = {
            "avg_latency_ms": 8.5,
            "min_latency_ms": 7.2,
            "max_latency_ms": 10.1,
            "total_params": 500000,
            "param_size_mb": 2.0,
        }

        with patch("sys.stdout", new=StringIO()) as fake_out:
            reporter._print_result(result)
            output = fake_out.getvalue()

        assert "8.5" in output
        assert "7.2" in output
        assert "10.1" in output
        assert "500,000" in output
        assert "2.0" in output

    def test_print_comparison(self):
        """Test _print_comparison method."""
        reporter = ConsoleReporter()

        original = {
            "avg_latency_ms": 10.0,
            "total_params": 1000000,
            "param_size_mb": 4.0,
        }

        compressed = {
            "avg_latency_ms": 5.0,
            "total_params": 500000,
            "param_size_mb": 2.0,
        }

        with patch("sys.stdout", new=StringIO()) as fake_out:
            reporter._print_comparison(original, compressed)
            output = fake_out.getvalue()

        assert "Speedup" in output
        assert "2.00x" in output  # 10.0 / 5.0 = 2.0x
        assert "Size Reduction" in output
        assert "50.00%" in output  # (1 - 2.0/4.0) * 100 = 50%
        assert "Parameter Reduction" in output

    def test_comparison_calculations(self):
        """Test that comparison calculations are correct."""
        reporter = ConsoleReporter()

        original = {
            "avg_latency_ms": 20.0,
            "total_params": 1000000,
            "param_size_mb": 4.0,
        }

        compressed = {
            "avg_latency_ms": 4.0,
            "total_params": 300000,
            "param_size_mb": 1.2,
        }

        with patch("sys.stdout", new=StringIO()) as fake_out:
            reporter._print_comparison(original, compressed)
            output = fake_out.getvalue()

        # Speedup: 20.0 / 4.0 = 5.0x
        assert "5.00x" in output

        # Size reduction: (1 - 1.2/4.0) * 100 = 70%
        assert "70.00%" in output

        # Param reduction: (1 - 300000/1000000) * 100 = 70%
        assert "70.00%" in output
