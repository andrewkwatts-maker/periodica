"""
Pytest plugins and hooks for automatic report generation.

This module provides automatic configuration for HTML and JSON report generation
when the required plugins (pytest-html, pytest-json-report) are installed.
"""

import os
from pathlib import Path


def pytest_configure(config):
    """
    Configure pytest with HTML and JSON report options if plugins are available.

    This hook runs during pytest configuration and automatically adds
    report generation options when the plugins are installed.
    """
    # Get the root directory (where pyproject.toml is located)
    rootdir = config.rootdir
    reports_dir = Path(rootdir) / "outputs" / "reports" / "test_results"

    # Ensure the reports directory exists
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Check if pytest-html is available and configure it
    if config.pluginmanager.hasplugin("html"):
        html_path = reports_dir / "report.html"
        if not config.option.htmlpath:
            config.option.htmlpath = str(html_path)
        if hasattr(config.option, 'self_contained_html'):
            config.option.self_contained_html = True

    # Check if pytest-json-report is available and configure it
    if config.pluginmanager.hasplugin("json_report"):
        json_path = reports_dir / "report.json"
        if hasattr(config.option, 'json_report_file') and not config.option.json_report_file:
            config.option.json_report_file = str(json_path)
        if hasattr(config.option, 'json_report_indent'):
            config.option.json_report_indent = 2
        if hasattr(config.option, 'json_report'):
            config.option.json_report = True


def pytest_report_header(config):
    """
    Add report paths to the pytest header output.
    """
    lines = []
    rootdir = config.rootdir
    reports_dir = Path(rootdir) / "outputs" / "reports" / "test_results"

    if config.pluginmanager.hasplugin("html"):
        lines.append(f"HTML report: {reports_dir / 'report.html'}")

    if config.pluginmanager.hasplugin("json_report"):
        lines.append(f"JSON report: {reports_dir / 'report.json'}")

    return lines if lines else None
