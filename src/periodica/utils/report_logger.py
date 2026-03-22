"""Report logging utility for AI generation and other operations.

Logs success/failure messages to files instead of requiring user interaction.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class ReportLogger:
    """Logger that writes operation results to report files."""

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the report logger.

        Args:
            base_dir: Base directory for reports. Defaults to project outputs/reports.
        """
        if base_dir is None:
            # Default to project outputs/reports directory
            project_root = Path(__file__).parent.parent
            base_dir = project_root / "outputs" / "reports"

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different report types
        self.ai_dir = self.base_dir / "ai_generation"
        self.ai_dir.mkdir(exist_ok=True)

    def log_ai_generation(
        self,
        asset_type: str,
        name: str,
        success: bool,
        config: Optional[dict] = None,
        error: Optional[str] = None
    ) -> str:
        """Log an AI generation result.

        Args:
            asset_type: Type of asset generated (e.g., 'protein', 'element')
            name: Name of the generated asset
            success: Whether generation succeeded
            config: The generated configuration (on success)
            error: Error message (on failure)

        Returns:
            Path to the log file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        status = "success" if success else "failure"
        filename = f"{asset_type}_{status}_{timestamp}.json"
        filepath = self.ai_dir / filename

        report = {
            "timestamp": datetime.now().isoformat(),
            "asset_type": asset_type,
            "name": name,
            "success": success,
            "status": status,
        }

        if success and config:
            report["config"] = config
            report["message"] = f"Successfully created AI-generated {asset_type}: '{name}'"
        elif not success:
            report["error"] = error or "Unknown error"
            report["message"] = f"Failed to save AI-generated {asset_type}: {error or 'Unknown error'}"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)

        return str(filepath)

    def log_operation(
        self,
        operation: str,
        asset_type: str,
        name: str,
        success: bool,
        details: Optional[str] = None
    ) -> str:
        """Log a general operation result.

        Args:
            operation: Type of operation (e.g., 'export', 'import', 'delete')
            asset_type: Type of asset
            name: Name of the asset
            success: Whether operation succeeded
            details: Additional details

        Returns:
            Path to the log file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        status = "success" if success else "failure"
        filename = f"{operation}_{asset_type}_{status}_{timestamp}.json"

        # Create operation-specific subdirectory
        op_dir = self.base_dir / operation
        op_dir.mkdir(exist_ok=True)
        filepath = op_dir / filename

        report = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "asset_type": asset_type,
            "name": name,
            "success": success,
            "status": status,
            "details": details
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        return str(filepath)

    def get_latest_reports(self, count: int = 10) -> list[dict]:
        """Get the most recent reports.

        Args:
            count: Number of reports to retrieve

        Returns:
            List of report dictionaries
        """
        reports = []

        # Gather all JSON files from all subdirectories
        for json_file in self.base_dir.rglob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                    report["_filepath"] = str(json_file)
                    reports.append(report)
            except (json.JSONDecodeError, IOError):
                continue

        # Sort by timestamp descending
        reports.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

        return reports[:count]

    def get_summary(self) -> dict:
        """Get a summary of all logged operations.

        Returns:
            Dictionary with operation counts and success rates
        """
        reports = self.get_latest_reports(count=1000)

        summary = {
            "total": len(reports),
            "success": sum(1 for r in reports if r.get("success")),
            "failure": sum(1 for r in reports if not r.get("success")),
            "by_type": {},
            "by_operation": {}
        }

        for report in reports:
            asset_type = report.get("asset_type", "unknown")
            operation = report.get("operation", "ai_generation")

            if asset_type not in summary["by_type"]:
                summary["by_type"][asset_type] = {"success": 0, "failure": 0}
            if operation not in summary["by_operation"]:
                summary["by_operation"][operation] = {"success": 0, "failure": 0}

            status = "success" if report.get("success") else "failure"
            summary["by_type"][asset_type][status] += 1
            summary["by_operation"][operation][status] += 1

        return summary


# Global instance for convenience
_logger: Optional[ReportLogger] = None


def get_report_logger() -> ReportLogger:
    """Get the global report logger instance."""
    global _logger
    if _logger is None:
        _logger = ReportLogger()
    return _logger
