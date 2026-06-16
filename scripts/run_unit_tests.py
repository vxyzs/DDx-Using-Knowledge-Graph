import datetime
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class MarkdownTestResult(unittest.TestResult):
    """
    Custom TestResult collector that builds a Markdown report database.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.results.append({
            "name": test.id(),
            "status": "SUCCESS",
            "message": ""
        })

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.results.append({
            "name": test.id(),
            "status": "FAIL",
            "message": self._exc_info_to_string(err, test)
        })

    def addError(self, test, err):
        super().addError(test, err)
        self.results.append({
            "name": test.id(),
            "status": "ERROR",
            "message": self._exc_info_to_string(err, test)
        })


def main():
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="core/tests", pattern="test_*.py")

    runner = unittest.TextTestRunner(resultclass=MarkdownTestResult)
    result = runner.run(suite)

    os.makedirs("results", exist_ok=True)
    report_path = "results/test_report.md"

    total_run = result.testsRun
    failures_count = len(result.failures)
    errors_count = len(result.errors)
    success_count = total_run - failures_count - errors_count

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Unit Test Execution Report\n\n")
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"**Execution Date**: {timestamp}\n\n")
        
        f.write("## Summary\n\n")
        f.write("| Metric | Count |\n")
        f.write("| --- | --- |\n")
        f.write(f"| **Total Tests** | {total_run} |\n")
        f.write(f"| **Passed** | {success_count} |\n")
        f.write(f"| **Failed** | {failures_count} |\n")
        f.write(f"| **Errors** | {errors_count} |\n\n")

        f.write("## Test Case Details\n\n")
        f.write("| Test ID | Status | Message |\n")
        f.write("| --- | --- | --- |\n")

        for r in result.results:
            status_emoji = "✅ SUCCESS"
            if r["status"] == "FAIL":
                status_emoji = "❌ FAIL"
            elif r["status"] == "ERROR":
                status_emoji = "⚠️ ERROR"

            msg = "-"
            if r["message"]:
                msg = (
                    r["message"]
                    .replace("\n", "<br>")
                    .replace("|", "\\|")
                )
            f.write(f"| `{r['name']}` | {status_emoji} | {msg} |\n")

    print(f"\nTest report successfully generated at: {report_path}")

    if not result.wasSuccessful():
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
