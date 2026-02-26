from typer.testing import CliRunner

from polyalpha.cli import app


runner = CliRunner()


def test_topics_command() -> None:
    result = runner.invoke(app, ["topics"])
    assert result.exit_code == 0
    assert "google" in result.stdout
