from typer.testing import CliRunner

from polyalpha.cli import app


runner = CliRunner()


def test_topics_command() -> None:
    result = runner.invoke(app, ["topics"])
    assert result.exit_code == 0
    assert "google" in result.stdout


def test_export_unknown_topic_fails() -> None:
    result = runner.invoke(app, ["export", "--topic", "missing-topic"])
    assert result.exit_code != 0
