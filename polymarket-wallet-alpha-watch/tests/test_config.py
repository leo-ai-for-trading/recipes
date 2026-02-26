from pathlib import Path

from polyalpha.config import get_topic_or_raise, load_topics


def test_load_google_topic() -> None:
    config = load_topics(Path("config/topics.yaml"))
    topic = get_topic_or_raise(config, "google")
    assert "google" in topic.keywords[0].lower()
