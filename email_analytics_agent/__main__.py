"""Allow ``python -m email_analytics_agent`` to run the demonstration CLI."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())

