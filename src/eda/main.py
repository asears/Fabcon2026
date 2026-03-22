"""Legacy entry point — delegates to the eda CLI package.

Prefer: uv run eda <command>
Or:     uv run python -m eda.main <command>
"""
from eda.main import cli

if __name__ == "__main__":
    cli()
