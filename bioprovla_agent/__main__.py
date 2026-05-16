"""Allow: python -m bioprovla_agent (from directory that contains bioprovla_agent on PYTHONPATH)."""

from bioprovla_agent.run_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
