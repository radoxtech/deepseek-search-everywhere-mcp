# Contributing

Thanks for your interest in DeepSeek Search Everywhere MCP. Contributions of all sizes are welcome.

## Reporting bugs and requesting features

Open an [issue](https://github.com/radoxtech/deepseek-search-everywhere-mcp/issues). For bugs, include the
version, your client, what you expected, and steps to reproduce. Never paste API keys or tokens.

## Development setup

```bash
git clone https://github.com/radoxtech/deepseek-search-everywhere-mcp.git
cd deepseek-search-everywhere-mcp
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test,lint,http]" build twine
```

## Running checks

```bash
pytest -q                               # offline tests; no network or key needed
ruff check . && ruff format --check .   # lint and format
python -m build && twine check dist/*   # package build
```

Live tests (`pytest -m live`) need your own `DEEPSEEK_API_KEY` and are optional.

## Commits and pull requests

Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages and PR titles, for example
`fix: handle empty search results`. Releases and the changelog are generated from them. Keep each commit to one
logical change and make sure checks pass. Details are in [AGENTS.md](AGENTS.md).

## Security issues

Do not open a public issue. Follow [SECURITY.md](SECURITY.md).

## License

This project is MIT licensed. By contributing, you agree that your contributions are licensed under the same
license.
