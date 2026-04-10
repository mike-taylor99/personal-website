# Personal Website — Copilot Instructions

## Local Development

This is a Jekyll site using GitHub Pages. To run locally:

```bash
PAGES_REPO_NWO="mike-taylor99/personal-website" bundle exec jekyll serve --host 0.0.0.0 --port 4000 --config _config.yml,_config_dev.yml
```

The `_config_dev.yml` override removes `jekyll-github-metadata` from the plugins list, avoiding 401 errors from the VS Code credential helper without needing to block `api.github.com`. GitHub metadata warnings in the output are harmless — the site renders correctly without it.

Install dependencies first if needed:

```bash
bundle install
```
