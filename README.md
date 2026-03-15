# mtylr.com

Personal website built with [Jekyll](https://jekyllrb.com/) and [GitHub Pages](https://pages.github.com/).

**Live site:** [mtylr.com](https://mtylr.com)

## Local Development

### Prerequisites

- Ruby 3.x
- Bundler (`gem install bundler`)

### Setup

```bash
bundle config set --local path 'vendor/bundle'
bundle install
```

### Run locally

```bash
export PAGES_REPO_NWO="mike-taylor99/personal-website"
bundle exec jekyll serve
```

Then open [http://localhost:4000](http://localhost:4000).

> **Note:** GitHub metadata (repos, profile info) requires a `JEKYLL_GITHUB_TOKEN` environment variable set to a [personal access token](https://github.com/settings/tokens) with `public_repo` scope. Without it, the site still builds but some fields may be missing.

### Dev Container

This repo includes a [dev container](.devcontainer/devcontainer.json) configuration for VS Code / GitHub Codespaces with Ruby, Node.js, and all dependencies pre-configured.

## Project Structure

| Path | Description |
|------|-------------|
| `_config.yml` | Site configuration, social links, topics, welcome text |
| `_data/portfolio.yml` | Resume data (education, work, projects, awards, etc.) |
| `_data/social_media.yml` | Social media service definitions and icons |
| `_data/colors.json` | Language color mappings for repo cards |
| `_includes/` | Reusable HTML components (cards, sections, masthead) |
| `_layouts/` | Page templates (home, portfolio, post, default) |
| `_posts/` | Blog posts in Markdown |
| `_sass/` | SCSS partials (syntax highlighting) |
| `assets/` | Main stylesheet |

## License

See [LICENSE.txt](LICENSE.txt).
