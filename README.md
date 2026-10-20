# Henrique Corneau — Portfolio

Static site (HTML + CSS + vanilla JS, no build step). English by default, with an EN/PT switch.

## Run locally

```bash
python3 -m http.server 8000   # http://localhost:8000
```

## Edit content

All content lives in `main.js`:

- `projects` – project cards (title/description in `en` and `pt`, image, stack, links, filter categories)
- `papers`, `experience`, `education`
- `strings` – UI text and the bio for each language

Project images go in `assets/img/projects/` (16:10 crops look best).

## Deploy (GitHub Pages)

Push this folder to a repo and enable **Settings → Pages → Deploy from branch → main / root**.
