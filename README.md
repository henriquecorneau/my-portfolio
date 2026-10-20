# Henrique Corneau — Portfolio

Static site (HTML + CSS + vanilla JS, no build step). English by default, with an EN/PT switch.

## Run locally

```bash
python3 -m http.server 8000   # http://localhost:8000
```

## Edit content

All content lives in `data.js` (`main.js` only renders it):

- `projects` – cards on the home page **and** the content of each project page
  (summary, overview, how it works, highlights, gallery, links), in `en` and `pt`
- `papers`, `experience`, `education`
- `strings` – UI text and the bio for each language

Each project has its own page in `projects/<slug>.html`. To add a project, add an entry
to `projects` in `data.js`, then copy any page in `projects/` to `<new-slug>.html` and
change `data-project` and the `<title>`.

Images: card covers in `assets/img/projects/` (16:10 crops look best), extra pictures in
`assets/img/gallery/`, videos in `assets/video/`. A project without `cover` gets a
generated cover showing `coverLabel`.

## Deploy (GitHub Pages)

Push this folder to a repo and enable **Settings → Pages → Deploy from branch → main / root**.

## Computer Vision course (`course/`)

A 20-lesson course (English) served as static pages from `course/`. The HTML is generated and
committed, so deploying still needs no build step. To edit it:

- lesson text: `course/src/NN-*.md` (Markdown + `$math$`; `!fig name | caption` inserts `course/img/name.webp`)
- code: `course/code/NN_*.py`, one runnable script per lesson; pages include code sections from
  these scripts (`--8<-- "file.py:section"`), so page and code never diverge
- regenerate:

```bash
pip install markdown pymdown-extensions pygments -r course/code/requirements.txt
python course/build.py                    # pages only
python course/build.py --figures --zip    # also re-run every script (figures) and rebuild the zip
```
