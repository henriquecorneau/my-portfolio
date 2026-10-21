"""Build the course pages: course/src/*.md -> course/*.html.

The site itself has no build step; this script only regenerates the HTML that is
committed next to it. Code blocks are pulled from the tested scripts in code/
(`--8<-- "file.py:section"`), so the pages always show the code that actually runs.

    pip install markdown pymdown-extensions pygments
    python course/build.py                 # pages only
    python course/build.py --figures       # also re-run every lesson script to refresh img/
    python course/build.py --zip           # also pack code/ into cv-course-code.zip
"""
import argparse
import html
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
SRC, CODE, IMG = ROOT / "src", ROOT / "code", ROOT / "img"

PARTS = [
    ("I", "Foundations", ["00-setup", "01-images-as-arrays", "02-sampling-quantization", "03-geometric-transforms"]),
    ("II", "Enhancement", ["04-histograms-lut", "05-noise-filtering"]),
    ("III", "Edges, shapes and colour", ["06-edges", "07-canny", "08-hough", "09-morphology", "10-color"]),
    ("IV", "Features", ["11-corners-tracking", "12-keypoints-orb-sift", "13-homography-ransac"]),
    ("V", "Geometry and 3-D", ["14-camera-calibration", "15-stereo-vision", "16-structure-from-motion",
                               "17-visual-slam"]),
    ("VI", "Case studies", ["18-ct-lungs", "19-fingerprints"]),
]
ORDER = [slug for _, _, slugs in PARTS for slug in slugs]


def read_lesson(slug):
    text = (SRC / f"{slug}.md").read_text()
    meta, body = {}, text
    if text.startswith("---"):
        _, head, body = text.split("---", 2)
        for line in head.strip().splitlines():
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip("\"")
    return meta, body


def figures(body):
    """`!fig name | caption` -> <figure>. Images live in img/<name>.webp."""
    def repl(m):
        name, cap = m.group(1).strip(), m.group(2).strip()
        return (f'<figure class="fig"><a href="img/{name}.webp" target="_blank" rel="noopener">'
                f'<img src="img/{name}.webp" alt="{html.escape(cap)}" loading="lazy"></a>'
                f'<figcaption markdown="span">{cap}</figcaption></figure>')
    return re.sub(r"^!fig\s+([^|\n]+)\|(.+)$", repl, body, flags=re.M)


def render(body):
    md = markdown.Markdown(extensions=[
        "tables", "attr_list", "md_in_html", "admonition", "toc", "footnotes",
        "pymdownx.superfences", "pymdownx.highlight", "pymdownx.inlinehilite",
        "pymdownx.arithmatex", "pymdownx.snippets", "pymdownx.details", "pymdownx.betterem",
    ], extension_configs={
        "toc": {"permalink": "#", "toc_depth": "2-3"},
        "pymdownx.highlight": {"css_class": "highlight", "guess_lang": False},
        "pymdownx.arithmatex": {"generic": True},
        "pymdownx.snippets": {"base_path": [str(CODE)], "check_paths": True, "dedent_subsections": True},
    })
    out = md.convert(figures(body))
    return out, md.toc_tokens


def toc_html(tokens):
    items = []
    for t in tokens:
        items.append(f'<li><a href="#{t["id"]}">{t["name"]}</a></li>')
        for c in t.get("children", []):
            items.append(f'<li class="sub"><a href="#{c["id"]}">{c["name"]}</a></li>')
    return "\n".join(items)


def sidebar(current):
    out = []
    for num, name, slugs in PARTS:
        out.append(f'<div class="side-part"><span class="mono">Part {num}</span> {name}</div><ol>')
        for slug in slugs:
            meta, _ = read_lesson(slug)
            cls = ' class="active" aria-current="page"' if slug == current else ""
            out.append(f'<li><a href="{slug}.html"{cls}><span class="mono">{slug[:2]}</span>'
                       f'{html.escape(meta["title"])}</a></li>')
        out.append("</ol>")
    return "\n".join(out)


HEAD = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <meta name="description" content="{desc}">
  <meta property="og:title" content="{title}">
  <meta property="og:description" content="{desc}">
  <meta property="og:image" content="{og}">
  <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='20' fill='%23111'/><text x='50' y='66' font-size='46' font-family='monospace' fill='%23fff' text-anchor='middle'>HC</text></svg>">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
  <link rel="stylesheet" href="../style.css">
  <link rel="stylesheet" href="course.css">
  <script>
    try {{ var t = localStorage.getItem('theme'); if (t) document.documentElement.dataset.theme = t; }} catch (e) {{}}
  </script>
</head>
<body data-page="course">
  <header class="site-header">
    <div class="container header-inner wide">
      <a href="../index.html" class="logo">henrique<span>.corneau</span></a>
      <nav class="nav" aria-label="Main">
        <a href="../index.html#projects">Projects</a>
        <a href="index.html" class="current">CV Course</a>
        <a href="../ml-course/index.html">ML Course</a>
        <a href="../index.html#papers">Papers</a>
        <a href="../index.html#about">About</a>
      </nav>
      <div class="header-actions">
        {menu}
        <button type="button" class="icon-btn" id="theme-toggle" aria-label="Toggle theme">
          <svg class="i-sun" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
          <svg class="i-moon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>
        </button>
      </div>
    </div>
  </header>
"""

MENU_BTN = """<button type="button" class="icon-btn menu-btn" id="menu-toggle" aria-label="Lessons" aria-expanded="false" aria-controls="course-side">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 6h16M4 12h16M4 18h16"/></svg>
        </button>"""

FOOT = """
  <footer class="site-footer">
    <div class="container footer-inner mono wide">
      <span>© <span id="year"></span> Henrique Corneau · course text and code under MIT</span>
      <span><a href="https://github.com/henriquecorneau" target="_blank" rel="noopener">GitHub</a> · <a href="https://www.linkedin.com/in/henriquecorneau/" target="_blank" rel="noopener">LinkedIn</a></span>
    </div>
  </footer>
  <script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
  <script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
  <script defer src="course.js"></script>
</body>
</html>
"""


def lesson_page(slug):
    meta, body = read_lesson(slug)
    content, toc = render(body)
    i = ORDER.index(slug)
    prev_ = ORDER[i - 1] if i > 0 else None
    next_ = ORDER[i + 1] if i + 1 < len(ORDER) else None
    part = next((f"Part {n} · {name}" for n, name, s in PARTS if slug in s), "")

    def pager_link(s, label, cls=""):
        if s is None:
            return '<a href="index.html"' + (f' class="{cls}"' if cls else "") + \
                   f'><span class="mono">{label}</span><strong>Course overview</strong></a>'
        m, _ = read_lesson(s)
        return (f'<a href="{s}.html"' + (f' class="{cls}"' if cls else "") +
                f'><span class="mono">{label}</span><strong>{s[:2]} · {html.escape(m["title"])}</strong></a>')

    script = meta.get("script")
    run = ""
    if script:
        run = (f'<div class="run-box"><div><span class="mono label">Run this lesson</span>'
               f'<code>python {script}</code><code>python {script} --save out</code></div>'
               f'<a class="btn btn-ghost" href="code/{script}" download>Download {script}</a></div>')
    sources = meta.get("sources", "")
    title = f'{slug[:2]} · {meta["title"]} · Computer Vision Course'
    page = HEAD.format(title=html.escape(title), desc=html.escape(meta.get("summary", "")),
                       og="../assets/img/profile.jpg", menu=MENU_BTN)
    page += f"""
  <div class="course-layout container wide">
    <aside class="course-side" id="course-side" aria-label="Lessons">
      <a class="side-home" href="index.html">← Course overview</a>
      {sidebar(slug)}
    </aside>
    <main class="lesson" id="top">
      <header class="lesson-head">
        <p class="eyebrow mono">{part} · Lesson {slug[:2]}</p>
        <h1>{html.escape(meta["title"])}</h1>
        <p class="lead">{meta.get("summary", "")}</p>
        {f'<p class="lesson-sources mono">Based on: {sources}</p>' if sources else ''}
        {run}
      </header>
      <article class="prose">
{content}
      </article>
      <nav class="pager" aria-label="Lessons">
        {pager_link(prev_, "Previous")}
        {pager_link(next_, "Next", "next")}
      </nav>
    </main>
    <aside class="lesson-toc" aria-label="On this page">
      <div class="sub first">On this page</div>
      <ul>{toc_html(toc)}</ul>
    </aside>
  </div>
"""
    return page + FOOT


def index_page():
    meta, body = read_lesson("index")
    content, _ = render(body)
    parts = []
    for num, name, slugs in PARTS:
        cards = []
        for slug in slugs:
            m, _ = read_lesson(slug)
            cards.append(f"""<a class="lesson-card" href="{slug}.html">
          <span class="mono num">{slug[:2]}</span>
          <strong>{html.escape(m["title"])}</strong>
          <span class="desc">{m.get("summary", "")}</span>
          <span class="tags-line mono">{html.escape(m.get("topics", ""))}</span>
        </a>""")
        parts.append(f"""<section class="part">
      <h2><span class="mono">Part {num}</span> {name}</h2>
      <div class="lesson-grid">{''.join(cards)}</div>
    </section>""")
    page = HEAD.format(title="Computer Vision Course · Henrique Corneau",
                       desc=html.escape(meta["summary"]), og="img/16-reconstruction.webp", menu="")
    page += f"""
  <main class="container course-index">
    <section class="course-hero">
      <p class="eyebrow mono">Free course · Python · OpenCV · NumPy</p>
      <h1>{html.escape(meta["title"])}</h1>
      <p class="lead">{meta["summary"]}</p>
      <div class="detail-actions">
        <a class="btn" href="00-setup.html">Start with lesson 00 →</a>
        <a class="btn btn-ghost" href="cv-course-code.zip" download>Download all code + data (zip)</a>
      </div>
      <figure class="hero-strip">
        <img src="img/07-steps.webp" alt="Canny edge detector, step by step" loading="eager">
      </figure>
    </section>
    <article class="prose intro">{content}</article>
    {''.join(parts)}
  </main>
"""
    return page + FOOT


def refresh_figures():
    for script in sorted(CODE.glob("[0-9][0-9]_*.py")):
        print(f"running {script.name}")
        subprocess.run([sys.executable, script.name, "--save", str(IMG)], cwd=CODE, check=True,
                       stdout=subprocess.DEVNULL)


def make_zip():
    out = ROOT / "cv-course-code.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(CODE.rglob("*")):
            rel = p.relative_to(CODE)
            if p.is_file() and "__pycache__" not in rel.parts and rel.parts[0] not in ("out",) \
                    and rel.name != "camera_calibration.yaml":
                z.write(p, Path("cv-course") / rel)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()
    if args.figures:
        refresh_figures()
    for slug in ORDER:
        (ROOT / f"{slug}.html").write_text(lesson_page(slug))
    (ROOT / "index.html").write_text(index_page())
    print(f"built {len(ORDER)} lessons + index")
    if args.zip:
        make_zip()


if __name__ == "__main__":
    main()
