(function () {
  'use strict';

  const { projects, papers, experience, education, strings } = window.SITE;
  const body = document.body;
  const BASE = body.dataset.base || '';          // '' on the home page, '../' on project pages
  const PAGE = body.dataset.page || 'home';

  // ---------- Helpers ----------
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const t = (v, lang) => (v && typeof v === 'object' && !Array.isArray(v) ? v[lang] : v);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const asset = (p) => BASE + p;
  const projectUrl = (slug) => `${BASE}projects/${slug}.html`;
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ } }
  };
  const ARROW = '↗';

  function coverHtml(p, lang, eager) {
    if (p.cover) {
      return `<img src="${asset(p.cover)}" alt="${esc(t(p.title, lang))}" ${eager ? '' : 'loading="lazy"'} width="960" height="600">`;
    }
    return `<div class="cover-gen" aria-hidden="true"><span class="mono">${esc(p.coverLabel || t(p.title, lang))}</span></div>`;
  }

  function mediaHtml(m, lang, cls) {
    if (m.type === 'video') {
      return `<video class="${cls || ''}" src="${asset(m.src)}" poster="${m.poster ? asset(m.poster) : ''}" muted loop playsinline autoplay preload="metadata" controls></video>`;
    }
    return `<img class="${cls || ''}" src="${asset(m.src)}" alt="${esc(t(m.caption, lang) || '')}" loading="lazy">`;
  }

  // ---------- Home page ----------
  let currentFilter = 'all';

  function renderProjects(lang) {
    const S = strings[lang];
    $('#project-grid').innerHTML = projects.map((p) => `
      <article class="card reveal" data-cats="${p.cats.join(' ')}">
        <a class="card-media" href="${projectUrl(p.slug)}" tabindex="-1" aria-hidden="true">${coverHtml(p, lang)}</a>
        <div class="card-body">
          <div class="card-meta"><span>${p.year}</span></div>
          <h3><a href="${projectUrl(p.slug)}">${esc(t(p.title, lang))}</a></h3>
          <p>${esc(t(p.summary, lang))}</p>
          <ul class="tags">${p.stack.slice(0, 4).map((s) => `<li>${esc(s)}</li>`).join('')}</ul>
          <div class="card-links">
            <a href="${projectUrl(p.slug)}" class="card-details">${esc(S['card.details'])} →</a>
            ${p.links.filter((l) => l.primary).map((l) => `<a href="${l.href}" target="_blank" rel="noopener">GitHub ${ARROW}</a>`).join('')}
          </div>
        </div>
      </article>`).join('');
    applyFilter();
  }

  function renderPapers() {
    $('#paper-list').innerHTML = papers.map((p) => `
      <li class="reveal">
        <span class="paper-year">${p.year}</span>
        <div>
          <a href="${p.href}" target="_blank" rel="noopener">${esc(p.title)} ${ARROW}</a>
          <div class="paper-authors">${esc(p.authors).replace('Henrique Corneau', '<strong>Henrique Corneau</strong>')}</div>
          <div class="paper-venue">${esc(p.venue)}</div>
        </div>
      </li>`).join('');
  }

  function renderTimeline(sel, items, lang) {
    $(sel).innerHTML = items.map((i) => `
      <li>
        <div class="tl-date">${esc(t(i.date, lang))}</div>
        <div class="tl-title">${esc(t(i.title, lang))}</div>
        <div class="tl-org">${esc(i.org)}</div>
        ${i.desc ? `<p class="tl-desc">${esc(t(i.desc, lang))}</p>` : ''}
      </li>`).join('');
  }

  function applyFilter() {
    $$('#project-grid .card').forEach((c) => {
      c.classList.toggle('hidden', currentFilter !== 'all' && !c.dataset.cats.split(' ').includes(currentFilter));
    });
  }

  function renderHome(lang) {
    renderProjects(lang);
    renderTimeline('#experience', experience, lang);
    renderTimeline('#education', education, lang);
  }

  // ---------- Project page ----------
  function section(title, inner) {
    return `<section class="detail-section reveal"><h2>${esc(title)}</h2>${inner}</section>`;
  }

  function renderDetail(lang) {
    const S = strings[lang];
    const slug = body.dataset.project;
    const i = projects.findIndex((p) => p.slug === slug);
    const root = $('#project');
    if (i < 0) { root.innerHTML = `<p>${esc(S['detail.notfound'])}</p>`; return; }
    const p = projects[i];
    const prev = projects[(i - 1 + projects.length) % projects.length];
    const next = projects[(i + 1) % projects.length];
    const paper = p.paper != null ? papers[p.paper] : null;
    const title = t(p.title, lang);

    document.title = `${title} · Henrique Corneau`;
    $('meta[name="description"]').setAttribute('content', t(p.summary, lang));

    const hero = p.hero ? mediaHtml(p.hero, lang, 'detail-hero-media') : coverHtml(p, lang, true);
    const steps = p.steps ? t(p.steps, lang) : null;
    const highlights = p.highlights ? t(p.highlights, lang) : null;
    const heroSrc = p.hero ? p.hero.src : p.cover;
    const heroIsImg = p.hero ? p.hero.type !== 'video' : Boolean(p.cover);
    const gallery = (p.gallery || []).filter((g) => g.src !== heroSrc);

    root.innerHTML = `
      <a class="back-link" href="${BASE}index.html#projects">${esc(S['detail.back'])}</a>
      <header class="detail-head">
        <div class="card-meta"><span>${p.year}</span><span>·</span><span>${p.cats.map((c) => esc(S['filters.' + c])).join(' · ')}</span></div>
        <h1>${esc(title)}</h1>
        <p class="lead">${esc(t(p.summary, lang))}</p>
        ${p.context ? `<p class="detail-context mono">${esc(t(p.context, lang))}</p>` : ''}
        <div class="detail-actions">
          ${p.links.map((l) => `<a class="${l.primary ? 'btn' : 'btn btn-ghost'}" href="${l.href}" target="_blank" rel="noopener">${esc(t(l.label, lang))} ${ARROW}</a>`).join('')}
          ${paper ? `<a class="btn btn-ghost" href="${paper.href}" target="_blank" rel="noopener">${esc(paper.venue)} ${ARROW}</a>` : ''}
        </div>
      </header>
      <div class="detail-hero${heroIsImg ? ' is-img' : ''}">${hero}</div>

      <div class="detail-grid">
        <div class="detail-main">
          ${section(S['detail.overview'], t(p.overview, lang).map((x) => `<p>${esc(x)}</p>`).join(''))}
          ${steps ? section(S['detail.how'], `<ol class="steps">${steps.map((x) => `<li>${esc(x)}</li>`).join('')}</ol>`) : ''}
          ${highlights ? section(S['detail.highlights'], `<ul class="checks">${highlights.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>`) : ''}
          ${gallery.length ? section(S['detail.gallery'], `<div class="gallery">${gallery.map((g) => `
            <figure>${mediaHtml(g, lang)}<figcaption>${esc(t(g.caption, lang))}</figcaption></figure>`).join('')}</div>`) : ''}
        </div>
        <aside class="detail-side">
          <div class="side-block">
            <h3 class="sub first">${esc(S['detail.stack'])}</h3>
            <ul class="tags">${p.stack.map((s) => `<li>${esc(s)}</li>`).join('')}</ul>
          </div>
          <div class="side-block">
            <h3 class="sub">${esc(S['detail.links'])}</h3>
            <ul class="side-links">${p.links.map((l) => `<li><a href="${l.href}" target="_blank" rel="noopener">${esc(t(l.label, lang))} ${ARROW}</a></li>`).join('')}</ul>
          </div>
          ${paper ? `<div class="side-block">
            <h3 class="sub">${esc(S['detail.paper'])}</h3>
            <a class="side-paper" href="${paper.href}" target="_blank" rel="noopener">${esc(paper.title)} ${ARROW}</a>
            <div class="paper-venue">${esc(paper.venue)} · ${paper.year}</div>
          </div>` : ''}
        </aside>
      </div>

      <nav class="pager" aria-label="Projects">
        <a href="${projectUrl(prev.slug)}"><span class="mono">${esc(S['detail.prev'])}</span><strong>← ${esc(t(prev.title, lang))}</strong></a>
        <a href="${projectUrl(next.slug)}" class="next"><span class="mono">${esc(S['detail.next'])}</span><strong>${esc(t(next.title, lang))} →</strong></a>
      </nav>`;
  }

  // ---------- Language ----------
  function setLang(lang, persist) {
    if (!strings[lang]) lang = 'en';
    const S = strings[lang];
    document.documentElement.lang = lang === 'pt' ? 'pt-BR' : 'en';
    $$('[data-i18n]').forEach((el) => { const v = S[el.dataset.i18n]; if (v != null) el.textContent = v; });
    $$('[data-i18n-html]').forEach((el) => { const v = S[el.dataset.i18nHtml]; if (v != null) el.innerHTML = v; });
    $$('[data-cv]').forEach((el) => { el.href = asset(S.cv); });
    $$('.lang-switch button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.lang === lang)));

    if (PAGE === 'project') {
      renderDetail(lang);
    } else {
      $('meta[name="description"]').setAttribute('content', S.metaDesc);
      renderHome(lang);
    }
    observeReveal();
    if (persist) store.set('lang', lang);
  }

  // ---------- Theme ----------
  function initTheme() {
    $('#theme-toggle').addEventListener('click', () => {
      const root = document.documentElement;
      const isDark = root.dataset.theme
        ? root.dataset.theme === 'dark'
        : window.matchMedia('(prefers-color-scheme: dark)').matches;
      root.dataset.theme = isDark ? 'light' : 'dark';
      store.set('theme', root.dataset.theme);
    });
  }

  // ---------- Reveal animation ----------
  let io;
  function observeReveal() {
    const els = $$('.reveal:not(.in)');
    if (!('IntersectionObserver' in window)) { els.forEach((e) => e.classList.add('in')); return; }
    io = io || new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
    }, { rootMargin: '0px 0px -40px 0px' });
    els.forEach((e) => io.observe(e));
  }

  // ---------- Init ----------
  $$('.lang-switch button').forEach((b) => b.addEventListener('click', () => setLang(b.dataset.lang, true)));
  if (PAGE === 'home') {
    $$('.chip').forEach((c) => c.addEventListener('click', () => {
      currentFilter = c.dataset.filter;
      $$('.chip').forEach((x) => x.classList.toggle('active', x === c));
      applyFilter();
    }));
    renderPapers();
  }
  $('#year').textContent = new Date().getFullYear();

  initTheme();
  const urlLang = new URLSearchParams(location.search).get('lang');
  setLang(urlLang || store.get('lang') || 'en', Boolean(urlLang));
})();
