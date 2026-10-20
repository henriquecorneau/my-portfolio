(function () {
  'use strict';
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const store = {
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ } }
  };

  // Theme (same behaviour as the rest of the site)
  $('#theme-toggle').addEventListener('click', () => {
    const root = document.documentElement;
    const isDark = root.dataset.theme
      ? root.dataset.theme === 'dark'
      : window.matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = isDark ? 'light' : 'dark';
    store.set('theme', root.dataset.theme);
  });
  $('#year').textContent = new Date().getFullYear();

  // Lesson list drawer on small screens
  const menu = $('#menu-toggle');
  if (menu) {
    const setOpen = (open) => {
      document.body.classList.toggle('side-open', open);
      menu.setAttribute('aria-expanded', String(open));
    };
    menu.addEventListener('click', () => setOpen(!document.body.classList.contains('side-open')));
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') setOpen(false); });
    document.addEventListener('click', (e) => {
      if (document.body.classList.contains('side-open') && !e.target.closest('.course-side, #menu-toggle')) setOpen(false);
    });
    const active = $('.course-side a.active');
    if (active) active.scrollIntoView({ block: 'center' });
  }

  // Copy buttons on code blocks
  $$('.prose .highlight').forEach((block) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'copy-btn';
    btn.textContent = 'copy';
    btn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(block.querySelector('pre').innerText);
        btn.textContent = 'copied';
      } catch (e) {
        btn.textContent = 'press Ctrl+C';
      }
      setTimeout(() => { btn.textContent = 'copy'; }, 1500);
    });
    block.appendChild(btn);
  });

  // Highlight the current section in "On this page"
  const tocLinks = $$('.lesson-toc a');
  if (tocLinks.length && 'IntersectionObserver' in window) {
    const byId = new Map(tocLinks.map((a) => [decodeURIComponent(a.hash.slice(1)), a]));
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          tocLinks.forEach((a) => a.classList.remove('active'));
          const a = byId.get(e.target.id);
          if (a) a.classList.add('active');
        }
      });
    }, { rootMargin: '-80px 0px -70% 0px' });
    $$('.prose h2[id], .prose h3[id]').forEach((h) => io.observe(h));
  }

  // Math (KaTeX is loaded with defer before this script)
  if (window.renderMathInElement) {
    $$('.arithmatex').forEach((el) => window.renderMathInElement(el, {
      delimiters: [
        { left: '\\[', right: '\\]', display: true },
        { left: '\\(', right: '\\)', display: false }
      ],
      throwOnError: false
    }));
  }
})();
