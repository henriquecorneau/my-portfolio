// Smooth scroll para links âncora
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const href = a.getAttribute('href');
    if(href.length>1){
      e.preventDefault();
      document.querySelector(href).scrollIntoView({behavior:'smooth', block:'start'});
    }
  })
});

// Exemplo de adição dinâmica de projetos
const projects = [
  {title:'Medição estéreo', desc:'Estimativa de distâncias usando câmera estéreo e calibração.'},
  {title:'Filtro IIR em Octave', desc:'Projeto e implementação de filtros para remoção de ruído.'}
];
const grid = document.getElementById('projectsGrid');
projects.forEach(p => {
  const el = document.createElement('article'); el.className='proj';
  el.innerHTML = `<h3>${p.title}</h3><p>${p.desc}</p>`;
  grid.appendChild(el);
});