(function () {
  'use strict';

  const GH = 'https://github.com/henriquecorneau/';

  // ---------- Content ----------
  // Each project: tags drive the filter chips; text is { en, pt }.
  const projects = [
    {
      id: 'drone-map-loc',
      year: 2026,
      cats: ['drones', 'ros', 'vision'],
      img: 'assets/img/projects/drone-map-loc.jpg',
      title: { en: 'Drone Map Localization', pt: 'Localização de Drone em Mapa' },
      desc: {
        en: 'Builds an ORB-SLAM3 map from an RGB-D camera and localizes a monocular drone camera against the saved atlas — groundwork for autonomous navigation with Nav2.',
        pt: 'Constrói um mapa com ORB-SLAM3 a partir de uma câmera RGB-D e localiza a câmera monocular de um drone no atlas salvo — base para navegação autônoma com Nav2.'
      },
      stack: ['ROS 2', 'ORB-SLAM3', 'C++', 'Docker'],
      links: [
        { label: 'GitHub', href: GH + 'drone-map-loc' },
        { label: 'ORB-SLAM3 + ROS 2 Docker', href: GH + 'orbslam3-ros2-docker' }
      ]
    },
    {
      id: 'boat-vision',
      year: 2025,
      cats: ['vision'],
      img: 'assets/img/projects/boat-vision.jpg',
      title: { en: 'Boat Vision', pt: 'Boat Vision' },
      desc: {
        en: 'Perception for an autonomous robotic boat: YOLOv11 detection, stereo disparity maps and IMU + GPS fusion to estimate the global coordinates of detected objects.',
        pt: 'Percepção para um barco robótico autônomo: detecção com YOLOv11, mapas de disparidade estéreo e fusão de IMU + GPS para estimar as coordenadas globais dos objetos detectados.'
      },
      stack: ['Python', 'YOLOv11', 'Stereo Vision', 'GPS/IMU'],
      links: [{ label: 'GitHub', href: GH + 'boat-vision' }]
    },
    {
      id: 'slam-kinect',
      year: 2026,
      cats: ['ros', 'vision'],
      img: 'assets/img/projects/slam-kinect.jpg',
      title: { en: 'SLAM with Kinect', pt: 'SLAM com Kinect' },
      desc: {
        en: 'RGB-D SLAM using a Kinect v1 and ORB-SLAM3 inside a ROS 2 node, with a Dockerized libfreenect driver that publishes image and depth topics.',
        pt: 'SLAM RGB-D usando um Kinect v1 e ORB-SLAM3 em um nó ROS 2, com um driver libfreenect em Docker que publica os tópicos de imagem e profundidade.'
      },
      stack: ['ROS 2', 'ORB-SLAM3', 'libfreenect', 'Docker'],
      links: [
        { label: 'GitHub', href: GH + 'kinect-orbslam' },
        { label: 'kinect-ros2', href: GH + 'kinect-ros2' }
      ]
    },
    {
      id: 'tello-tracking',
      year: 2024,
      cats: ['drones', 'vision'],
      img: 'assets/img/projects/tello-tracking.jpg',
      title: { en: 'Tello Tracking', pt: 'Tello Tracking' },
      desc: {
        en: 'Real-time object tracking with a DJI Tello drone, combining YOLO detections with PID control to keep the target centered.',
        pt: 'Rastreamento de objetos em tempo real com um drone DJI Tello, combinando detecções do YOLO com controle PID para manter o alvo centralizado.'
      },
      stack: ['Python', 'YOLO', 'PID', 'DJI Tello'],
      links: [{ label: { en: 'Paper', pt: 'Artigo' }, href: 'https://www.even3.com.br/anais/cobicet2024/884110-drone-based-object-tracking-with-convolutional-neural-networks/' }]
    },
    {
      id: 'tello-hand',
      year: 2025,
      cats: ['drones', 'vision'],
      img: 'assets/img/projects/tello-hand.jpg',
      title: { en: 'Tello Hand Control', pt: 'Controle do Tello por Gestos' },
      desc: {
        en: 'Real-time hand gesture recognition with a neural network classifier, used to fly a drone with simple hand commands.',
        pt: 'Reconhecimento de gestos manuais em tempo real com um classificador de rede neural, usado para pilotar um drone com comandos simples.'
      },
      stack: ['Python', 'Neural Networks', 'OpenCV', 'DJI Tello'],
      links: [{ label: { en: 'Paper', pt: 'Artigo' }, href: 'https://periodicos.ufersa.edu.br/ecop/article/view/14093' }]
    },
    {
      id: 'logistics-robot',
      year: 2025,
      cats: ['ros'],
      img: 'assets/img/projects/logistics-robot.jpg',
      title: { en: 'Logistics Robot', pt: 'Robô Logístico' },
      desc: {
        en: 'LIDAR mapping with Cartographer and autonomous navigation with Nav2 on a TurtleBot3, plus a custom obstacle-avoidance node, simulated in Gazebo.',
        pt: 'Mapeamento com LIDAR via Cartographer e navegação autônoma com Nav2 em um TurtleBot3, além de um nó próprio de desvio de obstáculos, simulado no Gazebo.'
      },
      stack: ['ROS 2', 'Nav2', 'LIDAR', 'Gazebo'],
      links: [{ label: 'GitHub', href: GH + 'logistics-robot' }]
    },
    {
      id: 'rasp-tello',
      year: 2025,
      cats: ['drones', 'ros', 'embedded'],
      img: 'assets/img/projects/rasp-tello.jpg',
      title: { en: 'Rasp-Tello ROS', pt: 'Rasp-Tello ROS' },
      desc: {
        en: 'HTTP-to-ROS 2 bridge: a web page served by a Raspberry Pi Pico W sends commands through micro-ROS to control a DJI Tello drone.',
        pt: 'Ponte HTTP–ROS 2: uma página web servida por uma Raspberry Pi Pico W envia comandos via micro-ROS para controlar um drone DJI Tello.'
      },
      stack: ['C', 'micro-ROS', 'Pico W', 'ROS 2'],
      links: [{ label: 'GitHub', href: GH + 'rasp_ros_iot' }]
    },
    {
      id: 'arduino-drone',
      year: 2025,
      cats: ['drones', 'embedded'],
      img: 'assets/img/projects/arduino-drone.jpg',
      title: { en: 'Arduino Drone', pt: 'Drone com Arduino' },
      desc: {
        en: 'A drone flight controller written in bare-metal AVR-C on top of FreeRTOS, flown with a standard RC transmitter.',
        pt: 'Controladora de voo de drone escrita em AVR-C bare-metal sobre FreeRTOS, pilotada com um rádio controle (RC) convencional.'
      },
      stack: ['AVR-C', 'FreeRTOS', 'Arduino', 'RC'],
      links: [{ label: 'GitHub', href: GH + 'nano-drone' }]
    },
    {
      id: 'tank-secure',
      year: 2025,
      cats: ['embedded'],
      img: 'assets/img/projects/tank-secure.jpg',
      title: { en: 'Tank Secure', pt: 'Tank Secure' },
      desc: {
        en: 'Water-tank level and temperature monitoring on a Raspberry Pi Pico W, with RFID authentication for the OLED display and telemetry sent to ThingSpeak.',
        pt: 'Monitoramento de nível e temperatura de reservatório com Raspberry Pi Pico W, autenticação RFID para o display OLED e telemetria enviada ao ThingSpeak.'
      },
      stack: ['C', 'Pico SDK', 'RFID', 'ThingSpeak'],
      links: [{ label: 'GitHub', href: GH + 'tank-secure' }]
    },
    {
      id: 'robotic-vision',
      year: 2025,
      cats: ['vision'],
      img: 'assets/img/projects/robotic-vision.jpg',
      title: { en: 'Robotic Vision Library', pt: 'Biblioteca de Visão Robótica' },
      desc: {
        en: 'Computer vision techniques implemented from scratch for a graduate Robotic Vision course — an in-depth study of the algorithms behind OpenCV.',
        pt: 'Técnicas de visão computacional implementadas do zero na disciplina de pós-graduação em Visão Robótica — um estudo aprofundado dos algoritmos por trás do OpenCV.'
      },
      stack: ['Python', 'NumPy', 'OpenCV', 'Jupyter'],
      links: [{ label: 'GitHub', href: GH + 'visao_robotica' }]
    }
  ];

  const papers = [
    {
      year: 2025,
      title: 'Drone Controlado por Gestos Manuais Classificados por Rede Neural',
      authors: 'Henrique Corneau, Ramon Souza, João Victor de Andrade, Bruno Silva',
      venue: 'ECOP 2025',
      href: 'https://periodicos.ufersa.edu.br/ecop/article/view/14093'
    },
    {
      year: 2025,
      title: 'Drone Control by Chatbot with Artificial Intelligence and Visual Analysis',
      authors: 'Murilo Barros, Henrique Corneau, Ramon Souza, Vinícius Silva, Helton Maia, Bruno Silva',
      venue: 'COBICET 2025',
      href: 'https://www.even3.com.br/anais/congresso-brasileiro-interdisciplinar-ciencia-tecnologia/1198237-drone-control-by-chatbot-with-artificial-intelligence-and-visual-analysis/'
    },
    {
      year: 2024,
      title: 'Drone-Based Object Tracking with Convolutional Neural Networks',
      authors: 'Ramon Souza, Henrique Corneau, Murilo Barros, Vinícius Silva, César Melo, Helton Maia',
      venue: 'COBICET 2024',
      href: 'https://www.even3.com.br/anais/cobicet2024/884110-drone-based-object-tracking-with-convolutional-neural-networks/'
    }
  ];

  const experience = [
    {
      date: '2025',
      title: { en: 'Development Intern', pt: 'Estagiário de Desenvolvimento' },
      org: 'EMBARCATECH · IFRN',
      desc: {
        en: 'Designed and developed embedded and IoT products for companies, using microcontrollers and a variety of sensors.',
        pt: 'Concepção e desenvolvimento de produtos e aplicações de Sistemas Embarcados e IoT para empresas, usando microcontroladores e diversos sensores.'
      }
    },
    {
      date: '2024 – 2025',
      title: { en: 'Undergraduate Researcher (PIBIC)', pt: 'Iniciação Científica (PIBIC)' },
      org: 'UFRN',
      desc: {
        en: 'Developed, implemented and tested robotic perception techniques for visual navigation on an autonomous sailboat.',
        pt: 'Desenvolvimento, implementação e testes de técnicas de percepção robótica para navegação visual aplicadas a um veleiro autônomo.'
      }
    },
    {
      date: '2023 – 2025',
      title: { en: 'Founder · Autonomous Drone Team', pt: 'Fundador · Equipe de Drones Autônomos' },
      org: 'UFRN',
      desc: {
        en: 'Built and programmed drones from scratch to perform autonomous tasks using computer vision, AI and ROS. Authored and co-authored papers at COBICET 2024, ECOP 2025 and COBICET 2025.',
        pt: 'Construção e programação de drones do zero para tarefas autônomas usando visão computacional, IA e ROS. Autor e coautor de artigos no COBICET 2024, ECOP 2025 e COBICET 2025.'
      }
    },
    {
      date: '2022',
      title: { en: 'Junior Company Member', pt: 'Membro de Empresa Júnior' },
      org: 'Include Engenharia · UFRN',
      desc: {
        en: 'Developed web and mobile applications in React for corporate clients.',
        pt: 'Desenvolvimento de aplicações web e mobile em React para clientes corporativos.'
      }
    }
  ];

  const education = [
    {
      date: { en: '2025 – Present', pt: '2025 – Atual' },
      title: { en: 'Computer Science & Information Technology (exchange)', pt: 'Engenharia de Computação e Tecnologias da Informação (intercâmbio)' },
      org: 'INSA Rouen Normandie · France',
      desc: { en: '4th year — Génie Informatique et Technologies de l\'Information.', pt: '4º ano — Génie Informatique et Technologies de l\'Information.' }
    },
    {
      date: { en: '2022 – Present', pt: '2022 – Atual' },
      title: { en: 'B.Eng. Mechatronics Engineering', pt: 'Engenharia Mecatrônica' },
      org: 'Universidade Federal do Rio Grande do Norte · Brazil'
    },
    {
      date: '2018',
      title: { en: 'Technical High School in Building Construction', pt: 'Ensino Médio Técnico em Edificações' },
      org: 'Instituto Federal da Bahia · Barreiras'
    }
  ];

  // ---------- UI strings ----------
  const strings = {
    en: {
      'nav.projects': 'Projects', 'nav.papers': 'Papers', 'nav.about': 'About', 'nav.contact': 'Contact',
      'hero.eyebrow': 'Robotics · Computer Vision · Embedded Systems',
      'hero.title': "Hi, I'm Henrique!",
      'hero.lead': 'Mechatronics engineering student at <strong>UFRN</strong> (Brazil), currently on exchange at <strong>INSA Rouen Normandie</strong> (France). I build autonomous drones, robot perception pipelines and embedded systems — and I document my projects here.',
      'hero.status': 'Looking for an internship in Computer Vision, Robotics, Embedded Systems or AI.',
      'hero.cv': 'Resume',
      'projects.title': 'Featured Projects', 'projects.all': 'See all on GitHub →',
      'filters.all': 'All', 'filters.drones': 'Drones', 'filters.vision': 'Computer Vision', 'filters.ros': 'ROS 2 / SLAM', 'filters.embedded': 'Embedded / IoT',
      'papers.title': 'Publications',
      'about.title': 'About',
      'about.p1': "I'm a mechatronics engineering student at the Universidade Federal do Rio Grande do Norte (UFRN), passionate about robotics, embedded systems and computer vision.",
      'about.p2': 'During my undergraduate studies I founded a robotics team focused on autonomous drones: we built them from scratch and programmed them to perform autonomous tasks using AI, computer vision and ROS.',
      'about.p3': 'As a scientific researcher (PIBIC), I worked on robotic perception for an autonomous sailboat, fusing YOLO detections, stereo vision, IMU and GPS to estimate the global coordinates of detected objects. I also took part in EMBARCATECH, a technology residency where our team developed embedded and IoT solutions for companies.',
      'about.p4': "Today I'm an exchange student in Computer Science and Information Technology at INSA Rouen Normandie, in France.",
      'about.skills': 'Skills', 'about.languages': 'Languages', 'about.experience': 'Experience', 'about.education': 'Education',
      'lang.pt': 'Portuguese', 'lang.en': 'English', 'lang.fr': 'French', 'lang.es': 'Spanish', 'lang.native': 'native',
      'contact.title': "Let's talk",
      'contact.text': 'Open to internships and collaborations in robotics, computer vision and embedded systems.',
      cv: 'assets/cv_Henrique_CORNEAU_en.pdf',
      metaDesc: 'Henrique Corneau — Mechatronics engineering student working on robotics, computer vision and embedded systems.'
    },
    pt: {
      'nav.projects': 'Projetos', 'nav.papers': 'Artigos', 'nav.about': 'Sobre', 'nav.contact': 'Contato',
      'hero.eyebrow': 'Robótica · Visão Computacional · Sistemas Embarcados',
      'hero.title': 'Olá, eu sou o Henrique!',
      'hero.lead': 'Estudante de Engenharia Mecatrônica na <strong>UFRN</strong> (Brasil), atualmente em intercâmbio no <strong>INSA Rouen Normandie</strong> (França). Desenvolvo drones autônomos, sistemas de percepção robótica e sistemas embarcados — e documento meus projetos aqui.',
      'hero.status': 'Em busca de estágio em Visão Computacional, Robótica, Sistemas Embarcados ou IA.',
      'hero.cv': 'Currículo',
      'projects.title': 'Projetos em Destaque', 'projects.all': 'Ver todos no GitHub →',
      'filters.all': 'Todos', 'filters.drones': 'Drones', 'filters.vision': 'Visão Computacional', 'filters.ros': 'ROS 2 / SLAM', 'filters.embedded': 'Embarcados / IoT',
      'papers.title': 'Publicações',
      'about.title': 'Sobre',
      'about.p1': 'Sou estudante de Engenharia Mecatrônica na Universidade Federal do Rio Grande do Norte (UFRN), apaixonado por robótica, sistemas embarcados e visão computacional.',
      'about.p2': 'Durante a graduação, fundei uma equipe de robótica focada em drones autônomos: construímos os drones do zero e os programamos para realizar tarefas autônomas usando IA, visão computacional e ROS.',
      'about.p3': 'Como pesquisador de iniciação científica (PIBIC), trabalhei com percepção robótica para um veleiro autônomo, fundindo detecções do YOLO, visão estéreo, IMU e GPS para estimar as coordenadas globais de objetos detectados. Também participei do EMBARCATECH, uma residência tecnológica em que nossa equipe desenvolveu soluções embarcadas e de IoT para empresas.',
      'about.p4': 'Hoje sou intercambista em Engenharia de Computação e Tecnologias da Informação no INSA Rouen Normandie, na França.',
      'about.skills': 'Habilidades', 'about.languages': 'Idiomas', 'about.experience': 'Experiência', 'about.education': 'Formação',
      'lang.pt': 'Português', 'lang.en': 'Inglês', 'lang.fr': 'Francês', 'lang.es': 'Espanhol', 'lang.native': 'nativo',
      'contact.title': 'Vamos conversar',
      'contact.text': 'Aberto a estágios e colaborações em robótica, visão computacional e sistemas embarcados.',
      cv: 'assets/cv_Henrique_CORNEAU_pt.pdf',
      metaDesc: 'Henrique Corneau — Estudante de Engenharia Mecatrônica trabalhando com robótica, visão computacional e sistemas embarcados.'
    }
  };

  // ---------- Helpers ----------
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const t = (v, lang) => (v && typeof v === 'object' ? v[lang] : v);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ } }
  };

  let currentFilter = 'all';

  // ---------- Renderers ----------
  function renderProjects(lang) {
    $('#project-grid').innerHTML = projects.map((p) => `
      <article class="card reveal" data-cats="${p.cats.join(' ')}">
        <div class="card-media"><img src="${p.img}" alt="${esc(t(p.title, lang))}" loading="lazy" width="960" height="600"></div>
        <div class="card-body">
          <div class="card-meta"><span>${p.year}</span></div>
          <h3>${esc(t(p.title, lang))}</h3>
          <p>${esc(t(p.desc, lang))}</p>
          <ul class="tags">${p.stack.map((s) => `<li>${esc(s)}</li>`).join('')}</ul>
          <div class="card-links">${p.links.map((l) => `<a href="${l.href}" target="_blank" rel="noopener">${esc(t(l.label, lang))} ↗</a>`).join('')}</div>
        </div>
      </article>`).join('');
    applyFilter();
  }

  function renderPapers() {
    $('#paper-list').innerHTML = papers.map((p) => `
      <li class="reveal">
        <span class="paper-year">${p.year}</span>
        <div>
          <a href="${p.href}" target="_blank" rel="noopener">${esc(p.title)} ↗</a>
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

  // ---------- Language ----------
  function setLang(lang, persist) {
    if (!strings[lang]) lang = 'en';
    const S = strings[lang];
    document.documentElement.lang = lang === 'pt' ? 'pt-BR' : 'en';
    $$('[data-i18n]').forEach((el) => { const v = S[el.dataset.i18n]; if (v != null) el.textContent = v; });
    $$('[data-i18n-html]').forEach((el) => { const v = S[el.dataset.i18nHtml]; if (v != null) el.innerHTML = v; });
    $$('[data-cv]').forEach((el) => { el.href = S.cv; });
    $('meta[name="description"]').setAttribute('content', S.metaDesc);
    $$('.lang-switch button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.lang === lang)));

    renderProjects(lang);
    renderTimeline('#experience', experience, lang);
    renderTimeline('#education', education, lang);
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
  $$('.chip').forEach((c) => c.addEventListener('click', () => {
    currentFilter = c.dataset.filter;
    $$('.chip').forEach((x) => x.classList.toggle('active', x === c));
    applyFilter();
  }));
  $('#year').textContent = new Date().getFullYear();

  initTheme();
  renderPapers();
  const urlLang = new URLSearchParams(location.search).get('lang');
  setLang(urlLang || store.get('lang') || 'en', false);
})();
