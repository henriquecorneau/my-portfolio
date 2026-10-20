/* Site content. Every user-facing text has an `en` and a `pt` version.
   To add a project: add an entry to `projects` and copy any file in /projects,
   changing only `data-project` and the <title>. */
(function () {
  'use strict';

  const ME = 'https://github.com/henriquecorneau/';
  const ZUNE = 'https://github.com/zunedrones/';

  const team = {
    zune: {
      en: 'Team project developed with the Zune Drones team (UFRN), which I founded.',
      pt: 'Projeto em equipe desenvolvido com a Zune Drones (UFRN), equipe que eu fundei.'
    },
    embarcatech: {
      en: 'Developed during the EMBARCATECH technology residency (IFRN), on the BitDogLab board.',
      pt: 'Desenvolvido durante a residência tecnológica EMBARCATECH (IFRN), na placa BitDogLab.'
    }
  };

  const projects = [
    {
      slug: 'drone-map-localization',
      year: 2026,
      cats: ['drones', 'ros', 'vision'],
      cover: 'assets/img/projects/drone-map-loc.jpg',
      title: { en: 'Drone Map Localization', pt: 'Localização de Drone em Mapa' },
      summary: {
        en: 'Builds an ORB-SLAM3 map from an RGB-D camera and localizes a monocular drone camera against the saved atlas — groundwork for autonomous navigation with Nav2.',
        pt: 'Constrói um mapa com ORB-SLAM3 a partir de uma câmera RGB-D e localiza a câmera monocular de um drone no atlas salvo — base para navegação autônoma com Nav2.'
      },
      context: {
        en: 'Current undergraduate research project (PIBIC, UFRN, 2026 – present).',
        pt: 'Projeto atual de iniciação científica (PIBIC, UFRN, 2026 – atual).'
      },
      stack: ['ROS 2 Humble', 'ORB-SLAM3', 'C++', 'Docker', 'Pangolin', 'Kinect v1', 'DJI Tello'],
      links: [
        { label: 'drone-map-loc', href: ME + 'drone-map-loc', primary: true },
        { label: 'orbslam3-ros2-docker', href: ME + 'orbslam3-ros2-docker' },
        { label: 'kinect-ros2', href: ME + 'kinect-ros2' },
        { label: 'tello_ros', href: ZUNE + 'tello_ros' }
      ],
      hero: { type: 'video', src: 'assets/video/drone-map-loc-localization.mp4', poster: 'assets/video/drone-map-loc-localization.jpg' },
      overview: {
        en: [
          'The goal of this project is to let a small drone fly autonomously through an environment that was mapped beforehand. Instead of asking the drone to build a map while it flies — hard with a single, low-quality camera — the map is built once with an RGB-D sensor, and the drone only has to find itself inside it.',
          'The map is created with ORB-SLAM3 using a Kinect v1 (color + depth). The resulting atlas is saved to disk and later loaded by a monocular ORB-SLAM3 node that receives the DJI Tello camera stream and runs in localization-only mode. The whole stack runs inside Docker, so ORB-SLAM3, Pangolin, OpenCV and ROS 2 Humble do not need to be installed on the host.'
        ],
        pt: [
          'O objetivo deste projeto é permitir que um drone pequeno voe de forma autônoma em um ambiente mapeado previamente. Em vez de pedir ao drone que construa o mapa enquanto voa — difícil com uma única câmera de baixa qualidade — o mapa é construído uma vez com um sensor RGB-D, e o drone só precisa se localizar dentro dele.',
          'O mapa é criado com o ORB-SLAM3 usando um Kinect v1 (cor + profundidade). O atlas resultante é salvo em disco e depois carregado por um nó monocular do ORB-SLAM3, que recebe o vídeo da câmera do DJI Tello e roda em modo somente localização. Toda a stack roda em Docker, então ORB-SLAM3, Pangolin, OpenCV e ROS 2 Humble não precisam ser instalados na máquina.'
        ]
      },
      steps: {
        en: [
          'Record the environment with a Kinect v1 through a Dockerized libfreenect ROS 2 driver, publishing synchronized /image_raw and /depth/image_raw topics.',
          'Run the RGB-D ORB-SLAM3 node over the recording; on shutdown it saves the atlas as an .osa file (--save-map).',
          'Record the drone flight: a ROS 2 node publishes the Tello camera as sensor_msgs/Image on /tello/camera, which can be stored in a bag.',
          'Start the monocular node with --load-map and --localize-only: it relocalizes the drone camera inside the RGB-D map and publishes pose, odometry, map point cloud, tracking image and keyframe markers.',
          'Next step: feed this pose to Nav2 to plan and follow paths through the mapped space.'
        ],
        pt: [
          'Gravar o ambiente com um Kinect v1 usando um driver ROS 2 com libfreenect em Docker, que publica os tópicos sincronizados /image_raw e /depth/image_raw.',
          'Rodar o nó RGB-D do ORB-SLAM3 sobre a gravação; ao encerrar, ele salva o atlas em um arquivo .osa (--save-map).',
          'Gravar o voo do drone: um nó ROS 2 publica a câmera do Tello como sensor_msgs/Image em /tello/camera, que pode ser salva em um bag.',
          'Iniciar o nó monocular com --load-map e --localize-only: ele relocaliza a câmera do drone dentro do mapa RGB-D e publica pose, odometria, nuvem de pontos do mapa, imagem de tracking e marcadores de keyframes.',
          'Próximo passo: enviar essa pose ao Nav2 para planejar e seguir trajetórias pelo espaço mapeado.'
        ]
      },
      highlights: {
        en: [
          'Cross-camera localization: the map is built with one sensor (Kinect RGB-D) and used by another (Tello monocular camera).',
          'Custom ROS 2 wrapper for ORB-SLAM3 with map save/load and a localization-only start mode.',
          'Reproducible Docker environment with ORB-SLAM3, Pangolin and ROS 2 Humble.',
          'Works with recorded ROS 2 bags, so mapping and localization can be tested without flying.'
        ],
        pt: [
          'Localização entre câmeras diferentes: o mapa é construído com um sensor (Kinect RGB-D) e usado por outro (câmera monocular do Tello).',
          'Wrapper ROS 2 próprio para o ORB-SLAM3, com salvar/carregar mapa e início em modo somente localização.',
          'Ambiente Docker reproduzível com ORB-SLAM3, Pangolin e ROS 2 Humble.',
          'Funciona com bags do ROS 2 gravados, então mapeamento e localização podem ser testados sem voar.'
        ]
      },
      gallery: [
        { type: 'video', src: 'assets/video/drone-map-loc-map.mp4', poster: 'assets/video/drone-map-loc-map.jpg',
          caption: { en: 'Building the map: point cloud and keyframes generated by ORB-SLAM3 (4× speed).', pt: 'Construção do mapa: nuvem de pontos e keyframes gerados pelo ORB-SLAM3 (4× mais rápido).' } },
        { type: 'video', src: 'assets/video/drone-map-loc-localization.mp4', poster: 'assets/video/drone-map-loc-localization.jpg',
          caption: { en: 'Localizing a monocular camera inside the previously built map (4× speed).', pt: 'Localizando uma câmera monocular dentro do mapa construído antes (4× mais rápido).' } }
      ]
    },

    {
      slug: 'tello-depth',
      year: 2026,
      cats: ['drones', 'vision'],
      cover: null,
      coverLabel: 'depth(frame) → meters',
      title: { en: 'Tello Depth', pt: 'Tello Depth' },
      summary: {
        en: 'Real-time monocular depth estimation on a DJI Tello with Depth Anything V2 and MiDaS, including metric scale calibration using the drone\'s own sensors.',
        pt: 'Estimativa de profundidade monocular em tempo real em um DJI Tello com Depth Anything V2 e MiDaS, incluindo calibração de escala métrica com os próprios sensores do drone.'
      },
      context: team.zune,
      stack: ['Python', 'PyTorch', 'LibreYOLO', 'Depth Anything V2', 'MiDaS', 'OpenCV', 'tello-zune'],
      links: [{ label: 'tello-depth', href: ZUNE + 'tello-depth', primary: true }],
      overview: {
        en: [
          'The Tello has a single forward camera and no depth sensor pointing forward. This project estimates depth from that single camera in real time, so the drone can tell how far it is from what it is looking at.',
          'The drone is flown from the keyboard while the camera feed is shown side by side with the depth map. A square at the center of the image measures the average depth of the region in front of the drone, and a calibration step converts the relative output of the network into meters.'
        ],
        pt: [
          'O Tello tem uma única câmera frontal e nenhum sensor de profundidade apontado para frente. Este projeto estima a profundidade a partir dessa única câmera em tempo real, para que o drone saiba a que distância está do que está vendo.',
          'O drone é pilotado pelo teclado enquanto o vídeo da câmera aparece lado a lado com o mapa de profundidade. Um quadrado no centro da imagem mede a profundidade média da região à frente do drone, e uma etapa de calibração converte a saída relativa da rede em metros.'
        ]
      },
      steps: {
        en: [
          'Load the camera intrinsics (calibrated for 960×720) and undistort every frame before inference.',
          'Run the depth model (Depth Anything V2 Small by default; MiDaS and ZipDepth are also supported) in a separate thread, so video and controls stay responsive.',
          'Average the relative inverse depth inside the central square.',
          'Convert to meters with a saved scale factor, calibrated either from one known distance (key C) or automatically (key V).',
          'Automatic calibration: the drone checks its height with the downward TOF sensor, flies about 50 cm forward while IMU/odometry measures the displacement, and computes the scale from the depth change.'
        ],
        pt: [
          'Carregar os parâmetros intrínsecos da câmera (calibrada para 960×720) e corrigir a distorção de cada frame antes da inferência.',
          'Rodar o modelo de profundidade (Depth Anything V2 Small por padrão; MiDaS e ZipDepth também são suportados) em uma thread separada, mantendo vídeo e controles responsivos.',
          'Calcular a média da profundidade inversa relativa dentro do quadrado central.',
          'Converter para metros com um fator de escala salvo, calibrado a partir de uma distância conhecida (tecla C) ou automaticamente (tecla V).',
          'Calibração automática: o drone confere a altura com o sensor TOF voltado para baixo, avança cerca de 50 cm enquanto a IMU/odometria mede o deslocamento, e calcula a escala a partir da variação de profundidade.'
        ]
      },
      highlights: {
        en: [
          'Solves the scale ambiguity of monocular depth using only sensors already on the drone.',
          'Interchangeable models: MiDaS Small/Large, Depth Anything V2 Small and ZipDepth.',
          'Runs on CPU; the calibration is saved and reused across runs.',
          'Emergency stop key that cuts the motors immediately.'
        ],
        pt: [
          'Resolve a ambiguidade de escala da profundidade monocular usando apenas sensores que já estão no drone.',
          'Modelos intercambiáveis: MiDaS Small/Large, Depth Anything V2 Small e ZipDepth.',
          'Roda em CPU; a calibração é salva e reutilizada entre execuções.',
          'Tecla de parada de emergência que corta os motores imediatamente.'
        ]
      }
    },

    {
      slug: 'boat-vision',
      year: 2025,
      cats: ['vision'],
      cover: 'assets/img/projects/boat-vision.jpg',
      title: { en: 'Boat Vision', pt: 'Boat Vision' },
      summary: {
        en: 'Perception for an autonomous robotic boat: YOLOv11 detection, stereo disparity maps and IMU + GPS fusion to estimate the global coordinates of detected objects.',
        pt: 'Percepção para um barco robótico autônomo: detecção com YOLOv11, mapas de disparidade estéreo e fusão de IMU + GPS para estimar as coordenadas globais dos objetos detectados.'
      },
      context: {
        en: 'Undergraduate research project (PIBIC, UFRN, 2024 – 2025) on visual navigation for an autonomous sailboat.',
        pt: 'Projeto de iniciação científica (PIBIC, UFRN, 2024 – 2025) sobre navegação visual para um veleiro autônomo.'
      },
      stack: ['Python', 'YOLOv11', 'OpenCV', 'Stereo Vision', 'pyproj', 'GPS', 'IMU'],
      links: [{ label: 'boat-vision', href: ME + 'boat-vision', primary: true }],
      overview: {
        en: [
          'An autonomous surface vessel needs to know not only that there is a boat or an obstacle ahead, but where it is in the world. This project turns a detection in a stereo image into a latitude/longitude coordinate.',
          'It combines three sources: a YOLO model trained to detect boats, a calibrated stereo camera that gives the distance to the object, and the vessel\'s own GPS and IMU, which give its position and heading.'
        ],
        pt: [
          'Uma embarcação autônoma precisa saber não só que existe um barco ou obstáculo à frente, mas onde ele está no mundo. Este projeto transforma uma detecção em uma imagem estéreo em uma coordenada de latitude/longitude.',
          'Ele combina três fontes: um modelo YOLO treinado para detectar barcos, uma câmera estéreo calibrada que fornece a distância até o objeto, e o GPS e a IMU da própria embarcação, que fornecem sua posição e orientação.'
        ]
      },
      steps: {
        en: [
          'Detect the object in the left image with a custom-trained YOLOv11 model.',
          'Compute the disparity map with OpenCV block matching (128 disparities) inside the bounding box.',
          'Convert disparity to distance using the calibrated focal length and baseline: Z = f · B / d.',
          'Compute the bearing of the object from its horizontal pixel offset and the focal length, and add the IMU heading corrected by the local magnetic declination.',
          'Project range and bearing from the vessel\'s UTM position and convert the result to latitude/longitude with pyproj.'
        ],
        pt: [
          'Detectar o objeto na imagem esquerda com um modelo YOLOv11 treinado para a tarefa.',
          'Calcular o mapa de disparidade com block matching do OpenCV (128 disparidades) dentro da bounding box.',
          'Converter disparidade em distância usando a distância focal e a baseline calibradas: Z = f · B / d.',
          'Calcular o ângulo do objeto a partir do deslocamento horizontal em pixels e da distância focal, somando a orientação da IMU corrigida pela declinação magnética local.',
          'Projetar distância e ângulo a partir da posição UTM da embarcação e converter o resultado para latitude/longitude com o pyproj.'
        ]
      },
      highlights: {
        en: [
          'End-to-end pipeline from pixels to geographic coordinates.',
          'Modular classes for distance measurement (DistanceMeter) and georeferencing (GPSMarker).',
          'Model training notebook included in the repository.'
        ],
        pt: [
          'Pipeline completo de pixels até coordenadas geográficas.',
          'Classes modulares para medição de distância (DistanceMeter) e georreferenciamento (GPSMarker).',
          'Notebook de treinamento do modelo incluído no repositório.'
        ]
      },
      gallery: [
        { src: 'assets/img/gallery/boat-vision-stereo.jpg', caption: { en: 'Left and right images of the stereo pair.', pt: 'Imagens esquerda e direita do par estéreo.' } },
        { src: 'assets/img/projects/boat-vision.jpg', caption: { en: 'YOLOv11 detection.', pt: 'Detecção com YOLOv11.' } },
        { src: 'assets/img/gallery/boat-vision-disparity.jpg', caption: { en: 'Disparity map used to measure distance.', pt: 'Mapa de disparidade usado para medir a distância.' } },
        { src: 'assets/img/gallery/boat-vision-gps.jpg', caption: { en: 'Estimated position of the detected boat on the map.', pt: 'Posição estimada do barco detectado no mapa.' } }
      ]
    },

    {
      slug: 'stereo-dynamic-programming',
      year: 2026,
      cats: ['vision'],
      cover: 'assets/img/projects/stereo-dp.jpg',
      title: { en: 'Stereo Matching with Dynamic Programming', pt: 'Visão Estéreo com Programação Dinâmica' },
      summary: {
        en: 'Dense stereo matching from scratch, inspired by Ohta & Kanade: each image row is solved as a shortest path on a graph, with an inter-scanline prior for consistency.',
        pt: 'Correspondência estéreo densa feita do zero, inspirada em Ohta & Kanade: cada linha da imagem é resolvida como um caminho mínimo em um grafo, com uma restrição entre linhas para dar consistência.'
      },
      context: {
        en: 'Developed during my exchange year at INSA Rouen Normandie (2025 – 2026).',
        pt: 'Desenvolvido durante o intercâmbio no INSA Rouen Normandie (2025 – 2026).'
      },
      stack: ['Python', 'NumPy', 'Dynamic Programming', 'OpenCV', 'Matplotlib', 'pytest'],
      links: [{ label: 'stereo-method-scratch', href: ME + 'stereo-method-scratch', primary: true }],
      hero: { type: 'image', src: 'assets/img/gallery/stereo-comparison.jpg' },
      overview: {
        en: [
          'Given two rectified images of the same scene, the goal is to find, for every pixel of the left image, how far its match is shifted in the right image. This shift — the disparity — is inversely proportional to depth, so the result is a dense depth map.',
          'The method follows the idea of Ohta & Kanade (1985): every image row is a graph whose nodes are (column, disparity) pairs, and the best disparity profile is the cheapest path through it, found with dynamic programming. The row above is then used as a prior, which removes the horizontal streaks of purely row-by-row methods. OpenCV\'s StereoSGBM is used as a baseline.'
        ],
        pt: [
          'Dadas duas imagens retificadas da mesma cena, o objetivo é descobrir, para cada pixel da imagem esquerda, o quanto o seu correspondente está deslocado na imagem direita. Esse deslocamento — a disparidade — é inversamente proporcional à profundidade, então o resultado é um mapa de profundidade denso.',
          'O método segue a ideia de Ohta & Kanade (1985): cada linha da imagem é um grafo cujos nós são pares (coluna, disparidade), e o melhor perfil de disparidades é o caminho mais barato por ele, encontrado com programação dinâmica. A linha de cima é usada como referência, o que elimina as listras horizontais dos métodos que tratam cada linha isoladamente. O StereoSGBM do OpenCV é usado como base de comparação.'
        ]
      },
      steps: {
        en: [
          'Matching cost: for each row, compare pixel x of the left image with pixel x − d of the right image for every disparity d; pixels without a valid match get an occlusion cost.',
          'Intra-scanline search: dynamic programming finds the cheapest path along the row, where changing disparity between neighbouring pixels costs a penalty P_h.',
          'Backtracking from the cheapest final node recovers the disparity of every pixel of the row.',
          'Inter-scanline consistency: before solving a row, every disparity different from the one chosen on the row above costs an extra penalty P_v.',
          'Repeat for all rows to build the dense disparity map, then compare with OpenCV StereoSGBM.'
        ],
        pt: [
          'Custo de correspondência: para cada linha, comparar o pixel x da imagem esquerda com o pixel x − d da direita para cada disparidade d; pixels sem correspondente válido recebem um custo de oclusão.',
          'Busca na linha (intra-scanline): a programação dinâmica encontra o caminho mais barato ao longo da linha, em que mudar de disparidade entre pixels vizinhos custa uma penalidade P_h.',
          'O backtracking a partir do nó final mais barato recupera a disparidade de cada pixel da linha.',
          'Consistência entre linhas (inter-scanline): antes de resolver uma linha, toda disparidade diferente da escolhida na linha de cima recebe uma penalidade extra P_v.',
          'Repetir para todas as linhas para formar o mapa de disparidade denso e comparar com o StereoSGBM do OpenCV.'
        ]
      },
      highlights: {
        en: [
          'Dense output with no invalid pixels, unlike SGBM, which leaves holes at occlusions.',
          'The inter-scanline prior removes most streaks at no extra run time.',
          'Vectorised DP: about 1.3 s for a 450×375 image with 64 disparity levels.',
          'Step-by-step notebook, command-line script and unit tests.'
        ],
        pt: [
          'Saída densa, sem pixels inválidos, ao contrário do SGBM, que deixa buracos nas oclusões.',
          'A restrição entre linhas elimina a maior parte das listras sem aumentar o tempo de execução.',
          'Programação dinâmica vetorizada: cerca de 1,3 s para uma imagem de 450×375 com 64 níveis de disparidade.',
          'Notebook passo a passo, script de linha de comando e testes unitários.'
        ]
      },
      gallery: [
        { src: 'assets/img/gallery/stereo-comparison.jpg', caption: { en: 'Row-by-row DP, DP with inter-scanline consistency, and OpenCV StereoSGBM (Middlebury Cones).', pt: 'DP linha a linha, DP com consistência entre linhas e StereoSGBM do OpenCV (Middlebury Cones).' } },
        { src: 'assets/img/gallery/stereo-surface_3d.jpg', caption: { en: '3D view of the disparity map.', pt: 'Visualização 3D do mapa de disparidade.' } },
        { src: 'assets/img/gallery/stereo-pentagon.jpg', caption: { en: 'Aerial stereo pair (Pentagon).', pt: 'Par estéreo aéreo (Pentágono).' } }
      ]
    },

    {
      slug: 'orb-from-scratch',
      year: 2026,
      cats: ['vision'],
      cover: 'assets/img/projects/orb.jpg',
      title: { en: 'ORB from Scratch', pt: 'ORB do Zero' },
      summary: {
        en: 'Oriented FAST — the keypoint detector of ORB — implemented from scratch and verified against OpenCV, plus a study of ORB feature matching under rotation, zoom, noise and blur.',
        pt: 'Oriented FAST — o detector de pontos-chave do ORB — implementado do zero e validado contra o OpenCV, além de um estudo do matching do ORB com rotação, zoom, ruído e desfoque.'
      },
      context: {
        en: 'Developed during my exchange year at INSA Rouen Normandie (2025 – 2026).',
        pt: 'Desenvolvido durante o intercâmbio no INSA Rouen Normandie (2025 – 2026).'
      },
      stack: ['Python', 'NumPy', 'OpenCV', 'FAST', 'ORB', 'RANSAC', 'pytest'],
      links: [{ label: 'orb-scratch', href: ME + 'orb-scratch', primary: true }],
      overview: {
        en: [
          'ORB (Oriented FAST and Rotated BRIEF) is one of the most used feature detectors in robotics — it is the front end of ORB-SLAM. This project rebuilds its detection stage from scratch to understand exactly what it does.',
          'The FAST segment test, the corner score, non-maximum suppression and the intensity-centroid orientation are implemented in NumPy. The detector gives exactly the same corners as OpenCV\'s FAST. The second part uses OpenCV\'s full ORB to study how matching behaves when the image is rotated, zoomed, blurred, noisy or low-contrast, and where it fails.'
        ],
        pt: [
          'O ORB (Oriented FAST and Rotated BRIEF) é um dos detectores de características mais usados em robótica — é a base do ORB-SLAM. Este projeto reconstrói a etapa de detecção do zero para entender exatamente o que ela faz.',
          'O teste de segmento do FAST, o score dos cantos, a supressão de não-máximos e a orientação por centroide de intensidade são implementados em NumPy. O detector encontra exatamente os mesmos cantos que o FAST do OpenCV. A segunda parte usa o ORB completo do OpenCV para estudar como o matching se comporta quando a imagem é girada, ampliada, desfocada, ruidosa ou com pouco contraste, e onde ele falha.'
        ]
      },
      steps: {
        en: [
          'FAST: a pixel is a corner if at least 9 contiguous pixels of the 16-pixel circle around it are all brighter or all darker than it by a threshold t. The test runs on the whole image at once.',
          'Each corner gets a score, and non-maximum suppression keeps only the strongest corner in each neighbourhood.',
          'Orientation: the angle from the corner to the intensity centroid of a circular 31-pixel patch, which rotates with the image.',
          'Matching with OpenCV ORB: binary rBRIEF descriptors, brute-force Hamming matching with cross-check.',
          'Geometric verification with a RANSAC homography to separate real matches from coincidences.'
        ],
        pt: [
          'FAST: um pixel é um canto se pelo menos 9 pixels contíguos do círculo de 16 pixels ao seu redor forem todos mais claros ou todos mais escuros que ele por um limiar t. O teste roda na imagem inteira de uma vez.',
          'Cada canto recebe um score, e a supressão de não-máximos mantém apenas o canto mais forte de cada vizinhança.',
          'Orientação: o ângulo do canto até o centroide de intensidade de uma região circular de 31 pixels, que gira junto com a imagem.',
          'Matching com o ORB do OpenCV: descritores binários rBRIEF e busca exaustiva por distância de Hamming com verificação cruzada.',
          'Verificação geométrica com uma homografia estimada por RANSAC para separar correspondências reais de coincidências.'
        ]
      },
      highlights: {
        en: [
          'Exactly the same corners as cv2.FastFeatureDetector, checked by a unit test.',
          'Orientation is repeatable: after a 30° rotation, 89% of the corners turn by 30° ± 10°.',
          'RANSAC keeps 71% of the matches for the rotated cathedral but only 3% against a different building.',
          'Vectorised NumPy detector, about 26× faster than the first pure-Python version.'
        ],
        pt: [
          'Exatamente os mesmos cantos que o cv2.FastFeatureDetector, verificado por um teste unitário.',
          'A orientação é repetível: após uma rotação de 30°, 89% dos cantos giram 30° ± 10°.',
          'O RANSAC mantém 71% dos matches com a catedral girada, mas só 3% com um prédio diferente.',
          'Detector vetorizado em NumPy, cerca de 26× mais rápido que a primeira versão em Python puro.'
        ]
      },
      gallery: [
        { src: 'assets/img/gallery/orb-ofast_rouen.jpg', caption: { en: 'From-scratch oFAST: 300 strongest corners with their orientation.', pt: 'oFAST feito do zero: os 300 cantos mais fortes com sua orientação.' } },
        { src: 'assets/img/gallery/orb-orientation.png', caption: { en: 'Angle change of each corner after rotating the image by 30°.', pt: 'Variação do ângulo de cada canto após girar a imagem em 30°.' } },
        { src: 'assets/img/gallery/orb-match_cathedrale_rotated_cropped.jpg', caption: { en: 'ORB matching between the cathedral and a rotated, cropped version.', pt: 'Matching do ORB entre a catedral e uma versão girada e recortada.' } },
        { src: 'assets/img/gallery/orb-false_matches_mecca.jpg', caption: { en: 'False matches between two different buildings, removed by RANSAC.', pt: 'Falsos matches entre dois prédios diferentes, eliminados pelo RANSAC.' } }
      ]
    },

    {
      slug: 'slam-kinect',
      year: 2026,
      cats: ['ros', 'vision'],
      cover: 'assets/img/projects/slam-kinect.jpg',
      title: { en: 'SLAM with Kinect', pt: 'SLAM com Kinect' },
      summary: {
        en: 'RGB-D SLAM using a Kinect v1 and ORB-SLAM3 inside ROS 2 nodes, with a Dockerized libfreenect driver that publishes image and depth topics.',
        pt: 'SLAM RGB-D usando um Kinect v1 e ORB-SLAM3 em nós ROS 2, com um driver libfreenect em Docker que publica os tópicos de imagem e profundidade.'
      },
      stack: ['ROS 2 Humble', 'ORB-SLAM3', 'C++', 'libfreenect', 'Docker', 'RViz'],
      links: [
        { label: 'kinect-orbslam', href: ME + 'kinect-orbslam', primary: true },
        { label: 'kinect-ros2', href: ME + 'kinect-ros2' }
      ],
      overview: {
        en: [
          'An old Xbox 360 Kinect is a cheap and surprisingly capable RGB-D sensor. This project brings it into ROS 2 and uses it to run ORB-SLAM3, producing a 3D map of the environment and the camera trajectory.',
          'It is split in two parts: a Docker image that builds libfreenect from source and runs a ROS 2 driver for the Kinect, and a ROS 2 package that wraps ORB-SLAM3 in monocular, stereo and RGB-D nodes.'
        ],
        pt: [
          'Um Kinect antigo do Xbox 360 é um sensor RGB-D barato e surpreendentemente capaz. Este projeto o integra ao ROS 2 e o usa para rodar o ORB-SLAM3, gerando um mapa 3D do ambiente e a trajetória da câmera.',
          'Ele é dividido em duas partes: uma imagem Docker que compila o libfreenect a partir do código-fonte e roda um driver ROS 2 para o Kinect, e um pacote ROS 2 que encapsula o ORB-SLAM3 em nós monocular, estéreo e RGB-D.'
        ]
      },
      steps: {
        en: [
          'The Kinect driver publishes /image_raw, /depth/image_raw and the camera_info topics; launch files add RViz visualization.',
          'The RGB-D SLAM node subscribes to the synchronized color and depth images and tracks the camera with ORB-SLAM3.',
          'Camera configuration files are included for the Kinect, the Tello, a stereo camera and the TUM dataset.',
          'The keyframe trajectory is saved at the end of each run.'
        ],
        pt: [
          'O driver do Kinect publica /image_raw, /depth/image_raw e os tópicos camera_info; arquivos de launch adicionam visualização no RViz.',
          'O nó de SLAM RGB-D assina as imagens de cor e profundidade sincronizadas e rastreia a câmera com o ORB-SLAM3.',
          'Há arquivos de configuração para o Kinect, o Tello, uma câmera estéreo e o dataset TUM.',
          'A trajetória dos keyframes é salva ao final de cada execução.'
        ]
      },
      highlights: {
        en: [
          'Turns low-cost hardware into a working RGB-D SLAM setup.',
          'Fully containerized driver — no libfreenect installation on the host.',
          'Became the mapping stage of the Drone Map Localization project.'
        ],
        pt: [
          'Transforma um hardware de baixo custo em um sistema de SLAM RGB-D funcional.',
          'Driver totalmente em contêiner — sem instalar o libfreenect na máquina.',
          'Virou a etapa de mapeamento do projeto Localização de Drone em Mapa.'
        ]
      },
      gallery: [
        { src: 'assets/img/projects/slam-kinect.jpg', caption: { en: 'ORB-SLAM3 map and trajectory built with the Kinect.', pt: 'Mapa e trajetória do ORB-SLAM3 construídos com o Kinect.' } }
      ]
    },

    {
      slug: 'tello-zune',
      year: 2024,
      cats: ['drones'],
      cover: null,
      coverLabel: 'pip install tello-zune',
      title: { en: 'tello-zune (Python library)', pt: 'tello-zune (biblioteca Python)' },
      summary: {
        en: 'Event-driven, thread-safe Python library published on PyPI to control the DJI Tello, read telemetry and stream video — the base of all our Tello projects.',
        pt: 'Biblioteca Python orientada a eventos e thread-safe, publicada no PyPI, para controlar o DJI Tello, ler telemetria e transmitir vídeo — a base de todos os nossos projetos com o Tello.'
      },
      context: team.zune,
      stack: ['Python', 'UDP', 'Threading', 'OpenCV', 'PyPI'],
      links: [
        { label: 'tello_zune', href: ZUNE + 'tello_zune', primary: true },
        { label: 'PyPI', href: 'https://pypi.org/project/tello-zune/' }
      ],
      overview: {
        en: [
          'The Tello is controlled through text commands sent over UDP. When video, telemetry and flight commands share the same program, it is easy to send overlapping packets or block the main loop waiting for a reply — and the drone behaves unpredictably.',
          'tello-zune solves this with command queues, background state monitoring and safety locks, so the drone responds predictably even while running long routes. It is published on PyPI and used by the team\'s tracking, gesture control, QR code, ROS 2 and depth estimation projects.'
        ],
        pt: [
          'O Tello é controlado por comandos de texto enviados via UDP. Quando vídeo, telemetria e comandos de voo dividem o mesmo programa, é fácil enviar pacotes sobrepostos ou travar o loop principal esperando uma resposta — e o drone se comporta de forma imprevisível.',
          'A tello-zune resolve isso com filas de comandos, monitoramento de estado em segundo plano e travas de segurança, para que o drone responda de forma previsível mesmo executando rotas longas. Ela está publicada no PyPI e é usada nos projetos de tracking, controle por gestos, QR code, ROS 2 e estimativa de profundidade da equipe.'
        ]
      },
      highlights: {
        en: [
          'Thread-safe communication using locks and queues to avoid UDP packet collisions.',
          'Passive telemetry on port 8890 (battery, height, speed, temperature) without blocking flight commands.',
          'Integrated video capture and resizing with OpenCV.',
          'Periodic routes and events with a custom delay command (e.g. patrols).',
          'Interactive terminal mode and an emergency_stop() that clears the queue and cuts the motors.'
        ],
        pt: [
          'Comunicação thread-safe usando locks e filas para evitar colisões de pacotes UDP.',
          'Telemetria passiva na porta 8890 (bateria, altura, velocidade, temperatura) sem bloquear os comandos de voo.',
          'Captura e redimensionamento de vídeo integrados com OpenCV.',
          'Rotas e eventos periódicos com um comando delay próprio (por exemplo, patrulhas).',
          'Modo interativo pelo terminal e um emergency_stop() que limpa a fila e corta os motores.'
        ]
      }
    },

    {
      slug: 'tello-tracking',
      year: 2024,
      cats: ['drones', 'vision'],
      cover: 'assets/img/projects/tello-tracking.jpg',
      title: { en: 'Tello Tracking', pt: 'Tello Tracking' },
      summary: {
        en: 'Real-time object tracking with a DJI Tello drone, combining a custom YOLOv8 model with PD control to keep the target centered and at a fixed distance.',
        pt: 'Rastreamento de objetos em tempo real com um drone DJI Tello, combinando um modelo YOLOv8 próprio com controle PD para manter o alvo centralizado e a uma distância fixa.'
      },
      context: team.zune,
      stack: ['Python', 'YOLOv8', 'Ultralytics', 'OpenCV', 'PD control', 'tello-zune'],
      links: [{ label: 'tello_tracking', href: ZUNE + 'tello_tracking', primary: true }],
      paper: 2,
      overview: {
        en: [
          'The drone looks for a specific object in its camera image and follows it: it turns, climbs or descends to keep the object in the center of the frame, and moves forward or backward to keep a constant distance.',
          'The detector is a custom-trained YOLOv8m model. The results were published at COBICET 2024.'
        ],
        pt: [
          'O drone procura um objeto específico na imagem da câmera e o segue: gira, sobe ou desce para manter o objeto no centro do quadro, e avança ou recua para manter uma distância constante.',
          'O detector é um modelo YOLOv8m treinado especificamente para a tarefa. Os resultados foram publicados no COBICET 2024.'
        ]
      },
      steps: {
        en: [
          'Capture the 960×720 video stream from the Tello.',
          'Detect the target with the custom YOLOv8 model and take the center of its bounding box.',
          'Compute the horizontal and vertical error between the box center and the image center.',
          'A PD controller turns these errors into yaw and up/down speeds, clipped to the drone\'s ±100 range.',
          'The box area controls distance: too small → move forward, too large → move back. Without a detection, the drone hovers.'
        ],
        pt: [
          'Capturar o vídeo de 960×720 do Tello.',
          'Detectar o alvo com o modelo YOLOv8 próprio e obter o centro da bounding box.',
          'Calcular o erro horizontal e vertical entre o centro da caixa e o centro da imagem.',
          'Um controlador PD transforma esses erros em velocidades de giro (yaw) e subida/descida, limitadas à faixa ±100 do drone.',
          'A área da caixa controla a distância: pequena demais → avança, grande demais → recua. Sem detecção, o drone fica parado no ar.'
        ]
      },
      gallery: [
        { src: 'assets/img/gallery/tello-tracking-yolo.jpg', caption: { en: 'Detection and tracking from the drone\'s point of view.', pt: 'Detecção e tracking do ponto de vista do drone.' } },
        { src: 'assets/img/projects/tello-tracking.jpg', caption: { en: 'Test flight following the target.', pt: 'Voo de teste seguindo o alvo.' } }
      ]
    },

    {
      slug: 'tello-hand-control',
      year: 2025,
      cats: ['drones', 'vision'],
      cover: 'assets/img/projects/tello-hand.jpg',
      title: { en: 'Tello Hand Control', pt: 'Controle do Tello por Gestos' },
      summary: {
        en: 'Fly a drone with hand gestures: MediaPipe hand landmarks and a neural network classifier turn gestures into takeoff, stop, follow and land commands.',
        pt: 'Pilote um drone com gestos: pontos da mão do MediaPipe e um classificador de rede neural transformam gestos em comandos de decolar, parar, seguir e pousar.'
      },
      context: team.zune,
      stack: ['Python', 'MediaPipe', 'TensorFlow Lite', 'OpenCV', 'tello-zune'],
      links: [{ label: 'tello_hand_control', href: ZUNE + 'tello_hand_control', primary: true }],
      paper: 0,
      overview: {
        en: [
          'The drone is controlled only by the pilot\'s hand, seen through its own camera. No remote control, phone or computer input is needed during the flight.',
          'Hand landmarks are extracted with MediaPipe and classified by a small neural network (TensorFlow Lite), building on an open-source keypoint classifier. The work was published at ECOP 2025.'
        ],
        pt: [
          'O drone é controlado apenas pela mão do piloto, vista pela própria câmera do drone. Não é preciso controle remoto, celular ou computador durante o voo.',
          'Os pontos da mão são extraídos com o MediaPipe e classificados por uma pequena rede neural (TensorFlow Lite), com base em um classificador de keypoints open source. O trabalho foi publicado no ECOP 2025.'
        ]
      },
      steps: {
        en: [
          'MediaPipe detects the hand and its 21 landmarks in each frame.',
          'Landmarks are normalized and classified into a gesture by the keypoint classifier.',
          'Gestures map to commands: ✌️ V sign → takeoff, ✊ fist → stop, ✋ open hand → tracking, 🤙 L sign → land.',
          'In tracking mode, a PD controller keeps the hand centered and at a constant distance, so the drone follows the pilot.'
        ],
        pt: [
          'O MediaPipe detecta a mão e seus 21 pontos em cada frame.',
          'Os pontos são normalizados e classificados em um gesto pelo classificador de keypoints.',
          'Os gestos viram comandos: ✌️ V → decolar, ✊ punho fechado → parar, ✋ mão aberta → seguir, 🤙 L → pousar.',
          'No modo de seguir, um controlador PD mantém a mão centralizada e a uma distância constante, e o drone acompanha o piloto.'
        ]
      },
      gallery: [
        { src: 'assets/img/gallery/tello-hand-gestures.jpg', caption: { en: 'Gesture recognized from the drone camera.', pt: 'Gesto reconhecido pela câmera do drone.' } },
        { src: 'assets/img/projects/tello-hand.jpg', caption: { en: 'Controlling the drone with an open hand.', pt: 'Controlando o drone com a mão aberta.' } }
      ]
    },

    {
      slug: 'logistics-robot',
      year: 2025,
      cats: ['ros'],
      cover: 'assets/img/projects/logistics-robot.jpg',
      title: { en: 'Logistics Robot', pt: 'Robô Logístico' },
      summary: {
        en: 'LiDAR mapping with Cartographer and autonomous navigation with Nav2 on a TurtleBot3, plus a custom C++ obstacle-avoidance node, simulated in Gazebo.',
        pt: 'Mapeamento com LiDAR via Cartographer e navegação autônoma com Nav2 em um TurtleBot3, além de um nó próprio em C++ de desvio de obstáculos, simulado no Gazebo.'
      },
      stack: ['ROS 2', 'C++', 'Nav2', 'Cartographer', 'LiDAR', 'Gazebo', 'TurtleBot3'],
      links: [{ label: 'logistics-robot', href: ME + 'logistics-robot', primary: true }],
      overview: {
        en: [
          'A mobile robot for a logistics scenario must explore an unknown space, build a map of it and then move autonomously between points without hitting anything.',
          'This project implements that full cycle with a TurtleBot3 in Gazebo: exploration with a reactive obstacle avoider, LiDAR SLAM with Cartographer, and path planning and following with Nav2 on the saved map.'
        ],
        pt: [
          'Um robô móvel em um cenário logístico precisa explorar um espaço desconhecido, construir um mapa dele e depois se mover de forma autônoma entre pontos sem colidir.',
          'Este projeto implementa esse ciclo completo com um TurtleBot3 no Gazebo: exploração com um desviador de obstáculos reativo, SLAM com LiDAR usando o Cartographer, e planejamento e execução de trajetórias com o Nav2 sobre o mapa salvo.'
        ]
      },
      steps: {
        en: [
          'Launch the TurtleBot3 world in Gazebo and start Cartographer.',
          'Explore with the custom obstacle_avoider_node: it reads the /scan ranges at 0° and ±30° and publishes /cmd_vel — straight ahead when the way is clear, turning away when something is closer than 1 m.',
          'Save the occupancy grid with nav2_map_server.',
          'Launch Nav2 with the saved map and send navigation goals.'
        ],
        pt: [
          'Abrir o mundo do TurtleBot3 no Gazebo e iniciar o Cartographer.',
          'Explorar com o obstacle_avoider_node próprio: ele lê as distâncias de /scan em 0° e ±30° e publica /cmd_vel — segue em frente quando o caminho está livre e desvia quando algo está a menos de 1 m.',
          'Salvar o mapa de ocupação com o nav2_map_server.',
          'Iniciar o Nav2 com o mapa salvo e enviar objetivos de navegação.'
        ]
      },
      gallery: [
        { src: 'assets/img/projects/logistics-robot.jpg', caption: { en: 'TurtleBot3 in Gazebo and the map built with the LiDAR.', pt: 'TurtleBot3 no Gazebo e o mapa construído com o LiDAR.' } }
      ]
    },

    {
      slug: 'rasp-tello-ros',
      year: 2025,
      cats: ['drones', 'ros', 'embedded'],
      cover: 'assets/img/projects/rasp-tello.jpg',
      title: { en: 'Rasp-Tello ROS', pt: 'Rasp-Tello ROS' },
      summary: {
        en: 'HTTP-to-ROS 2 bridge: a web page served by a Raspberry Pi Pico W sends commands through micro-ROS to control a DJI Tello drone.',
        pt: 'Ponte HTTP–ROS 2: uma página web servida por uma Raspberry Pi Pico W envia comandos via micro-ROS para controlar um drone DJI Tello.'
      },
      stack: ['C', 'Pico SDK', 'micro-ROS', 'lwIP', 'ROS 2', 'Python'],
      links: [{ label: 'rasp_ros_iot', href: ME + 'rasp_ros_iot', primary: true }],
      overview: {
        en: [
          'This project connects the IoT world to ROS 2. A microcontroller serves a simple web page; any phone on the network can open it and send flight commands, which arrive in ROS 2 as regular topic messages and are forwarded to the drone.',
          'It was a way to learn micro-ROS on a resource-constrained board while combining it with Wi-Fi networking on the same device.'
        ],
        pt: [
          'Este projeto conecta o mundo IoT ao ROS 2. Um microcontrolador serve uma página web simples; qualquer celular na rede pode abri-la e enviar comandos de voo, que chegam ao ROS 2 como mensagens comuns de tópico e são repassados ao drone.',
          'Foi uma forma de aprender micro-ROS em uma placa com recursos limitados, combinando-o com rede Wi-Fi no mesmo dispositivo.'
        ]
      },
      steps: {
        en: [
          'The Pico W joins the network, shows its IP address on the OLED display and serves an HTTP page with the commands (lwIP).',
          'When a button is pressed, the firmware publishes the command on the /pico_publisher topic through micro-ROS, using a serial (USB) transport.',
          'The micro-ROS agent on the computer bridges the messages into the ROS 2 graph.',
          'A Python ROS 2 node (tello_subscriber) receives the commands and sends them to the Tello over Wi-Fi.'
        ],
        pt: [
          'A Pico W entra na rede, mostra seu endereço IP no display OLED e serve uma página HTTP com os comandos (lwIP).',
          'Ao apertar um botão, o firmware publica o comando no tópico /pico_publisher via micro-ROS, usando transporte serial (USB).',
          'O micro-ROS agent no computador faz a ponte das mensagens para o grafo do ROS 2.',
          'Um nó ROS 2 em Python (tello_subscriber) recebe os comandos e os envia ao Tello via Wi-Fi.'
        ]
      },
      gallery: [
        { src: 'assets/img/projects/rasp-tello.jpg', caption: { en: 'micro-ROS agent, ROS 2 subscriber and the web page on a phone.', pt: 'micro-ROS agent, subscriber ROS 2 e a página web no celular.' } }
      ]
    },

    {
      slug: 'tank-secure',
      year: 2025,
      cats: ['embedded'],
      cover: 'assets/img/projects/tank-secure.jpg',
      title: { en: 'Tank Secure', pt: 'Tank Secure' },
      summary: {
        en: 'Water-tank monitoring on a Raspberry Pi Pico W with RFID-protected access to the OLED display, low-level alarms and telemetry sent to ThingSpeak.',
        pt: 'Monitoramento de reservatório com Raspberry Pi Pico W, acesso ao display OLED protegido por RFID, alarme de nível baixo e telemetria enviada ao ThingSpeak.'
      },
      context: team.embarcatech,
      stack: ['C', 'Pico SDK', 'RFID MFRC522', 'SSD1306 OLED', 'lwIP', 'ThingSpeak'],
      links: [
        { label: 'tank-secure', href: ME + 'tank-secure', primary: true },
        { label: 'thingspeak-pico-sdk', href: ME + 'thingspeak-pico-sdk' }
      ],
      overview: {
        en: [
          'A monitoring system for a water reservoir: it measures the water level and temperature, sends the data to the cloud and only shows the readings on the local display to authorized users.',
          'Built on the BitDogLab board (Raspberry Pi Pico W) with the official Pico SDK in C. To send data to ThingSpeak from the Pico SDK, I wrote a small reusable library, thingspeak-pico-sdk.'
        ],
        pt: [
          'Um sistema de monitoramento para um reservatório de água: mede o nível e a temperatura, envia os dados para a nuvem e só mostra as leituras no display local para usuários autorizados.',
          'Feito na placa BitDogLab (Raspberry Pi Pico W) com o Pico SDK oficial em C. Para enviar dados ao ThingSpeak a partir do Pico SDK, escrevi uma pequena biblioteca reutilizável, a thingspeak-pico-sdk.'
        ]
      },
      steps: {
        en: [
          'Read the tank level through the ADC (a joystick simulates the level sensor in the prototype) and the temperature from the RP2040 internal sensor.',
          'Wait for an RFID card (MFRC522 over SPI); authorized users unlock the readings on the SSD1306 OLED display.',
          'Every 25 seconds, an RTC alarm triggers the upload of level, temperature and user ID to a ThingSpeak channel over Wi-Fi.',
          'If the level drops below 15%, the system enters alert mode with the red LED and the buzzer.'
        ],
        pt: [
          'Ler o nível do reservatório pelo ADC (um joystick simula o sensor de nível no protótipo) e a temperatura pelo sensor interno do RP2040.',
          'Aguardar um cartão RFID (MFRC522 via SPI); usuários autorizados liberam as leituras no display OLED SSD1306.',
          'A cada 25 segundos, um alarme do RTC dispara o envio de nível, temperatura e ID do usuário para um canal do ThingSpeak via Wi-Fi.',
          'Se o nível cair abaixo de 15%, o sistema entra em modo de alerta com o LED vermelho e o buzzer.'
        ]
      },
      gallery: [
        { src: 'assets/img/projects/tank-secure.jpg', caption: { en: 'RFID authentication on the BitDogLab board.', pt: 'Autenticação RFID na placa BitDogLab.' } }
      ]
    },

    {
      slug: 'px4-keyboard-control',
      year: 2025,
      cats: ['drones', 'ros'],
      cover: 'assets/img/projects/px4-keyboard.jpg',
      title: { en: 'PX4 Keyboard Control', pt: 'Controle do PX4 pelo Teclado' },
      summary: {
        en: 'Offboard control of a simulated PX4 drone from the keyboard through ROS 2, with Gazebo, QGroundControl and the Micro XRCE-DDS bridge started by one launch file.',
        pt: 'Controle offboard de um drone PX4 simulado pelo teclado via ROS 2, com Gazebo, QGroundControl e a ponte Micro XRCE-DDS iniciados por um único launch.'
      },
      context: team.zune,
      stack: ['ROS 2', 'PX4', 'Gazebo', 'Micro XRCE-DDS', 'QGroundControl', 'Python'],
      links: [{ label: 'keyboard-control-ros-sim', href: ZUNE + 'keyboard-control-ros-sim', primary: true }],
      hero: { type: 'image', src: 'assets/img/gallery/px4-keyboard.jpg' },
      overview: {
        en: [
          'Before flying real autonomous drones, the team needed a safe place to test ROS 2 code against a real autopilot. This project sets up PX4 software-in-the-loop with Gazebo and controls the vehicle from ROS 2 in offboard mode.',
          'The keyboard node is the simplest possible "autonomy": it exercises the whole chain — arming, mode switching, takeoff, setpoints and landing — that later autonomous behaviors reuse.'
        ],
        pt: [
          'Antes de voar drones autônomos reais, a equipe precisava de um lugar seguro para testar código ROS 2 contra um piloto automático de verdade. Este projeto configura o PX4 em software-in-the-loop com o Gazebo e controla o veículo pelo ROS 2 em modo offboard.',
          'O nó de teclado é a "autonomia" mais simples possível: ele exercita toda a cadeia — armar, trocar de modo, decolar, enviar setpoints e pousar — que comportamentos autônomos depois reutilizam.'
        ]
      },
      steps: {
        en: [
          'One launch file starts the Micro XRCE-DDS agent, QGroundControl, the PX4 SITL simulation in Gazebo and the control node.',
          'The agent bridges PX4\'s internal uORB topics to ROS 2.',
          'The node arms the vehicle, switches to offboard mode and takes off to a set altitude.',
          'Keys send position/velocity setpoints: W/S/A/D to move, up/down, yaw left/right, return home and land.'
        ],
        pt: [
          'Um único arquivo de launch inicia o Micro XRCE-DDS agent, o QGroundControl, a simulação PX4 SITL no Gazebo e o nó de controle.',
          'O agent faz a ponte entre os tópicos uORB internos do PX4 e o ROS 2.',
          'O nó arma o veículo, troca para o modo offboard e decola até uma altitude definida.',
          'As teclas enviam setpoints de posição/velocidade: W/S/A/D para mover, subir/descer, girar, voltar para casa e pousar.'
        ]
      }
    }
  ];

  const papers = [
    {
      year: 2025,
      title: 'Drone Controlado por Gestos Manuais Classificados por Rede Neural',
      authors: 'Henrique Corneau, Ramon Souza, João Victor de Andrade, Bruno Silva',
      venue: 'ECOP 2025',
      href: 'https://periodicos.ufersa.edu.br/ecop/article/view/14093',
      project: 'tello-hand-control'
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
      href: 'https://www.even3.com.br/anais/cobicet2024/884110-drone-based-object-tracking-with-convolutional-neural-networks/',
      project: 'tello-tracking'
    }
  ];

  const experience = [
    {
      date: { en: '2026 – Present', pt: '2026 – Atual' },
      title: { en: 'Undergraduate Researcher (PIBIC)', pt: 'Iniciação Científica (PIBIC)' },
      org: 'UFRN',
      desc: {
        en: 'Development and implementation of autonomous navigation systems for drones, using SLAM for precise localization and path planning in pre-mapped environments.',
        pt: 'Desenvolvimento e implementação de sistemas de navegação autônoma para drones, utilizando técnicas de SLAM para localização precisa e planejamento de rotas em ambientes pré-mapeados.'
      }
    },
    {
      date: '2025',
      title: { en: 'Development Intern', pt: 'Estágio em Desenvolvimento' },
      org: 'EMBARCATECH · IFRN',
      desc: {
        en: 'Design and development of embedded systems and IoT products and applications for companies, using microcontrollers and a variety of sensors.',
        pt: 'Concepção e desenvolvimento de produtos e aplicações na área de sistemas embarcados e Internet das Coisas (IoT) para empresas, utilizando microcontroladores e diversos sensores.'
      }
    },
    {
      date: '2024 – 2025',
      title: { en: 'Undergraduate Researcher (PIBIC)', pt: 'Iniciação Científica (PIBIC)' },
      org: 'UFRN',
      desc: {
        en: 'Development, implementation and testing of robotic perception techniques for visual navigation on an autonomous sailboat.',
        pt: 'Desenvolvimento, implementação e testes de técnicas de percepção robótica para navegação visual aplicada a um barco a vela autônomo.'
      }
    },
    {
      date: '2023 – 2025',
      title: { en: 'Drone Design and Development · Founder', pt: 'Concepção e Desenvolvimento de Drones · Fundador' },
      org: 'Zune Drones · UFRN',
      desc: {
        en: 'Built and programmed drones designed to perform autonomous tasks, based on computer vision, artificial intelligence and ROS. Authored and co-authored papers at COBICET 2024, ECOP 2025 and COBICET 2025.',
        pt: 'Construção e programação de drones projetados para realizar tarefas autônomas, com base em visão computacional, inteligência artificial e ROS. Artigos publicados como autor e coautor no COBICET 2024, ECOP 2025 e COBICET 2025.'
      }
    },
    {
      date: '2022',
      title: { en: 'Junior Company Member', pt: 'Membro de Empresa Júnior' },
      org: 'Include Engenharia · UFRN',
      desc: {
        en: 'Development of web and mobile applications in React for companies.',
        pt: 'Desenvolvimento de aplicações web e mobile em React para empresas.'
      }
    }
  ];

  const education = [
    {
      date: { en: 'Sep 2025 – May 2026', pt: 'Set 2025 – Mai 2026' },
      title: { en: 'Computer Science & Information Technology — 4th year (exchange)', pt: 'Engenharia de Computação e Tecnologias da Informação — 4º ano (intercâmbio)' },
      org: 'INSA Rouen Normandie · France',
      desc: { en: "Génie Informatique et Technologies de l'Information.", pt: "Génie Informatique et Technologies de l'Information." }
    },
    {
      date: { en: 'Mar 2022 – Present', pt: 'Mar 2022 – Atual' },
      title: { en: 'Mechatronics Engineering', pt: 'Engenharia Mecatrônica' },
      org: 'Universidade Federal do Rio Grande do Norte · Brazil'
    },
    {
      date: '2018',
      title: { en: 'Technical High School in Building Construction', pt: 'Ensino Médio Técnico em Edificações' },
      org: 'Instituto Federal da Bahia · Barreiras'
    }
  ];

  const strings = {
    en: {
      'nav.projects': 'Projects', 'nav.papers': 'Papers', 'nav.about': 'About', 'nav.contact': 'Contact',
      'hero.eyebrow': 'Robotics · Computer Vision · Embedded Systems',
      'hero.title': "Hi, I'm Henrique!",
      'hero.lead': 'Mechatronics engineering student at <strong>UFRN</strong> (Brazil), with an exchange year at <strong>INSA Rouen Normandie</strong> (France). I build autonomous drones, robot perception pipelines and embedded systems — and I document my projects here.',
      'hero.status': 'Currently: undergraduate researcher at UFRN, working on autonomous drone navigation with SLAM.',
      'hero.cv': 'Resume',
      'projects.title': 'Featured Projects', 'projects.all': 'See all on GitHub →',
      'filters.all': 'All', 'filters.drones': 'Drones', 'filters.vision': 'Computer Vision', 'filters.ros': 'ROS 2 / SLAM', 'filters.embedded': 'Embedded / IoT',
      'papers.title': 'Publications',
      'about.title': 'About',
      'about.p1': "I'm a mechatronics engineering student at the Universidade Federal do Rio Grande do Norte (UFRN), passionate about robotics, computer vision, embedded systems and edge AI.",
      'about.p2': 'During my undergraduate studies I founded Zune Drones, a team focused on autonomous drones: we build and program drones to perform autonomous tasks using computer vision, AI and ROS, and published our work at COBICET and ECOP.',
      'about.p3': 'As an undergraduate researcher (PIBIC), I first worked on perception for an autonomous sailboat, fusing YOLO detections, stereo vision, IMU and GPS to georeference detected objects. Today my research is on autonomous drone navigation with SLAM in pre-mapped environments. I also took part in EMBARCATECH, a technology residency where I developed embedded and IoT solutions for companies.',
      'about.p4': 'Between 2025 and 2026 I spent an exchange year studying Computer Science and Information Technology at INSA Rouen Normandie, in France.',
      'about.skills': 'Skills', 'about.languages': 'Languages', 'about.experience': 'Experience', 'about.education': 'Education',
      'lang.pt': 'Portuguese', 'lang.en': 'English', 'lang.fr': 'French', 'lang.es': 'Spanish', 'lang.native': 'native',
      'contact.title': "Let's talk",
      'contact.text': 'Open to opportunities and collaborations in robotics, computer vision and embedded systems.',
      'card.details': 'Details',
      'detail.back': '← All projects',
      'detail.overview': 'Overview', 'detail.how': 'How it works', 'detail.highlights': 'Highlights',
      'detail.gallery': 'Gallery', 'detail.stack': 'Tech stack', 'detail.links': 'Links',
      'detail.paper': 'Related publication', 'detail.year': 'Year', 'detail.next': 'Next project',
      'detail.prev': 'Previous project', 'detail.notfound': 'Project not found.',
      cv: 'assets/cv_Henrique_CORNEAU_pt.pdf',
      metaDesc: 'Henrique Corneau — Mechatronics engineering student working on robotics, computer vision and embedded systems.'
    },
    pt: {
      'nav.projects': 'Projetos', 'nav.papers': 'Artigos', 'nav.about': 'Sobre', 'nav.contact': 'Contato',
      'hero.eyebrow': 'Robótica · Visão Computacional · Sistemas Embarcados',
      'hero.title': 'Olá, eu sou o Henrique!',
      'hero.lead': 'Estudante de Engenharia Mecatrônica na <strong>UFRN</strong> (Brasil), com um ano de intercâmbio no <strong>INSA Rouen Normandie</strong> (França). Desenvolvo drones autônomos, sistemas de percepção robótica e sistemas embarcados — e documento meus projetos aqui.',
      'hero.status': 'Atualmente: pesquisador de iniciação científica na UFRN, trabalhando com navegação autônoma de drones usando SLAM.',
      'hero.cv': 'Currículo',
      'projects.title': 'Projetos em Destaque', 'projects.all': 'Ver todos no GitHub →',
      'filters.all': 'Todos', 'filters.drones': 'Drones', 'filters.vision': 'Visão Computacional', 'filters.ros': 'ROS 2 / SLAM', 'filters.embedded': 'Embarcados / IoT',
      'papers.title': 'Publicações',
      'about.title': 'Sobre',
      'about.p1': 'Sou estudante de Engenharia Mecatrônica na Universidade Federal do Rio Grande do Norte (UFRN), apaixonado por robótica, visão computacional, sistemas embarcados e Edge AI.',
      'about.p2': 'Durante a graduação, fundei a Zune Drones, uma equipe focada em drones autônomos: construímos e programamos drones para realizar tarefas autônomas usando visão computacional, IA e ROS, e publicamos nossos trabalhos no COBICET e no ECOP.',
      'about.p3': 'Como pesquisador de iniciação científica (PIBIC), trabalhei primeiro com percepção para um veleiro autônomo, fundindo detecções do YOLO, visão estéreo, IMU e GPS para georreferenciar objetos detectados. Hoje minha pesquisa é sobre navegação autônoma de drones com SLAM em ambientes pré-mapeados. Também participei do EMBARCATECH, uma residência tecnológica em que desenvolvi soluções embarcadas e de IoT para empresas.',
      'about.p4': 'Entre 2025 e 2026 fiz um ano de intercâmbio em Engenharia de Computação e Tecnologias da Informação no INSA Rouen Normandie, na França.',
      'about.skills': 'Habilidades', 'about.languages': 'Idiomas', 'about.experience': 'Experiência', 'about.education': 'Formação',
      'lang.pt': 'Português', 'lang.en': 'Inglês', 'lang.fr': 'Francês', 'lang.es': 'Espanhol', 'lang.native': 'nativo',
      'contact.title': 'Vamos conversar',
      'contact.text': 'Aberto a oportunidades e colaborações em robótica, visão computacional e sistemas embarcados.',
      'card.details': 'Detalhes',
      'detail.back': '← Todos os projetos',
      'detail.overview': 'Visão geral', 'detail.how': 'Como funciona', 'detail.highlights': 'Destaques',
      'detail.gallery': 'Galeria', 'detail.stack': 'Tecnologias', 'detail.links': 'Links',
      'detail.paper': 'Publicação relacionada', 'detail.year': 'Ano', 'detail.next': 'Próximo projeto',
      'detail.prev': 'Projeto anterior', 'detail.notfound': 'Projeto não encontrado.',
      cv: 'assets/cv_Henrique_CORNEAU_pt.pdf',
      metaDesc: 'Henrique Corneau — Estudante de Engenharia Mecatrônica trabalhando com robótica, visão computacional e sistemas embarcados.'
    }
  };

  window.SITE = { projects, papers, experience, education, strings };
})();
