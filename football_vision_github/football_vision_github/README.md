# Football Vision — GitHub Pages / Browser-only

Aplicación estática que se publica con **GitHub Pages** y procesa el vídeo en el navegador. No necesita Python, Flask ni un servidor propio.

## Qué hace

- Subir un vídeo desde la web.
- Detectar jugadores y balón con YOLOv8n ONNX.
- Asignar IDs con un tracker IoU ligero.
- Intentar separar equipos por color dominante de camiseta.
- Inferir candidatos de posesión, pases, conducciones y tiros.
- Calcular coordenadas normalizadas del campo (105 × 68 m) y conservar también coordenadas de píxel.
- Crear mapas de calor, pases, tiros y Voronoi.
- Descargar `football_vision_actions.csv` con una fila por acción inferida y `football_vision_tracking.csv` con todas las muestras de tracking.
- Descargar un JSON completo del análisis.

## CSV de acciones

Cada fila incluye, como mínimo:

`action_id, timestamp, frame, action_type, team, player_id, player_track_id, receiver_player_id, receiver_track_id, start_x_m, start_y_m, end_x_m, end_y_m, ball_x_m, ball_y_m, distance_or_metric, outcome, xg, xa, confidence, coordinate_confidence, inference_source`

Esto permite relacionar cada acción con un jugador y sus coordenadas de inicio/fin y la posición del balón.

## Importante sobre la precisión

Esta versión está pensada como **prototipo browser-only**. El detector es un YOLOv8n generalista y el motor de eventos usa geometría. Para fútbol regional con cámara móvil, zoom, desenfoque, oclusiones y planos parciales, un sistema profesional necesita modelos entrenados específicamente para fútbol (jugador, balón, keypoints del campo, posesión y eventos) y una homografía dinámica/SLAM. Por tanto, `PASS_CANDIDATE`, `SHOT_CANDIDATE`, `CARRY_CANDIDATE` y `xG`/`xA` deben interpretarse como inferencias con una columna `confidence`, no como datos de proveedor profesional.

Además, el fallback de coordenadas usa un mapeo normalizado de la imagen al campo de 105 × 68 m. La columna `coordinate_confidence` marca la baja confianza de esta proyección cuando no se ha detectado una homografía real.

## Modelo

El navegador carga `yolov8n.onnx` desde Hugging Face. El README de Ultralytics para inferencia web recomienda alojar los modelos en el mismo origen o en un origen con CORS correcto; las URLs de assets de GitHub no son adecuadas para fetch directo desde el navegador por CORS. Por eso esta versión usa un recurso CORS habilitado durante el prototipo. Verifica las licencias del modelo antes de redistribuirlo públicamente.

## Publicar en GitHub Pages

1. Crea un repositorio público.
2. Sube `index.html`, `styles.css` y `app.js` a la raíz.
3. En **Settings → Pages**, selecciona `Deploy from a branch` y `main` / root.
4. Abre la URL que te dé GitHub Pages.

GitHub Pages solo sirve archivos estáticos; por eso toda la inferencia de esta versión ocurre dentro del navegador.

## Próxima versión recomendada

Para convertirlo en un analizador serio de fútbol regional, sustituye el detector generalista y el motor heurístico por:

1. detector de jugadores/árbitros/balón entrenado con frames de fútbol regional;
2. ReID para mantener IDs ante oclusiones;
3. keypoints del campo + homografía dinámica por frame;
4. modelo de posesión y receptor;
5. detector de eventos entrenado para pase, tiro, centro, recuperación, duelo, falta, despeje, intercepción, fuera de juego, etc.;
6. modelo xG calibrado con datos de partidos reales;
7. modelo xA que valore la probabilidad de asistencia a partir de la acción de pase.
