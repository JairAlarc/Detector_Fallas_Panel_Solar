# DETECTOR DE FALLAS PARA SISTEMAS DE PANELES SOLAR
Ejecución de Proyecto de detector de fallas y anomalias en paneles solares
![Portada](https://github.com/JairAlarc/Detector_Fallas_Panel_Solar/blob/main/paneles%20solar%20foto.png)

### Proyecto IA Detector de Fallas en Paneles Solares

#### Introducción

**Contexto Global de la Energía Solar**
La energía solar ha emergido como una solución prometedora para la generación de electricidad limpia y sostenible. La capacidad solar instalada global alcanzó 800 GW en 2022 y se espera que alcance 1000 GW para 2025 (Global Solar Report 2023, Raptor Maps).

**Situación de la Energía Solar en Colombia**
Colombia ha visto un aumento significativo en la capacidad instalada de energía solar fotovoltaica, pasando de 13 MW en 2018 a 457 MW en 2022. Se espera que la capacidad instalada continúe creciendo, alcanzando 1500 MW para 2025.

![ColombiaCapacidadPOUT](https://github.com/JairAlarc/Detector_Fallas_Panel_Solar/blob/main/PVOUT.png)

**Problemática de las Anomalías en los Paneles Solares**
Diversas anomalías (puntos calientes, microgrietas, sombras) pueden reducir la eficiencia de los paneles y aumentar los costos de mantenimiento. Las pérdidas económicas asociadas a estas anomalías son significativas, con estimaciones de 67 millones de euros anuales en Colombia y 2500 millones de dólares a nivel global (Raptor Maps).

**Ventajas y Desventajas de la Tecnología Solar**
- **Ventajas:**
  - Sostenibilidad Ambiental: La energía solar no emite gases de efecto invernadero durante su operación, contribuyendo a la reducción de la huella de carbono.
  - Costos en Disminución: El costo de los paneles solares ha disminuido significativamente en la última década, haciendo que la energía solar sea más accesible.
  - Energía Inagotable: La energía solar es una fuente inagotable mientras el sol siga brillando, lo que la hace una solución a largo plazo.
  - Flexibilidad de Aplicación: Los sistemas solares pueden instalarse en una variedad de ubicaciones, desde techos residenciales hasta grandes plantas solares.
- **Desventajas:**
  - Intermitencia: La energía solar es intermitente y depende de la disponibilidad de luz solar, lo que puede ser un desafío en regiones con variabilidad climática.
  - Costo Inicial de Instalación: Aunque los costos están disminuyendo, la instalación inicial de sistemas solares puede ser costosa.
  - Eficiencia Afectada por Anomalías: Las anomalías en los paneles solares pueden reducir significativamente su eficiencia y aumentar los costos de mantenimiento.
  - Necesidad de Espacio: Los sistemas solares a gran escala requieren una cantidad considerable de espacio para su instalación.

**Enfoque en las Doce Clases de Anomalías**
El proyecto se centra en detectar y clasificar las siguientes anomalías:
1. No-Anomaly
2. Offline-Module
3. Cell
4. Vegetation
5. Diode-Multi
6. Diode
7. Cell-Multi
8. Shadowing
9. Cracking
10. Hot-Spot
11. Hot-Spot-Multi
12. Soiling
**Fuente Dataset** [GitHub](https://github.com/RaptorMaps/InfraredSolarModules/blob/master/README.md)

**Descripción del Dataset**
- El dataset consta de 20,000 imágenes etiquetadas manualmente por expertos, capturadas mediante inspecciones termográficas y visuales con drones equipados con cámaras infrarrojas y de alta resolución.

**Importancia del Proyecto**
Este proyecto no solo mejora la eficiencia y reduce los costos de mantenimiento de los sistemas solares en Colombia, sino que también puede servir como modelo para otros países que buscan optimizar sus operaciones solares.

#### Objetivos del Proyecto

**Objetivo General**
Desarrollar un modelo de clasificación de imágenes capaz de detectar y clasificar anomalías automáticamente, mejorando la eficiencia del mantenimiento y reduciendo las pérdidas económicas.

**Objetivos Específicos**
1. **Implementar y Evaluar Modelos de Transfer Learning:**
   - Utilizar arquitecturas preentrenadas como ResNet y EfficientNet.
   - **Meta:** Alcanzar una precisión inicial superior al 35%.

2. **Optimizar la Distribución de Datos y Técnicas de Preprocesamiento:**
   - Mitigar el impacto de clases desbalanceadas mediante técnicas como undersampling, oversampling y data augmentation.
   - **Meta:** Mejorar la precisión del modelo a al menos un 70%.

3. **Desplegar el Modelo y Desarrollar una API para Predicciones en Tiempo Real:**
   - Utilizar FastAPI y Render.
   - **Meta:** Lograr una precisión final superior al 90%.

#### Stack Tecnológico

**Lenguajes y Bibliotecas**
- **Python:** Scripts para preprocesamiento, entrenamiento de modelos y creación de APIs.
- **TensorFlow y Keras:** Construcción y entrenamiento de modelos de deep learning.
- **PyTorch:** Implementación de EfficientNet.
- **scikit-learn:** Preprocesamiento de datos, validación cruzada y ajuste de hiperparámetros.
- **OpenCV y PIL (Pillow):** Manipulación y procesamiento de imágenes.
- **Pandas y NumPy:** Manipulación y análisis de datos.
- **Matplotlib y Seaborn:** Visualización de datos.

**Entornos y Plataformas**
- **Google Colab:** Entrenamiento de modelos y desarrollo colaborativo.
- **Jupyter Notebooks:** Prototipado y análisis exploratorio de datos.
- **Visual Studio Code:** Desarrollo de scripts de Python y gestión de archivos.

**Herramientas de Despliegue**
- **FastAPI:** Creación de APIs.
- **Render:** Despliegue del modelo en la nube.

#### Metodología CRISP-DM

**Comprensión del Negocio**
- Identificar problemas comunes en los paneles solares y su impacto económico.

**Comprensión de los Datos**
- Análisis inicial del dataset y etiquetado manual de imágenes.

**Preparación de los Datos**
- Preprocesamiento, balanceo de clases y aumento de datos.

**Modelado**
- Desarrollo y evaluación de modelos de machine learning utilizando transfer learning y modelos preentrenados.

**Evaluación**
- Pruebas y ajustes para mejorar el rendimiento del modelo.

**Despliegue**
- Implementación del modelo en una API utilizando FastAPI y despliegue en Render.

#### Resultados y Discusión

**Evaluación del Modelo**
- El modelo alcanzó una precisión final del 93% mediante técnicas avanzadas de preprocesamiento y ajuste de hiperparámetros.

**Despliegue del Modelo**
- Implementación de una API eficiente para la evaluación en tiempo real de imágenes de paneles solares en la plataforma Render.

#### Conclusiones y Recomendaciones

**Conclusiones Principales**
1. **Eficiencia en la Clasificación de Anomalías:**
   - El modelo desarrollado puede identificar y clasificar doce tipos de anomalías en paneles solares, mejorando la eficiencia operativa de los sistemas solares.
   
2. **Importancia del Preprocesamiento de Datos:**
   - Técnicas como PCA y clustering mejoraron significativamente la calidad del dataset.

3. **Uso de Técnicas de Data Augmentation:**
   - Incremento de la precisión del modelo mediante data augmentation.

4. **Validación y Generalización del Modelo:**
   - Utilización de validación cruzada y generación de checkpoints para asegurar la reproducibilidad.

5. **Deployment eficiente:**
   - Despliegue en Render utilizando FastAPI.

**Recomendaciones para Futuras Mejoras**
1. **Optimización Continua del Modelo:**
   - Explorar nuevas arquitecturas y técnicas de machine learning y deep learning.

2. **Ampliación del Dataset:**
   - Incluir más imágenes de diversas fuentes y condiciones operativas.

3. **Integración de Datos de Sensores Adicionales:**
   - Incorporar datos adicionales de sensores térmicos y otros dispositivos de monitoreo.

4. **Mejora de la Infraestructura de Despliegue:**
   - Evaluar nuevas tecnologías de despliegue y automatización del CI/CD.

5. **Evaluación en Entornos Reales:**
   - Realizar pruebas de campo y evaluaciones en entornos operativos reales.

6. **Desarrollo de Capacidades de Interpretabilidad:**
   - Implementar técnicas de interpretabilidad y explicabilidad de modelos de machine learning.

#### Impacto del Proyecto
- **Eficiencia Operativa:** Mejora la eficiencia operativa al permitir un diagnóstico rápido y preciso de las anomalías.
- **Sostenibilidad:** Maximiza la eficiencia de los sistemas solares, apoyando la transición hacia fuentes de energía renovable.
- **Innovación Tecnológica:** Demuestra el potencial de las tecnologías de machine learning y deep learning para resolver problemas complejos en el sector energético.

#### Anexos
- **Enlace a GitHub y documentación técnica:** [GitHub](https://github.com/JairAlarc/Detector_Fallas_Panel_Solar)
- **Enlace a Render:** [Render](https://detector-fallas-panel-solar.onrender.com/docs)
- **Documentación** [Drive](https://drive.google.com/drive/folders/12XkYnMwrBBEFF5_GYHS1znmGmBsE144u?usp=drive_link)
