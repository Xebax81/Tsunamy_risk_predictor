# 🌊 Predictor de Riesgo de Tsunami

Aplicación web interactiva para el análisis de terremotos y predicción de riesgo de tsunami usando Machine Learning.

## 🚀 Demo en Vivo

👉 [Ver aplicación en Streamlit Cloud](https://tsunamy-risk-predictor.streamlit.app/) 

## 📊 Características

- **Análisis de Terremotos**: Dashboard interactivo con mapas de calor geoespaciales
- **Predicción de Tsunami**: Modelo Random Forest con balanceo SMOTEENN
- **Mapa Interactivo**: Selección de ubicación mediante clics en el mapa
- **Visualizaciones Avanzadas**: Gráficos de densidad, regresión y análisis geográfico
- **Análisis de Distancia al Océano**: Cálculo preciso usando GeoPandas

## 🛠️ Tecnologías

- **Streamlit**: Framework web
- **Scikit-learn**: Machine Learning (Random Forest)
- **Imbalanced-learn**: Balanceo de clases con SMOTEENN
- **GeoPandas**: Operaciones geoespaciales
- **Plotly**: Visualizaciones interactivas
- **Folium**: Mapas interactivos

## 📦 Instalación Local

```bash
# Clonar el repositorio
git clone https://github.com/Xebax81/Tsunamy_risk_predictor.git
cd Tsunamy_risk_predictor

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
streamlit run app.py
```

## 📁 Estructura del Proyecto

```
Tsunamy_risk_predictor/
├── app.py                      # Aplicación principal
├── requirements.txt            # Dependencias
├── Sismos_data.csv            # Dataset de terremotos
├── README.md                   # Documentación
└── .gitignore                 # Archivos ignorados por Git
```

## 🎯 Uso

### Pestaña de Análisis
1. Utiliza los filtros en la barra lateral para refinar los datos
2. Explora mapas de calor y zonas de alta sismicidad
3. Analiza distribuciones por país, continente y magnitud
4. Visualiza relaciones entre profundidad, magnitud y distancia al océano

### Pestaña de Predicción
1. **Selecciona la ubicación**: Haz clic en el mapa o ingresa coordenadas manualmente
2. **Completa los datos**: Magnitud, profundidad, tipo de magnitud, significancia
3. **Predice**: Obtén probabilidades de riesgo de tsunami

## 📈 Modelo de Machine Learning

- **Algoritmo**: Random Forest Classifier
- **Balanceo**: SMOTEENN (combinación de SMOTE + ENN)
- **Features**: 8 características (lat, lon, magnitud, profundidad, tipo, significancia, distancia al océano, coordenadas cartesianas)
- **Preprocesamiento**: Pipeline con transformadores personalizados

## 🌍 Datos

Dataset de terremotos con:
- 20,000+ registros históricos
- Información geoespacial (latitud, longitud)
- Características sísmicas (magnitud, profundidad)
- Riesgo de tsunami asociado

## 👨‍💻 Autor

**Xebax81**
- GitHub: [@Xebax81](https://github.com/Xebax81)

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

## 🙏 Agradecimientos

- Natural Earth por los datos geográficos
- USGS por los datos sísmicos
- Comunidad de Streamlit
