import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
from folium.plugins import HeatMap
from streamlit_folium import folium_static, st_folium
import folium
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier


# Tsunami Prediction Classes
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')


class AddDistanceToOceanTsunami(BaseEstimator, TransformerMixin):
    """Calcula distancia al océano usando Haversine"""
    def __init__(self):
        self.ocean_points = np.array([
            [0, -160], [20, -150], [-20, -140], [40, -170], [-40, -130],
            [10, 150], [-10, 160], [30, 140], [-30, 170],
            [0, -30], [30, -40], [-30, -20], [50, -20], [-50, -10],
            [10, -50], [-10, -40],
            [-10, 70], [-20, 80], [-30, 90], [0, 60], [-40, 100],
            [80, 0], [75, 90], [75, -90],
            [-70, 0], [-65, 90], [-65, -90], [-65, 180]
        ])

    def fit(self, X, y=None):
        return self

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371
        lat1_rad = np.deg2rad(lat1)
        lon1_rad = np.deg2rad(lon1)
        lat2_rad = np.deg2rad(lat2)
        lon2_rad = np.deg2rad(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def transform(self, X):
        X_transformed = X.copy()
        distances = []
        for idx in range(len(X_transformed)):
            lat = X_transformed.iloc[idx]['latitude']
            lon = X_transformed.iloc[idx]['longitude']
            min_distance = float('inf')
            for ocean_point in self.ocean_points:
                ocean_lat, ocean_lon = ocean_point
                dist = self.haversine_distance(lat, lon, ocean_lat, ocean_lon)
                min_distance = min(min_distance, dist)
            distances.append(min_distance)
        X_transformed['distance_to_ocean_tsunami'] = distances
        return X_transformed


class LatLonToCartesian(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        lat_rad = np.deg2rad(X_transformed['latitude'])
        lon_rad = np.deg2rad(X_transformed['longitude'])
        R = 6371
        X_transformed['x'] = R * np.cos(lat_rad) * np.cos(lon_rad)
        X_transformed['y'] = R * np.cos(lat_rad) * np.sin(lon_rad)
        X_transformed['z'] = R * np.sin(lat_rad)
        return X_transformed.drop(columns=['latitude', 'longitude'])



@st.cache_data
def load_coastlines():
    """Carga líneas de costa de Natural Earth (solo una vez)"""
    import zipfile
    import os
    import requests
    
    # Verificar si ya existe el shapefile
    if not os.path.exists('ne_10m_coastline.shp'):
        st.info("Descargando líneas de costa de Natural Earth...")
        url = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"
        zip_path = "ne_10m_coastline.zip"
        
        # Descargar con headers apropiados
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extraer
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        
        # Limpiar zip
        os.remove(zip_path)
        st.success("✅ Líneas de costa descargadas")
    
    coastlines = gpd.read_file('ne_10m_coastline.shp')
    return coastlines


def calculate_distance_to_ocean_vectorized(df):
    """
    Calcula distancia al océano usando GeoPandas (mucho más rápido y preciso).
    Usa líneas de costa reales de Natural Earth.
    """
    coastlines = load_coastlines()
    
    # Crear geometría
    geometry = gpd.points_from_xy(df['longitude'], df['latitude'])
    geo_df = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    # Proyectar a metros para cálculos precisos
    geo_df_proj = geo_df.to_crs('EPSG:3857')
    coast_proj = coastlines.to_crs('EPSG:3857')
    coast_union = coast_proj.unary_union
    
    # Calcular distancias (vectorizado - mucho más rápido)
    distances = geo_df_proj.geometry.apply(lambda p: p.distance(coast_union))
    
    return distances / 1000  # Convertir metros a kilómetros


@st.cache_data
def load_world_map():
    """Carga el mapa mundial desde Natural Earth (se cachea)"""
    url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)
    return world


def enrich_with_location(df):
    """
    Enriquece el dataframe con país y continente basado en lat/lon
    usando geopandas para mayor precisión y velocidad
    """
    # Cargar mapa mundial
    world = load_world_map()
    
    # Crear geometría de puntos
    geometry = gpd.points_from_xy(df['longitude'], df['latitude'])
    geo_df = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    # Preparar subset del mundo con las columnas necesarias
    world_subset = world[['NAME', 'CONTINENT', 'geometry']].copy()
    world_subset = world_subset.rename(columns={
        'NAME': 'name',
        'CONTINENT': 'continent'
    })
    
    # Spatial join para encontrar país y continente
    result = gpd.sjoin(
        geo_df,
        world_subset,
        how='left',
        predicate='within'
    )
    
    # Limpiar resultados
    result = result.rename(columns={'name': 'country'})
    result = result.drop(columns=['geometry', 'index_right'])
    
    # Manejar sismos en el océano (sin país)
    result['country'] = result['country'].fillna('Ocean')
    result['continent'] = result['continent'].fillna('Ocean')
    
    return result


@st.cache_data
def load_data():
    """Load and preprocess the earthquake dataset."""
    import os
    
    # Check if preprocessed file exists
    processed_file = 'Sismos_data_processed.csv'
    
    if os.path.exists(processed_file):
        # Load preprocessed data (much faster)
        df = pd.read_csv(processed_file)
        df['date_time'] = pd.to_datetime(df['date_time'])
        return df
    
    # If not, process the original file
    # Usar placeholders para los mensajes que se actualizarán
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    status_placeholder.info("Procesando datos por primera vez... Esto puede tomar 1-2 minutos.")
    df = pd.read_csv('Sismos_data.csv')
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'depth_km': 'depth',
        'magnitude_type': 'magType',
        'tsunami_risk': 'tsunami',
        'place': 'location',
        'timestamp': 'date_time'
    })
    
    # Convert date_time to datetime
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # Add country and continent using geopandas (more accurate and faster)
    if 'country' not in df.columns or 'continent' not in df.columns:
        progress_placeholder.info("Calculando país y continente con GeoPandas...")
        df = enrich_with_location(df)
    
    # Calculate distance to ocean if not already present
    if 'distance_to_ocean' not in df.columns:
        progress_placeholder.info("Calculando distancias al océano con GeoPandas...")
        df['distance_to_ocean'] = calculate_distance_to_ocean_vectorized(df)
    
    # Rename sig column if exists
    if 'significance' in df.columns:
        df['sig'] = df['significance']
    
    # Save preprocessed data for future use
    progress_placeholder.info("Guardando datos procesados...")
    df.to_csv(processed_file, index=False)
    
    # Limpiar mensajes intermedios y mostrar solo el éxito
    status_placeholder.empty()
    progress_placeholder.empty()
    st.success("¡Datos procesados y guardados! La próxima carga será instantánea.")
    
    return df


@st.cache_resource
def load_tsunami_data():
    """Load the tsunami risk dataset."""
    df = pd.read_csv('Sismos_data.csv')
    return df


@st.cache_resource
def train_tsunami_model():
    """Train Random Forest model with SMOTEENN for tsunami prediction."""
    # Load data
    df = load_tsunami_data()
    
    # Rename columns to match expected names
    df = df.rename(columns={
        'depth_km': 'depth',
        'magnitude_type': 'magType'
    })
    
    # Define columns to drop
    columns_to_drop = ['id', 'place', 'network', 'updated', 'timestamp', 'detail_url', 'alert_level', 'event_type']
    
    # Create preprocessing pipeline
    preprocessing_pipeline = Pipeline([
        ('drop_columns', DropColumns(columns_to_drop=columns_to_drop)),
        ('add_distance', AddDistanceToOceanTsunami()),
        ('lat_lon_to_cartesian', LatLonToCartesian())
    ])
    
    # Separate features and target
    X = df.drop('tsunami_risk', axis=1)
    y = df['tsunami_risk']
    
    # Apply preprocessing
    X_preprocessed = preprocessing_pipeline.fit_transform(X)
    
    # Define numeric and categorical columns after preprocessing
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    column_transformer = ColumnTransformer([
        ('num', numeric_pipeline, make_column_selector(dtype_include=np.number)),
        ('cat', categorical_pipeline, make_column_selector(dtype_include=object))
    ])
    
    # Create pipeline with SMOTEENN and Random Forest
    model_pipeline = ImbPipeline([
        ('preprocessor', column_transformer),
        ('smoteenn', SMOTEENN(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train model
    model_pipeline.fit(X_preprocessed, y)
    
    return model_pipeline, preprocessing_pipeline


# Main application
def main():
    st.title("Análisis y Predicción de Terremotos")

    # Create tabs
    tab1, tab2 = st.tabs(["Análisis", "Predicción Tsunami"])

    # Analysis Tab
    with tab1:
        st.header("Panel de Análisis de Terremotos")
        df = load_data()
        df['date_time'] = pd.to_datetime(df['date_time'])

        # Sidebar for filters
        st.sidebar.header("Filtros")
        date_range = st.sidebar.date_input("Seleccionar Rango de Fechas", [df['date_time'].min(), df['date_time'].max()])
        min_magnitude = st.sidebar.slider("Magnitud Mínima", float(df['magnitude'].min()), float(df['magnitude'].max()), 6.5)
        max_depth = st.sidebar.slider("Profundidad Máxima (km)", 0, int(df['depth'].max()), int(df['depth'].max()))
        tsunami_filter = st.sidebar.selectbox("Riesgo de Tsunami", options=["Todos", "Con Riesgo de Tsunami", "Sin Riesgo de Tsunami"], index=0)

        # Apply filters
        filtered_df = df[(df['date_time'] >= pd.Timestamp(date_range[0])) & 
                         (df['date_time'] <= pd.Timestamp(date_range[1])) & 
                         (df['magnitude'] >= min_magnitude) & 
                         (df['depth'] <= max_depth)]
        
        # Apply tsunami filter
        if tsunami_filter == "Con Riesgo de Tsunami":
            filtered_df = filtered_df[filtered_df['tsunami'] == 1]
        elif tsunami_filter == "Sin Riesgo de Tsunami":
            filtered_df = filtered_df[filtered_df['tsunami'] == 0]

        # Display key metrics
        st.header("Métricas Clave")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Terremotos", len(filtered_df))
        col2.metric("Magnitud Promedio", round(filtered_df['magnitude'].mean(), 2))
        col3.metric("Magnitud Máxima", filtered_df['magnitude'].max())
        col4.metric("Terremotos con Tsunami", filtered_df['tsunami'].sum())

        # Display heatmap
        st.header("Perspectivas Geoespaciales")
        m = folium.Map(location=[filtered_df['latitude'].mean(), filtered_df['longitude'].mean()], zoom_start=1)
        heat_data = filtered_df[['latitude', 'longitude']].values.tolist()
        HeatMap(data=heat_data, radius=10, blur=15, max_zoom=1).add_to(m)
        folium_static(m)

        st.subheader("Zonas de Alta Sismicidad")
        m2 = folium.Map(location=[filtered_df['latitude'].mean(), filtered_df['longitude'].mean()], zoom_start=1)
        for _, row in filtered_df.iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                icon=folium.Icon(color='red' if row['magnitude'] >= 7 else 'orange', icon='info-sign'),
                popup=f"Ubicación: {row['location']}<br>Magnitud: {row['magnitude']}<br>Profundidad: {row['depth']} km<br>Tsunami: {'Sí' if row['tsunami'] == 1 else 'No'}"
            ).add_to(m2)
        folium_static(m2)

        # Section 3: Earthquake Counts by Continent and Country
        st.header("Conteo de Terremotos por Continente y País")
        
        # Filtrar "Ocean" ya que no es ni país ni continente
        continent_country_df = filtered_df[
            (filtered_df['country'] != 'Ocean') & 
            (filtered_df['continent'] != 'Ocean')
        ].groupby(['continent', 'country']).size().reset_index(name='count')

        # Bar chart for countries
        st.subheader("Top 10 Países con Mayor Cantidad de Terremotos")
        top_countries = continent_country_df.sort_values(by='count', ascending=False).head(10)
        fig = px.bar(top_countries, x='country', y='count', color='continent', title="Países con Mayor Actividad Sísmica")
        st.plotly_chart(fig)

        # Pie chart for continents
        st.subheader("Distribución de Terremotos por Continente")
        continent_counts = continent_country_df.groupby('continent')['count'].sum().reset_index()
        fig = px.pie(continent_counts, values='count', names='continent', title="Terremotos por Continente")
        st.plotly_chart(fig)

        # Section 4: Magnitude Insights
        st.header("Análisis de Magnitud")

        # Histogram for magnitude frequencies
        st.subheader("Frecuencia de Diferentes Magnitudes")
        fig = px.histogram(filtered_df, x='magnitude', nbins=20, title="Distribución de Frecuencia de Magnitud")
        st.plotly_chart(fig)

        # Depth vs. Magnitude Analysis
        st.subheader("Profundidad vs. Magnitud")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Density Contour Plot
            st.markdown("**Gráfico de Contorno de Densidad**")
            fig_contour = px.density_contour(
                filtered_df, 
                x='magnitude', 
                y='depth',
                title="Profundidad vs Magnitud - Contornos de Densidad",
                labels={'magnitude': 'Magnitud', 'depth': 'Profundidad (km)'},
                marginal_x="histogram",
                marginal_y="histogram"
            )
            # No aplicar update_traces ya que causa conflictos con histogramas
            st.plotly_chart(fig_contour, use_container_width=True)
        
        with col2:
            # Scatter with Regression Line
            st.markdown("**Dispersión con Línea de Tendencia**")
            fig_scatter = px.scatter(
                filtered_df, 
                x='magnitude', 
                y='depth',
                color='tsunami',
                size='sig',
                trendline="ols",
                title="Profundidad vs Magnitud - Análisis de Regresión",
                labels={'magnitude': 'Magnitud', 'depth': 'Profundidad (km)'},
                opacity=0.6
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Distance to Ocean Analysis
        st.header("Análisis de Distancia al Océano")
        
        # Scatter plot: Distance to Ocean vs Tsunami Risk
        st.subheader("Distancia al Océano vs Riesgo de Tsunami")
        fig = px.scatter(filtered_df, x='distance_to_ocean', y='magnitude', 
                        color='tsunami', size='sig',
                        title="Distancia al Océano vs Magnitud (Coloreado por Riesgo de Tsunami)",
                        labels={'distance_to_ocean': 'Distancia al Océano (km)', 'magnitude': 'Magnitud'})
        st.plotly_chart(fig)
        
        # Box plot: Distance to Ocean by Tsunami Risk
        st.subheader("Distribución de Distancia al Océano por Riesgo de Tsunami")
        # Create distance ranges (bins of 50km)
        filtered_df['distance_range'] = (filtered_df['distance_to_ocean'] // 50) * 50
        distance_counts = filtered_df.groupby('distance_range').size().reset_index(name='count')
        distance_counts['distance_label'] = distance_counts['distance_range'].astype(str) + '-' + (distance_counts['distance_range'] + 50).astype(str) + ' km'
        fig = px.bar(distance_counts, x='count', y='distance_label', 
                    orientation='h',
                    title="Cantidad de Terremotos por Distancia al Océano (intervalos de 50km)",
                    labels={'count': 'Número de Terremotos', 'distance_label': 'Distancia al Océano'})
        st.plotly_chart(fig)

        # Section 5: Detailed Earthquake Table
        st.header("Listado de Terremotos Encontrados")
        st.dataframe(filtered_df[['magnitude', 'date_time', 'location', 'depth', 'tsunami', 'distance_to_ocean']].sort_values(by='date_time', ascending=False))

    # Tsunami Prediction Tab
    with tab2:
        st.header("Predicción de Riesgo de Tsunami")
        st.write("Modelo Random Forest con SMOTEENN")
        
        # Train model
        try:
            with st.spinner('Entrenando modelo...'):
                model_pipeline, preprocessing_pipeline = train_tsunami_model()
            st.success("✅ Modelo entrenado exitosamente")
        except Exception as e:
            st.error(f"Error al entrenar el modelo: {str(e)}")
            st.exception(e)
            st.stop()
        
        # Mapa interactivo para seleccionar ubicación
        st.subheader("📍 Seleccionar Ubicación del Sismo")
        st.info("👇 Haz clic en el mapa para seleccionar la ubicación del terremoto")
        
        # Inicializar coordenadas en session_state si no existen
        if 'selected_lat' not in st.session_state:
            st.session_state.selected_lat = 0.0
            st.session_state.selected_lon = 0.0
        
        # Crear mapa centrado en el mundo
        map_center = [st.session_state.selected_lat, st.session_state.selected_lon]
        m = folium.Map(location=map_center, zoom_start=2)
        
        # Añadir marcador si hay coordenadas seleccionadas
        if st.session_state.selected_lat != 0.0 or st.session_state.selected_lon != 0.0:
            folium.Marker(
                location=[st.session_state.selected_lat, st.session_state.selected_lon],
                popup=f"Lat: {st.session_state.selected_lat:.4f}<br>Lon: {st.session_state.selected_lon:.4f}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
        
        # Añadir funcionalidad de clic
        m.add_child(folium.LatLngPopup())
        
        # Mostrar mapa y capturar clics
        map_data = st_folium(m, width=700, height=400)
        
        # Actualizar coordenadas si se hizo clic en el mapa
        if map_data is not None and map_data.get('last_clicked') is not None:
            clicked_lat = map_data['last_clicked']['lat']
            clicked_lon = map_data['last_clicked']['lng']
            
            # Asegurarse de que los valores estén dentro del rango válido
            clicked_lat = max(-90.0, min(90.0, clicked_lat))
            clicked_lon = max(-180.0, min(180.0, clicked_lon))
            
            # Solo actualizar si los valores han cambiado significativamente
            if (abs(clicked_lat - st.session_state.selected_lat) > 0.001 or 
                abs(clicked_lon - st.session_state.selected_lon) > 0.001):
                st.session_state.selected_lat = clicked_lat
                st.session_state.selected_lon = clicked_lon
                st.rerun()
        
        st.subheader("Ingresar datos del sismo")
        
        col1, col2 = st.columns(2)
        
        # Asegurar que los valores estén siempre en rango válido
        safe_lat = max(-90.0, min(90.0, st.session_state.selected_lat))
        safe_lon = max(-180.0, min(180.0, st.session_state.selected_lon))
        
        with col1:
            latitude = st.number_input(
                "Latitud",
                min_value=-90.0,
                max_value=90.0,
                value=safe_lat,
                step=0.01,
                help="Selecciona en el mapa o ingresa manualmente"
            )
            longitude = st.number_input(
                "Longitud",
                min_value=-180.0,
                max_value=180.0,
                value=safe_lon,
                step=0.01,
                help="Selecciona en el mapa o ingresa manualmente"
            )
            magnitude = st.number_input(
                "Magnitud",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.1
            )
        
        with col2:
            depth = st.number_input(
                "Profundidad (km)",
                min_value=0.0,
                max_value=700.0,
                value=10.0,
                step=1.0
            )
            magType = st.selectbox(
                "Tipo de Magnitud",
                ["mb", "md", "ml", "ms", "mw", "mwb", "mwc", "mwr", "mww"],
                help="""**Guía de selección:**
                
• **Sismos grandes (>7.0)**: mw, mww, mwc (más precisas)
• **Sismos profundos**: mb (ondas internas)
• **Sismos locales**: ml (escala Richter)
• **Sismos superficiales**: ms (ondas superficiales)

**Tipos más comunes:**
- mww: Momento sísmico (W-phase) - Recomendado
- mwc: Momento sísmico (centroide)
- mw: Momento sísmico estándar
- mb: Magnitud de ondas internas
                """
            )
            significance = st.number_input(
                "Significancia",
                min_value=0,
                max_value=2000,
                value=500,
                step=10,
                help="""**Índice de importancia del sismo:**

• **0-200**: Sismos menores, poco significativos
• **200-500**: Significancia moderada
• **500-800**: Sismos importantes
• **800-1200**: Muy significativos, posible daño regional
• **>1200**: Extremadamente significativos, daño extenso

La significancia combina magnitud, población afectada y ubicación.
                """
            )
        
        if st.button("Predecir Riesgo de Tsunami", type="primary"):
            # Create input dataframe with column names matching the training data
            input_data = pd.DataFrame({
                'latitude': [latitude],
                'longitude': [longitude],
                'magnitude': [magnitude],
                'depth': [depth],
                'magType': [magType],
                'significance': [significance],
                'id': [''],
                'place': [''],
                'network': [''],
                'updated': [''],
                'timestamp': [''],
                'detail_url': [''],
                'alert_level': [''],
                'event_type': ['earthquake']
            })
            
            try:
                # Apply preprocessing
                input_preprocessed = preprocessing_pipeline.transform(input_data)
                
                # Make prediction
                prediction = model_pipeline.predict(input_preprocessed)[0]
                prediction_proba = model_pipeline.predict_proba(input_preprocessed)[0]
                
                st.subheader("Resultado de la Predicción")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error("⚠️ ALTO RIESGO DE TSUNAMI")
                        st.metric("Clasificación", "RIESGO ALTO")
                    else:
                        st.success("✅ BAJO RIESGO DE TSUNAMI")
                        st.metric("Clasificación", "RIESGO BAJO")
                
                with col2:
                    st.metric(
                        "Probabilidad Sin Tsunami", 
                        f"{prediction_proba[0]*100:.1f}%"
                    )
                    st.metric(
                        "Probabilidad Con Tsunami", 
                        f"{prediction_proba[1]*100:.1f}%"
                    )
                
                # Create probability chart
                fig_proba = go.Figure(data=[
                    go.Bar(
                        x=['Sin Tsunami', 'Con Tsunami'],
                        y=[prediction_proba[0]*100, prediction_proba[1]*100],
                        marker_color=['green', 'red'],
                        text=[
                            f"{prediction_proba[0]*100:.1f}%", 
                            f"{prediction_proba[1]*100:.1f}%"
                        ],
                        textposition='auto'
                    )
                ])
                fig_proba.update_layout(
                    title="Distribución de Probabilidades",
                    xaxis_title="Categoría",
                    yaxis_title="Probabilidad (%)",
                    height=400
                )
                st.plotly_chart(fig_proba, use_container_width=True)
                
                # Show input summary
                with st.expander("Ver detalles del sismo ingresado"):
                    st.write(f"**Ubicación:** Lat {latitude}°, Lon {longitude}°")
                    st.write(f"**Magnitud:** {magnitude} ({magType})")
                    st.write(f"**Profundidad:** {depth} km")
                    st.write(f"**Significancia:** {significance}")
                
            except Exception as e:
                st.error(f"Error al realizar la predicción: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()
