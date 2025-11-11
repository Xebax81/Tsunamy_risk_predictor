import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
from folium.plugins import HeatMap
from streamlit_folium import folium_static, st_folium
import folium
import numpy as np
import altair as alt
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
    # Configurar página para usar todo el ancho
    st.markdown("""
        <style>
        .main .block-container {
            max-width: 95%;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Banner con imagen de fondo y título
    st.markdown("""
        <style>
        .banner-container {
            position: relative;
            width: 100%;
            height: 250px;
            background-image: url('app/static/images/jcr_content.jpg');
            background-size: cover;
            background-position: center;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
        }
        .banner-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 10px;
        }
        .banner-title {
            position: relative;
            color: white;
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
            z-index: 1;
            padding: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Mostrar imagen de banner
    st.image("images/jcr_content.jpg", use_container_width=True)
    
    # Título superpuesto
    st.markdown("""
        <div style='margin-top: -180px; margin-bottom: 50px; text-align: center;'>
            <h1 style='color: white; font-size: 3em; text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.9); 
                       background: rgba(0, 0, 0, 0.4); padding: 20px; border-radius: 10px; 
                       display: inline-block;'>
                Análisis de Terremotos y Predicción de Tsunami
            </h1>
        </div>
    """, unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Análisis", "Análisis de Modelos de Predicción", "Predicción Tsunami"])

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
        
        # Logo y créditos en el sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Datos proporcionados por")
        st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/USGS_logo_green.svg/320px-USGS_logo_green.svg.png", width=200)
        st.sidebar.markdown("""
        <div style='text-align: center; font-size: 0.8em; color: #666;'>
        <a href='https://earthquake.usgs.gov/' target='_blank' style='text-decoration: none; color: #666;'>
        U.S. Geological Survey
        </a>
        </div>
        """, unsafe_allow_html=True)

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

        # Display interactive map with Altair
        st.header("Perspectivas Geoespaciales")
        
        # Configuración de Altair para gráficos más grandes
        alt.data_transformers.disable_max_rows()

        # Preparar etiquetas descriptivas
        filtered_df_map = filtered_df.copy()
        filtered_df_map['riesgo_tsunami'] = filtered_df_map['tsunami'].map({0: 'Sin Riesgo', 1: 'Con Riesgo de Tsunami'})

        # Mapa base del mundo
        countries = alt.topo_feature('https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json', 'countries')

        background = alt.Chart(countries).mark_geoshape(
            fill='lightgray',
            stroke='white'
        ).project('equalEarth').properties(
            width=900,
            height=500
        )

        # Capa de sismos (usando los datos ya filtrados)
        sismos = alt.Chart(filtered_df_map).mark_circle(
            opacity=0.7
        ).encode(
            longitude='longitude:Q',
            latitude='latitude:Q',
            color=alt.Color('riesgo_tsunami:N',
                            scale=alt.Scale(domain=['Sin Riesgo', 'Con Riesgo de Tsunami'],
                                           range=['#87CEEB', '#F44336']),
                            title='Riesgo de Tsunami'),
            size=alt.Size('magnitude:Q', scale=alt.Scale(range=[10, 200]), title='Magnitud'),
            tooltip=[
                alt.Tooltip('location:N', title='Ubicación'),
                alt.Tooltip('magnitude:Q', title='Magnitud', format='.2f'),
                alt.Tooltip('depth:Q', title='Profundidad (km)', format='.2f'),
                alt.Tooltip('riesgo_tsunami:N', title='Riesgo de Tsunami')
            ]
        ).project('equalEarth').properties(
            title={
                "text": "Distribución Geográfica de Sismos y Riesgo de Tsunami",
                "subtitle": "Color = Riesgo de Tsunami | Tamaño = Magnitud"
            }
        )

        mapa_interactivo = (background + sismos).configure_view(
            strokeWidth=0
        ).configure_legend(
            orient='bottom',
            titleFontSize=12,
            labelFontSize=11
        )
        
        st.altair_chart(mapa_interactivo, use_container_width=True)

        # Section 3: Earthquake Counts by Continent
        st.header("Distribución de Terremotos por Continente")
        
        # Filtrar "Ocean" ya que no es ni país ni continente
        continent_country_df = filtered_df[
            (filtered_df['country'] != 'Ocean') & 
            (filtered_df['continent'] != 'Ocean')
        ].groupby(['continent', 'country']).size().reset_index(name='count')

        # Pie chart for continents
        continent_counts = continent_country_df.groupby('continent')['count'].sum().reset_index()
        fig = px.pie(continent_counts, values='count', names='continent')
        st.plotly_chart(fig)

        # Section 4: Magnitude Insights
        st.header("Análisis de Magnitud")

        # Histogram for magnitude frequencies
        st.subheader("Frecuencia de Diferentes Magnitudes")
        fig = px.histogram(filtered_df, x='magnitude', nbins=20, title="Distribución de Frecuencia de Magnitud")
        st.plotly_chart(fig)

        # Depth vs. Magnitude Analysis
        st.subheader("Profundidad vs. Magnitud")
        
        # Preparar datos con tsunami como categórico
        filtered_df_plot = filtered_df.copy()
        filtered_df_plot['tsunami_label'] = filtered_df_plot['tsunami'].map({0: 'Sin Riesgo', 1: 'Con Riesgo'})
        
        # Scatter with Regression Line
        fig_scatter = px.scatter(
            filtered_df_plot, 
            x='magnitude', 
            y='depth',
            color='tsunami_label',
            size='sig',
            trendline="ols",
            title="Profundidad vs Magnitud - Análisis de Regresión",
            labels={'magnitude': 'Magnitud', 'depth': 'Profundidad (km)', 'tsunami_label': 'Riesgo de Tsunami'},
            color_discrete_map={'Sin Riesgo': '#87CEEB', 'Con Riesgo': '#F44336'},
            opacity=0.6
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Distance to Ocean Analysis
        st.header("Análisis de Distancia al Océano")
        
        # Scatter plot: Distance to Ocean vs Tsunami Risk
        st.subheader("Distancia al Océano vs Riesgo de Tsunami")
        
        # Preparar datos con tsunami como categórico
        filtered_df_ocean = filtered_df.copy()
        filtered_df_ocean['tsunami_label'] = filtered_df_ocean['tsunami'].map({0: 'Sin Riesgo', 1: 'Con Riesgo'})
        
        fig = px.scatter(filtered_df_ocean, x='distance_to_ocean', y='magnitude', 
                        color='tsunami_label', size='sig',
                        title="Distancia al Océano vs Magnitud (Coloreado por Riesgo de Tsunami)",
                        labels={'distance_to_ocean': 'Distancia al Océano (km)', 'magnitude': 'Magnitud', 'tsunami_label': 'Riesgo de Tsunami'},
                        color_discrete_map={'Sin Riesgo': '#87CEEB', 'Con Riesgo': '#F44336'})
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

        # Análisis de Localización Geográfica 3D
        st.header("Distribución Espacial 3D de Eventos Sísmicos")
        st.markdown("""
        Este análisis muestra la **distribución geográfica tridimensional** de los sismos usando coordenadas cartesianas (x, y, z).
        
        **¿Por qué es importante?** El modelo identificó que la **localización** es la variable más importante 
        para predecir tsunamis (38.6% de importancia). Este gráfico visualiza cómo los sismos con riesgo de tsunami 
        se concentran en zonas específicas del planeta (anillos de fuego, zonas de subducción).
        """)
        
        # Preparar datos con coordenadas cartesianas
        df_3d = filtered_df.copy()
        
        # Convertir coordenadas geográficas a cartesianas
        lat_rad = np.deg2rad(df_3d['latitude'])
        lon_rad = np.deg2rad(df_3d['longitude'])
        R = 6371  # Radio de la Tierra en km
        
        df_3d['x'] = R * np.cos(lat_rad) * np.cos(lon_rad)
        df_3d['y'] = R * np.cos(lat_rad) * np.sin(lon_rad)
        df_3d['z'] = R * np.sin(lat_rad)
        df_3d['tsunami_label'] = df_3d['tsunami'].map({0: 'Sin Riesgo', 1: 'Con Riesgo'})
        
        # Tomar una muestra para mejor visualización (todos los tsunamis + muestra de no-tsunamis)
        df_tsunamis = df_3d[df_3d['tsunami'] == 1]
        df_no_tsunamis = df_3d[df_3d['tsunami'] == 0].sample(n=min(2000, len(df_3d[df_3d['tsunami'] == 0])), random_state=42)
        df_3d_sample = pd.concat([df_tsunamis, df_no_tsunamis])
        
        # Crear gráfico 3D interactivo
        fig_3d = px.scatter_3d(
            df_3d_sample,
            x='x',
            y='y',
            z='z',
            color='tsunami_label',
            color_discrete_map={'Sin Riesgo': '#87CEEB', 'Con Riesgo': '#F44336'},
            hover_data={
                'latitude': ':.2f',
                'longitude': ':.2f',
                'magnitude': ':.1f',
                'depth': ':.1f',
                'x': ':.0f',
                'y': ':.0f',
                'z': ':.0f'
            },
            title='Distribución Espacial 3D de Eventos Sísmicos (Coordenadas Cartesianas)',
            labels={
                'x': 'Coordenada X (km)',
                'y': 'Coordenada Y (km)',
                'z': 'Coordenada Z (km)',
                'tsunami_label': 'Riesgo de Tsunami'
            }
        )
        
        # Agregar esfera representando la Tierra
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = R * np.outer(np.cos(u), np.sin(v))
        y_sphere = R * np.outer(np.sin(u), np.sin(v))
        z_sphere = R * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig_3d.add_trace(go.Surface(
            x=x_sphere,
            y=y_sphere,
            z=z_sphere,
            opacity=0.15,
            colorscale=[[0, 'lightgray'], [1, 'lightgray']],
            showscale=False,
            name='Superficie Tierra',
            hoverinfo='skip'
        ))
        
        # Configurar layout
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='X (km)',
                yaxis_title='Y (km)',
                zaxis_title='Z (km)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            height=700,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(l=0, r=0, t=60, b=0)
        )
        
        # Agregar borde al gráfico
        fig_3d.update_xaxes(showline=True, linewidth=1, linecolor='lightgray', mirror=True)
        fig_3d.update_yaxes(showline=True, linewidth=1, linecolor='lightgray', mirror=True)
        
        st.plotly_chart(fig_3d, use_container_width=True)
        
        st.caption(f"""
        Interacción: Rota arrastrando, zoom con rueda del mouse, hover para detalles | 
        Datos: {len(df_3d_sample):,} eventos ({len(df_tsunamis):,} con riesgo + {len(df_no_tsunamis):,} muestra sin riesgo)
        """)

        # Visualización complementaria: Globo Terráqueo
        st.markdown("---")
        st.subheader("🌍 Vista de Globo Terráqueo Interactivo")
        st.markdown("""
        Esta visualización complementaria presenta los mismos datos en un globo terráqueo interactivo, 
        ofreciendo una perspectiva más intuitiva de la distribución geográfica de los eventos sísmicos 
        y su relación con el riesgo de tsunami.
        """)
        
        # Preparar datos para el globo
        df_globe = filtered_df.copy()
        df_globe['tsunami_label'] = df_globe['tsunami'].map({0: 'Sin Riesgo', 1: 'Con Riesgo'})
        
        # Tomar muestra (todos los tsunamis + muestra de no-tsunamis)
        df_tsunamis_globe = df_globe[df_globe['tsunami'] == 1]
        df_no_tsunamis_globe = df_globe[df_globe['tsunami'] == 0].sample(
            n=min(3000, len(df_globe[df_globe['tsunami'] == 0])), 
            random_state=42
        )
        df_globe_sample = pd.concat([df_tsunamis_globe, df_no_tsunamis_globe])
        
        # Crear figura con globo terráqueo
        fig_globe = go.Figure()
        
        # Agregar eventos SIN riesgo de tsunami
        fig_globe.add_trace(go.Scattergeo(
            lon=df_no_tsunamis_globe['longitude'],
            lat=df_no_tsunamis_globe['latitude'],
            text=df_no_tsunamis_globe.apply(
                lambda row: f"<b>Sin Riesgo</b><br>" +
                            f"Magnitud: {row['magnitude']:.1f}<br>" +
                            f"Profundidad: {row['depth']:.1f} km<br>" +
                            f"Lat: {row['latitude']:.2f}°, Lon: {row['longitude']:.2f}°<br>" +
                            f"Ubicación: {row['location']}",
                axis=1
            ),
            mode='markers',
            marker=dict(
                size=4,
                color='#87CEEB',
                line=dict(width=0.5, color='white'),
                opacity=0.7
            ),
            name='Sin Riesgo',
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Agregar eventos CON riesgo de tsunami (más visibles)
        fig_globe.add_trace(go.Scattergeo(
            lon=df_tsunamis_globe['longitude'],
            lat=df_tsunamis_globe['latitude'],
            text=df_tsunamis_globe.apply(
                lambda row: f"<b>⚠️ Con Riesgo</b><br>" +
                            f"Magnitud: {row['magnitude']:.1f}<br>" +
                            f"Profundidad: {row['depth']:.1f} km<br>" +
                            f"Lat: {row['latitude']:.2f}°, Lon: {row['longitude']:.2f}°<br>" +
                            f"Ubicación: {row['location']}",
                axis=1
            ),
            mode='markers',
            marker=dict(
                size=8,
                color='#F44336',
                line=dict(width=1, color='white'),
                opacity=0.9
            ),
            name='Con Riesgo',
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Configurar el globo terráqueo
        fig_globe.update_geos(
            projection_type="orthographic",
            showcountries=True,
            countrycolor="lightgray",
            showcoastlines=True,
            coastlinecolor="gray",
            showland=True,
            landcolor="rgb(243, 243, 243)",
            showocean=True,
            oceancolor="rgb(204, 229, 255)",
            showlakes=True,
            lakecolor="rgb(204, 229, 255)",
            projection_rotation=dict(lon=0, lat=20, roll=0),
            center=dict(lon=0, lat=0)
        )
        
        # Configurar layout
        fig_globe.update_layout(
            title=dict(
                text='Distribución Global de Eventos Sísmicos',
                font=dict(size=16, color='#333333')
            ),
            width=1000,
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1
            ),
            paper_bgcolor='white',
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # Agregar borde al contenedor del gráfico
        fig_globe.update_geos(
            bgcolor='white',
            framecolor='lightgray',
            framewidth=1
        )
        
        st.plotly_chart(fig_globe, use_container_width=True)
        
        st.caption(f"""
        Datos: {len(df_globe_sample):,} eventos ({len(df_tsunamis_globe):,} con riesgo, {len(df_no_tsunamis_globe):,} sin riesgo) | 
        Interacción: Arrastra para rotar, zoom con rueda, hover para detalles, click en leyenda para filtrar
        """)

        # Análisis de Tipo de Magnitud (37.5% importancia)
        st.markdown("---")
        st.header("Relación entre Tipo de Magnitud y Riesgo de Tsunami")
        st.markdown("""
        El análisis de importancia de variables reveló que el **tipo de magnitud representa el 37.5% de la importancia** 
        en el modelo de predicción. Este gráfico muestra cómo diferentes escalas de medición de magnitud sísmica 
        (Richter, Momento, Ondas de Cuerpo, etc.) se correlacionan con la probabilidad de tsunami.
        
        El **Score Normalizado** combina el porcentaje de tsunamis con el volumen total de eventos, identificando 
        qué tipos de magnitud son más confiables como predictores.
        """)
        
        # Preparar datos para análisis de magType
        df_magnitude_analysis = filtered_df.copy()
        df_magnitude_analysis['tsunami_label'] = df_magnitude_analysis['tsunami'].map({0: 'Sin Tsunami', 1: 'Con Tsunami'})
        
        # Calcular estadísticas
        total_by_magnitude = df_magnitude_analysis.groupby('magType').size().reset_index(name='total')
        tsunami_by_magnitude = df_magnitude_analysis[df_magnitude_analysis['tsunami'] == 1].groupby('magType').size().reset_index(name='tsunami_count')
        
        magnitude_summary = total_by_magnitude.merge(tsunami_by_magnitude, on='magType', how='left')
        magnitude_summary['tsunami_count'] = magnitude_summary['tsunami_count'].fillna(0)
        magnitude_summary['tsunami_pct'] = (magnitude_summary['tsunami_count'] / magnitude_summary['total'] * 100).round(2)
        
        # Calcular scores normalizados para ambas categorías
        magnitude_summary['normalized_score'] = magnitude_summary['tsunami_pct'] * np.log10(magnitude_summary['total'] + 1)
        magnitude_summary['normalized_score'] = (magnitude_summary['normalized_score'] / magnitude_summary['normalized_score'].max() * 100).round(2)
        
        magnitude_summary['no_tsunami_count'] = magnitude_summary['total'] - magnitude_summary['tsunami_count']
        magnitude_summary['no_tsunami_pct'] = ((magnitude_summary['no_tsunami_count'] / magnitude_summary['total']) * 100).round(2)
        magnitude_summary['no_tsunami_normalized_score'] = magnitude_summary['no_tsunami_pct'] * np.log10(magnitude_summary['total'] + 1)
        magnitude_summary['no_tsunami_normalized_score'] = (magnitude_summary['no_tsunami_normalized_score'] / magnitude_summary['no_tsunami_normalized_score'].max() * 100).round(2)
        
        # Preparar datos en formato largo
        data_tsunami_bar = magnitude_summary[['magType', 'normalized_score', 'total', 'tsunami_count', 'tsunami_pct']].copy()
        data_tsunami_bar['category'] = 'Con Tsunami'
        data_tsunami_bar['score'] = data_tsunami_bar['normalized_score']
        data_tsunami_bar = data_tsunami_bar.rename(columns={'tsunami_count': 'count', 'tsunami_pct': 'pct'})
        
        data_no_tsunami_bar = magnitude_summary[['magType', 'no_tsunami_normalized_score', 'total', 'no_tsunami_count', 'no_tsunami_pct']].copy()
        data_no_tsunami_bar['category'] = 'Sin Tsunami'
        data_no_tsunami_bar['score'] = data_no_tsunami_bar['no_tsunami_normalized_score']
        data_no_tsunami_bar = data_no_tsunami_bar.rename(columns={'no_tsunami_count': 'count', 'no_tsunami_pct': 'pct'})
        
        data_combined = pd.concat([data_tsunami_bar, data_no_tsunami_bar], ignore_index=True)
        
        # Selector interactivo
        selection_bar = alt.selection_point(
            fields=['category'],
            bind='legend',
            value=[{'category': 'Con Tsunami'}, {'category': 'Sin Tsunami'}]
        )
        
        # Gráfico de barras agrupadas
        chart_combined = alt.Chart(data_combined).mark_bar().encode(
            x=alt.X('score:Q', 
                    title='Score Normalizado (% × log(Total Eventos))',
                    scale=alt.Scale(domain=[0, 105])),
            y=alt.Y('magType:N', 
                    sort=alt.EncodingSortField(field='score', op='max', order='descending'),
                    title='Tipo de Magnitud'),
            color=alt.Color('category:N',
                            title='Tipo de Evento',
                            scale=alt.Scale(
                                domain=['Con Tsunami', 'Sin Tsunami'],
                                range=['#F44336', '#87CEEB']
                            )),
            yOffset=alt.YOffset('category:N'),
            opacity=alt.condition(selection_bar, alt.value(0.9), alt.value(0.2)),
            tooltip=[
                alt.Tooltip('magType:N', title='Tipo de Magnitud'),
                alt.Tooltip('category:N', title='Categoría'),
                alt.Tooltip('score:Q', title='Score Normalizado', format='.2f'),
                alt.Tooltip('total:Q', title='Total Eventos', format=','),
                alt.Tooltip('count:Q', title='Cantidad', format=','),
                alt.Tooltip('pct:Q', title='Porcentaje', format='.2f')
            ]
        ).properties(
            width=800,
            height=500,
            title='Score Normalizado por Tipo de Magnitud: Tsunami vs No Tsunami'
        ).add_params(selection_bar)
        
        # Agregar etiquetas de score
        text_combined = chart_combined.mark_text(
            align='left',
            baseline='middle',
            dx=3,
            fontSize=9,
            fontWeight='bold'
        ).encode(
            text=alt.Text('score:Q', format='.1f'),
            color=alt.Color('category:N', 
                          scale=alt.Scale(
                              domain=['Con Tsunami', 'Sin Tsunami'],
                              range=['#F44336', '#87CEEB']
                          ), 
                          legend=None)
        )
        
        chart_final = (chart_combined + text_combined).configure_axis(
            labelFontSize=11,
            titleFontSize=13
        ).configure_title(
            fontSize=16,
            anchor='start'
        ).configure_legend(
            titleFontSize=12,
            labelFontSize=11
        )
        
        st.altair_chart(chart_final, use_container_width=True)
        
        st.caption("""
        Interacción: Click en la leyenda para mostrar/ocultar categorías, hover para ver detalles
        """)
        
        # Mostrar tabla resumen
        with st.expander("📊 Ver Tabla Resumen de Tipos de Magnitud"):
            summary_display = magnitude_summary[['magType', 'total', 'tsunami_count', 'tsunami_pct']].copy()
            summary_display.columns = ['Tipo de Magnitud', 'Total Eventos', 'Con Tsunami', '% Tsunami']
            summary_display = summary_display.sort_values('% Tsunami', ascending=False)
            st.dataframe(summary_display, use_container_width=True, hide_index=True)
            
            st.markdown("**🔍 Interpretación:**")
            top_3 = summary_display.head(3)
            for idx, row in top_3.iterrows():
                st.write(f"- **{row['Tipo de Magnitud']}**: {row['% Tsunami']:.2f}% de probabilidad de tsunami ({int(row['Con Tsunami'])} de {int(row['Total Eventos'])} eventos)")

        # Gráfico Polar complementario
        st.markdown("---")
        st.subheader("Vista Polar: Comparación por Tipo de Magnitud")
        st.markdown("""
        Esta visualización polar ofrece una perspectiva alternativa de los datos, 
        facilitando la comparación visual de scores entre diferentes tipos de magnitud 
        en un formato radial.
        """)
        
        # Preparar datos para gráfico polar
        n_types = len(magnitude_summary)
        magnitude_summary_polar = magnitude_summary.copy()
        magnitude_summary_polar['theta_index'] = range(n_types)
        
        # Crear datasets para barras polares
        data_tsunami_polar = magnitude_summary_polar[['magType', 'theta_index', 'normalized_score', 'total', 'tsunami_count', 'tsunami_pct']].copy()
        data_tsunami_polar['category'] = 'Con Tsunami'
        data_tsunami_polar['score'] = data_tsunami_polar['normalized_score']
        data_tsunami_polar = data_tsunami_polar.rename(columns={'tsunami_count': 'count', 'tsunami_pct': 'pct'})
        
        data_no_tsunami_polar = magnitude_summary_polar[['magType', 'theta_index', 'no_tsunami_normalized_score', 'total', 'no_tsunami_count', 'no_tsunami_pct']].copy()
        data_no_tsunami_polar['category'] = 'Sin Tsunami'
        data_no_tsunami_polar['score'] = data_no_tsunami_polar['no_tsunami_normalized_score']
        data_no_tsunami_polar = data_no_tsunami_polar.rename(columns={'no_tsunami_count': 'count', 'no_tsunami_pct': 'pct'})
        
        # Combinar datasets
        data_polar = pd.concat([data_tsunami_polar, data_no_tsunami_polar], ignore_index=True)
        
        # Selector interactivo
        selection_polar = alt.selection_point(
            fields=['category'],
            bind='legend',
            value=[{'category': 'Con Tsunami'}, {'category': 'Sin Tsunami'}]
        )
        
        # Crear barras polares con mark_arc
        polar_bars = alt.Chart(data_polar).mark_arc(stroke='white', tooltip=True).encode(
            theta=alt.Theta('theta_index:O', title=None),
            radius=alt.Radius('score:Q', title=None).scale(type='linear', domain=[0, 100]),
            radius2=alt.datum(5),
            color=alt.Color('category:N',
                            title='Tipo de Evento',
                            scale=alt.Scale(
                                domain=['Con Tsunami', 'Sin Tsunami'],
                                range=['#F44336', '#87CEEB']
                            )),
            opacity=alt.condition(selection_polar, alt.value(0.9), alt.value(0.2)),
            tooltip=[
                alt.Tooltip('magType:N', title='Tipo de Magnitud'),
                alt.Tooltip('category:N', title='Categoría'),
                alt.Tooltip('score:Q', title='Score Normalizado', format='.2f'),
                alt.Tooltip('total:Q', title='Total Eventos', format=','),
                alt.Tooltip('count:Q', title='Cantidad', format=','),
                alt.Tooltip('pct:Q', title='Porcentaje', format='.2f')
            ],
            order=alt.Order('category:N')
        ).add_params(selection_polar)
        
        # Anillos de referencia
        axis_rings = alt.Chart(pd.DataFrame({'ring': [25, 50, 75, 100]})).mark_arc(
            stroke='lightgrey', 
            fill=None,
            strokeWidth=1
        ).encode(
            theta=alt.value(2 * np.pi),
            radius=alt.Radius('ring:Q').stack(False)
        )
        
        # Etiquetas de los anillos
        axis_rings_labels = axis_rings.mark_text(
            color='grey',
            radiusOffset=5,
            align='left',
            fontSize=9
        ).encode(
            text=alt.Text('ring:Q', format='d'),
            theta=alt.value(np.pi / 4)
        )
        
        # Líneas radiales
        axis_lines = alt.Chart(magnitude_summary_polar).mark_arc(
            stroke='lightgrey',
            fill=None,
            strokeWidth=0.5
        ).encode(
            theta=alt.Theta('theta_index:O'),
            radius=alt.value(100),
            radius2=alt.datum(5)
        )
        
        # Etiquetas de tipos de magnitud
        axis_labels = alt.Chart(magnitude_summary_polar).mark_text(
            color='#333333',
            radiusOffset=10,
            fontSize=10,
            fontWeight='bold',
            align=alt.expr(
                'datum.theta_index < ' + str(n_types//4) + ' ? "left" : ' +
                'datum.theta_index > ' + str(3*n_types//4) + ' ? "right" : "center"'
            ),
            baseline=alt.expr(
                'datum.theta_index == 0 ? "bottom" : ' +
                'datum.theta_index == ' + str(n_types//2) + ' ? "top" : "middle"'
            )
        ).encode(
            theta=alt.Theta('theta_index:O'),
            radius=alt.value(100),
            text='magType:N'
        )
        
        # Combinar todos los elementos
        polar_chart = alt.layer(
            axis_rings,
            axis_lines,
            polar_bars,
            axis_rings_labels,
            axis_labels
        ).properties(
            width=700,
            height=700,
            title='Gráfico Polar: Score Normalizado por Tipo de Magnitud'
        ).configure_view(
            strokeWidth=0
        ).configure_title(
            fontSize=16,
            anchor='start'
        )
        
        st.altair_chart(polar_chart, use_container_width=True)
        
        st.caption("""
        Interacción: Click en leyenda para filtrar, hover para detalles | Valores normalizados 0-100 | Cada segmento = tipo de magnitud
        """)

    # Model Analysis Tab
    with tab2:
        st.header("Análisis de Modelos de Predicción")
        st.write("Esta sección muestra el análisis y comparación de diferentes modelos de predicción de tsunamis.")
        
        # Introducción
        st.markdown("""
        Se evaluaron **9 modelos de Machine Learning** diferentes para predecir el riesgo de tsunami:
        - **3 modelos base:** Logistic Regression, Random Forest y Support Vector Classifier (SVC)
        - **3 modelos optimizados con GridSearch:** Con ajuste de hiperparámetros
        - **3 modelos con SMOTEENN:** Con balanceo de clases para manejar el desbalance de datos
        """)
        
        # Sección 1: Modelos Básicos
        st.markdown("---")
        st.header("Modelos Básicos")
        st.markdown("""
        Estos son los modelos entrenados con configuraciones predeterminadas, sin optimización de hiperparámetros.
        """)
        
        # Mostrar imagen de matrices de modelos básicos
        try:
            st.image("images/matrices_modelos_basicos.png", 
                     caption="Matrices de Confusión - Modelos Básicos",
                     use_container_width=True)
        except:
            st.warning("Imagen no disponible: matrices_modelos_basicos.png")
        
        # Análisis de modelos básicos
        st.subheader("Análisis de Modelos Básicos")
        st.markdown("""
             
        **Conclusión:** Random Forest muestra el mejor desempeño inicial, sugiriendo que las relaciones entre 
        variables sísmicas y riesgo de tsunami son complejas y no lineales.
        """)
        
        # Sección 2: Modelos con GridSearch
        st.markdown("---")
        st.header("Modelos Optimizados con GridSearch")
        st.markdown("""
        Estos modelos fueron optimizados mediante búsqueda exhaustiva de hiperparámetros usando **GridSearchCV**.
        El objetivo es encontrar la mejor combinación de parámetros para maximizar el F1-Score.
        """)
        
        # Mostrar imagen de matrices con GridSearch
        try:
            st.image("images/matrices_gridsearch.png", 
                     caption="Matrices de Confusión - Modelos Optimizados (GridSearch)",
                     use_container_width=True)
        except:
            st.warning("Imagen no disponible: matrices_gridsearch.png")
        
        # Análisis de modelos con GridSearch
        st.subheader("Análisis de Modelos con GridSearch")
        st.markdown("""
        **Hiperparámetros optimizados:**
        
        - **Logistic Regression:** `C=10`, `solver='lbfgs'`
        - **Random Forest:** `n_estimators=200`, `max_depth=None`, `min_samples_split=2`
        - **SVC:** `C=10`, `kernel='rbf'`, `gamma='scale'`
                       
        ** Problema crítico con GridSearch:** Aunque optimiza el F1-Score, aún deja 
        demasiados **falsos negativos** (tsunamis no detectados). En un sistema de alerta 
        de tsunamis, NO DETECTAR un tsunami es catastrófico.
        """)
        
        # Sección 3: Modelos con SMOTEENN
        st.markdown("---")
        st.header("Modelos con SMOTEENN")
        st.markdown("""
        **SMOTEENN (SMOTE + Edited Nearest Neighbors)** es una técnica de balanceo que combina:
        - **SMOTE:** Genera muestras sintéticas de la clase minoritaria (tsunamis)
        - **ENN:** Limpia ejemplos ruidosos de ambas clases
        
        Esto aborda el problema de **desbalance de clases** (solo 3.6% de eventos tienen tsunami).
        """)
        
        # Mostrar imagen de matrices con SMOTEENN
        try:
            st.image("images/matrices_smoteenn.png", 
                     caption="Matrices de Confusión - Modelos con SMOTEENN",
                     use_container_width=True)
        except:
            st.warning("Imagen no disponible: matrices_smoteenn.png")
        
        # Análisis de modelos con SMOTEENN
        st.subheader("Análisis de Modelos con SMOTEENN")
        st.markdown("""
                
        **Conclusión CRÍTICA:** En sistemas de alerta de tsunamis, **Random Forest + SMOTEENN** 
        es SUPERIOR porque minimiza falsos negativos (tsunamis no detectados) a costa de 
        algunos falsos positivos más. Este es el trade-off correcto para salvar vidas.
        """)
        
        # Sección 4: Importancia de Variables
        st.markdown("---")
        st.header("Importancia de Variables en el Modelo Elegido")
        st.markdown("""
        El modelo **Random Forest + SMOTEENN** identifica qué variables son más importantes 
        para predecir el riesgo de tsunami. Esto nos ayuda a entender qué factores sísmicos 
        son más relevantes en la generación de tsunamis.
        """)
        
        # Mostrar imagen de importancia de variables
        try:
            st.image("images/importancia_variables.png", 
                     caption="Importancia de Variables - Random Forest + SMOTEENN",
                     use_container_width=True)
        except:
            st.warning("Imagen no disponible: importancia_variables.png")
        
        # Análisis de importancia de variables
        st.subheader("Análisis de Importancia")
        st.markdown("""
        **Top 3 Variables Más Importantes:**
        
        1. **Localización (38.6%):** La ubicación geográfica del sismo (coordenadas x, y, z) 
           es el factor más importante. Ciertos puntos del planeta (zonas de subducción, 
           fosas oceánicas) tienen mayor probabilidad de generar tsunamis.
        
        2. **Tipo de Magnitud (37.5%):** El método usado para medir la magnitud (`magnitude_type`) 
           es casi tan importante como la ubicación. Tipos como `mww` (momento sísmico) son 
           más relevantes para tsunamis que otros como `mb` (ondas internas).
        
        3. **Distancia al Océano (12.1%):** Sismos más cercanos a la costa tienen mayor 
           probabilidad de generar tsunamis. Esta variable es crítica para el modelo.
        
        **Observaciones Clave:**
        - Las **top 5 variables explican el 99.88%** de la importancia total
        - La **magnitud en sí** (8.3%) es menos importante que el tipo de magnitud
        - La **profundidad** (3.3%) tiene impacto menor de lo esperado
        - El **tipo de evento** (0.1%) es casi irrelevante (la mayoría son terremotos)
        
        **Conclusión:** La ubicación y el tipo de magnitud son los factores dominantes. 
        Esto confirma que no todos los sismos grandes generan tsunamis, sino que depende 
        de **dónde** ocurren y **cómo** se mide su magnitud.
        """)
        
        st.markdown("---")
        st.caption("Análisis basado en 20,000 eventos sísmicos con 3.6% de incidencia de tsunami")

    # Tsunami Prediction Tab
    with tab3:
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
        
        # Guardar temporalmente las coordenadas del último clic
        if map_data is not None and map_data.get('last_clicked') is not None:
            st.session_state.temp_lat = map_data['last_clicked']['lat']
            st.session_state.temp_lon = map_data['last_clicked']['lng']
        
        # Botón para confirmar la ubicación seleccionada
        if st.button("✅ Confirmar Ubicación", type="primary"):
            if 'temp_lat' in st.session_state and 'temp_lon' in st.session_state:
                # Asegurarse de que los valores estén dentro del rango válido
                clicked_lat = max(-90.0, min(90.0, st.session_state.temp_lat))
                clicked_lon = max(-180.0, min(180.0, st.session_state.temp_lon))
                
                st.session_state.selected_lat = clicked_lat
                st.session_state.selected_lon = clicked_lon
                st.success(f"📍 Ubicación confirmada: Lat {clicked_lat:.4f}, Lon {clicked_lon:.4f}")
                st.rerun()
            else:
                st.warning("⚠️ Por favor, haz clic en el mapa primero")
        
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
        
        if st.button("Predecir Riesgo de Tsunami", type="primary"):
            # Calcular significance automáticamente basado en magnitud y profundidad
            # Fórmula aproximada: magnitud * 100 - profundidad (valores típicos 0-2000)
            auto_significance = max(0, min(2000, magnitude * 100 - depth * 0.5))
            
            # Create input dataframe with column names matching the training data
            input_data = pd.DataFrame({
                'latitude': [latitude],
                'longitude': [longitude],
                'magnitude': [magnitude],
                'depth': [depth],
                'magType': [magType],
                'significance': [auto_significance],
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
                
                # Consejos de seguridad según el resultado
                st.markdown("---")
                st.subheader("Consejos de Seguridad")
                
                if prediction == 1:
                    # Alto riesgo de tsunami
                    st.warning("MEDIDAS DE EMERGENCIA - RIESGO ALTO")
                    st.markdown("""
                    **Acciones inmediatas recomendadas:**
                    
                    1. **Evacuación urgente:** Diríjase inmediatamente a zonas altas (mínimo 30 metros sobre el nivel del mar) o al menos 3 km tierra adentro.
                    
                    2. **Alertas oficiales:** Manténgase atento a las comunicaciones de las autoridades locales y sistemas de alerta de tsunami.
                    
                    3. **Rutas de escape:** Identifique las rutas de evacuación más cercanas y sígalas sin demora. No espere a ver el agua.
                    
                    4. **Lleve lo esencial:** Solo documentos importantes, medicamentos, agua y elementos de supervivencia básicos.
                    
                    5. **Ayude a otros:** Alerte a vecinos, especialmente a personas con movilidad reducida, niños y ancianos.
                    
                    6. **Evite la costa:** Aléjese de playas, puertos, bahías y desembocaduras de ríos. Las primeras olas pueden llegar en minutos.
                    
                    7. **No regrese:** Permanezca en zona segura hasta que las autoridades confirmen que el peligro ha pasado. Pueden ocurrir múltiples olas.
                    
                    **IMPORTANTE:** Los tsunamis pueden llegar en 10-30 minutos después de un sismo costero fuerte. Actúe de inmediato.
                    """)
                else:
                    # Bajo riesgo de tsunami
                    st.info("MEDIDAS PREVENTIVAS - RIESGO BAJO")
                    st.markdown("""
                    **Recomendaciones generales:**
                    
                    1. **Mantenga la calma:** Aunque el riesgo es bajo, es importante estar preparado ante cualquier eventualidad.
                    
                    2. **Monitoreo:** Siga las noticias y comunicados oficiales. Verifique si hay actualizaciones de las autoridades locales.
                    
                    3. **Preparación:** Si vive en zona costera, revise que su plan de evacuación familiar esté actualizado.
                    
                    4. **Inspección:** Si hubo un sismo, revise su hogar en busca de daños estructurales o fugas de gas.
                    
                    5. **Kit de emergencia:** Verifique que tenga agua, alimentos no perecederos, linterna, radio a pilas y botiquín.
                    
                    6. **Comunicación:** Mantenga su celular cargado y tenga a mano números de emergencia locales.
                    
                    7. **Precaución costera:** Aunque el riesgo es bajo, evite acercarse demasiado al mar inmediatamente después del sismo.
                    
                    **RECUERDE:** Incluso con riesgo bajo, mantener la preparación es fundamental en zonas sísmicas.
                    """)
                
                # Show input summary
                with st.expander("Ver detalles del sismo ingresado"):
                    st.write(f"**Ubicación:** Lat {latitude}°, Lon {longitude}°")
                    st.write(f"**Magnitud:** {magnitude} ({magType})")
                    st.write(f"**Profundidad:** {depth} km")
                
            except Exception as e:
                st.error(f"Error al realizar la predicción: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()
