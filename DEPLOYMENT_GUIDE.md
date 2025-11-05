# 🚀 Guía de Despliegue en Streamlit Cloud

## Paso 1: Preparar el Repositorio de GitHub

### 1.1 Agregar archivos al repositorio

```bash
# Agregar todos los archivos nuevos
git add .

# Hacer commit de los cambios
git commit -m "Preparar aplicación para despliegue en Streamlit Cloud"

# Subir cambios a GitHub
git push origin main
```

### 1.2 Verificar que estos archivos estén en tu repositorio:

✅ **Archivos obligatorios:**
- `app.py` - Aplicación principal
- `requirements.txt` - Dependencias de Python
- `Sismos_data.csv` - Dataset principal
- `README.md` - Documentación

✅ **Archivos recomendados (ya creados):**
- `.gitignore` - Archivos a ignorar
- `packages.txt` - Dependencias del sistema (para GeoPandas)
- `.streamlit/config.toml` - Configuración de la app

## Paso 2: Desplegar en Streamlit Cloud

### 2.1 Ir a Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Haz clic en **"Sign in"** o **"Get started"**
3. Inicia sesión con tu cuenta de GitHub

### 2.2 Crear Nueva App

1. Haz clic en **"New app"** o **"Deploy an app"**
2. Selecciona tu repositorio: `Xebax81/Tsunamy_risk_predictor`
3. Configura los siguientes campos:

   ```
   Repository: Xebax81/Tsunamy_risk_predictor
   Branch: main
   Main file path: app.py
   App URL (custom): tsunami-risk-predictor (o el nombre que prefieras)
   ```

4. Haz clic en **"Deploy!"**

### 2.3 Esperar el Despliegue

- ⏱️ El primer despliegue puede tardar 3-5 minutos
- 📦 Streamlit Cloud instalará todas las dependencias de `requirements.txt`
- 🔧 También instalará las dependencias del sistema de `packages.txt`
- ✅ Una vez completado, verás tu app en vivo

## Paso 3: Tu App Estará en Vivo 🎉

Tu aplicación estará disponible en:
```
https://[tu-nombre-de-app].streamlit.app
```

Por ejemplo:
```
https://tsunami-risk-predictor.streamlit.app
```

## 🔧 Configuración Avanzada (Opcional)

### Secrets Management

Si necesitas claves API o credenciales:

1. En Streamlit Cloud, ve a tu app
2. Haz clic en **"Settings"** → **"Secrets"**
3. Agrega tus secrets en formato TOML:

```toml
[api_keys]
my_api_key = "tu-clave-secreta"
```

4. En tu código, accede con: `st.secrets["api_keys"]["my_api_key"]`

### Variables de Entorno

En el archivo `.streamlit/config.toml` puedes configurar:
- Tema de colores
- Puerto del servidor
- Configuraciones de caché
- Etc.

## 🐛 Solución de Problemas

### Problema: Error al instalar GeoPandas

**Solución:** Ya incluimos `packages.txt` con las dependencias del sistema necesarias.

### Problema: Archivo demasiado grande (>100MB)

**Solución:** 
1. Usa Git LFS para archivos grandes:
```bash
git lfs install
git lfs track "*.csv"
git add .gitattributes
git commit -m "Agregar Git LFS"
```

2. O descarga los datos desde una URL en tiempo de ejecución en `app.py`

### Problema: App muy lenta

**Soluciones:**
- ✅ Ya usas `@st.cache_data` y `@st.cache_resource` (correcto)
- ✅ Ya tienes `Sismos_data_processed.csv` para evitar reprocesamiento
- Considera reducir el tamaño del dataset si es necesario

## 🔄 Actualizaciones Futuras

Cada vez que hagas cambios:

```bash
# Hacer cambios en app.py u otros archivos
git add .
git commit -m "Descripción de cambios"
git push origin main
```

**Streamlit Cloud detectará automáticamente los cambios y redesplegará tu app** 🚀

## 📊 Monitoreo

En Streamlit Cloud puedes:
- 📈 Ver analytics de uso
- 📝 Ver logs en tiempo real
- ⚙️ Reiniciar la app manualmente
- 🔒 Configurar privacidad (público/privado)

## 🎓 Recursos Adicionales

- [Documentación Streamlit Cloud](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit Forums](https://discuss.streamlit.io/)
- [Streamlit Cheat Sheet](https://docs.streamlit.io/library/cheatsheet)

## ✅ Checklist Final

Antes de desplegar, verifica:

- [ ] `requirements.txt` tiene todas las dependencias
- [ ] `packages.txt` incluye dependencias del sistema (GeoPandas)
- [ ] `app.py` funciona localmente sin errores
- [ ] Dataset `Sismos_data.csv` está en el repositorio
- [ ] `.gitignore` está configurado correctamente
- [ ] `README.md` está actualizado
- [ ] Todo está en GitHub (git push)
- [ ] Cuenta de GitHub conectada a Streamlit Cloud

## 🎉 ¡Listo!

Una vez desplegado, comparte tu app:
- En tu README (ya actualizado con el enlace)
- En LinkedIn, Twitter, etc.
- En tu portafolio profesional

---

**¿Necesitas ayuda?** 
- 📧 Contacta al soporte de Streamlit
- 💬 Pregunta en [Streamlit Forums](https://discuss.streamlit.io/)
