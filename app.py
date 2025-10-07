# app.py
import os
import json
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import warnings   

from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px

# ---------------------------------------------------
# CARGA DE DATOS
# ---------------------------------------------------
warnings.filterwarnings("ignore") 
shapefile_path = r"C:\Users\maria\OneDrive\Escritorio\Visualizacion_de_datos\MGN2021_DPTO_POLITICO\MGN_DPTO_POLITICO.shp"
gdf = gpd.read_file(shapefile_path, encoding='utf-8')

df = pd.read_csv(r'C:\Users\maria\OneDrive\Escritorio\Visualizacion_de_datos\Actividades\corte2\DataViz_despliegue_dash\data\MEN_ESTADISTICAS_EN_EDUCACION.csv', sep = ",")

df = df[['AÑO', 'CÓDIGO_DEPARTAMENTO', 'DEPARTAMENTO', 'POBLACIÓN_5_16', 'DESERCIÓN', 'DESERCIÓN_TRANSICIÓN', 'DESERCIÓN_PRIMARIA', 'DESERCIÓN_SECUNDARIA', 'DESERCIÓN_MEDIA']]

available_departamentos = sorted(df['DEPARTAMENTO'].unique())

df.columns = df.columns.str.lower().str.replace("ñ", "n")
df = df.applymap(lambda x: x.lower().replace("ñ", "n") if isinstance(x, str) else x)
df.columns = df.columns.str.replace(" ", "_")

from unidecode import unidecode

df.columns = [unidecode(col) for col in df.columns]

df = df.applymap(lambda x: unidecode(x) if isinstance(x, str) else x)

df = df.dropna()

gdf.columns = gdf.columns.str.lower().str.replace("ñ", "n")
gdf = gdf.applymap(lambda x: x.lower().replace("ñ", "n") if isinstance(x, str) else x)
gdf.columns = gdf.columns.str.replace(" ", "_")
from unidecode import unidecode

gdf.columns = [unidecode(col) for col in gdf.columns]

gdf = gdf.applymap(lambda x: unidecode(x) if isinstance(x, str) else x)

df["departamento"] = df["departamento"].replace('bogota, d,c,', 'bogota, d.c.')
df = df.rename(columns={"departamento": "dpto_cnmbr"})

data = pd.merge(gdf, df, on ='dpto_cnmbr', how = 'outer')

data['ano'] = data['ano'].astype('object')
data['codigo_departamento'] = data['codigo_departamento'].astype('object')
data['dpto_cnmbr'] = data['dpto_cnmbr'].astype('object')
data['poblacion_5_16'] = data['poblacion_5_16'].astype('int')
data['codigo_departamento'] = data['codigo_departamento'].astype(str).str.zfill(2)


cols_porcentaje = [
    "desercion",
    "desercion_transicion",
    "desercion_primaria",
    "desercion_secundaria",
    "desercion_media"
]

for col in cols_porcentaje:
    data[col] = (
        data[col]
        .str.replace("%", "", regex=False)  
        .astype(float)                      
        / 100                               
    )

df = data.drop(columns='geometry')
gdf = gpd.GeoDataFrame(data, geometry='geometry')


# Crear el GeoDataFrame principal (con geometría)
gdf = gpd.GeoDataFrame(data, geometry='geometry')

# Crear versión sin geometría para cálculos
df = data.drop(columns=['geometry']).copy()

# Asegurar que la columna de años sea numérica
df['ano'] = df['ano'].astype(int)

# ---------------------------------------------------
# DATOS PARA CONTROLES
# ---------------------------------------------------
available_years = sorted(df['ano'].unique())[-3:]
available_departamentos = sorted(df['dpto_cnmbr'].unique())

# ---------------------------------------------------
# ETIQUETAS AMIGABLES PARA MOSTRAR EN DASH
# ---------------------------------------------------
labels_dict = {
    "ano": "Año",
    "codigo_departamento": "Código Departamento",
    "dpto_cnmbr": "Departamento",
    "poblacion_5_16": "Población",
    "desercion": "Deserción Total",
    "desercion_transicion": "Deserción Transición",
    "desercion_primaria": "Deserción Primaria",
    "desercion_secundaria": "Deserción Secundaria",
    "desercion_media": "Deserción Media"
}

# ---------------------------------------------------
# CORRECCIÓN Y NORMALIZACIÓN DE NOMBRES DE DEPARTAMENTOS
# ---------------------------------------------------
reemplazos_departamentos = {
    "amazonas": "Amazonas",
    "antioquia": "Antioquia",
    "arauca": "Arauca",
    "archipielago de san andres, providencia y santa catalina": "Archipiélago de San Andrés, Providencia y Santa Catalina",
    "atlantico": "Atlántico",
    "bogota, d.c.": "Bogotá D.C.",
    "bogota, d,c,": "Bogotá D.C.",
    "bolivar": "Bolívar",
    "boyaca": "Boyacá",
    "caldas": "Caldas",
    "caqueta": "Caquetá",
    "casanare": "Casanare",
    "cauca": "Cauca",
    "cesar": "Cesar",
    "choco": "Chocó",
    "cordoba": "Córdoba",
    "cundinamarca": "Cundinamarca",
    "guainia": "Guainía",
    "guaviare": "Guaviare",
    "huila": "Huila",
    "la guajira": "La Guajira",
    "magdalena": "Magdalena",
    "meta": "Meta",
    "narino": "Nariño",
    "norte de santander": "Norte de Santander",
    "putumayo": "Putumayo",
    "quindio": "Quindío",
    "risaralda": "Risaralda",
    "santander": "Santander",
    "sucre": "Sucre",
    "tolima": "Tolima",
    "valle del cauca": "Valle del Cauca",
    "vaupes": "Vaupés",
    "vichada": "Vichada"
}

# ---------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------
def kpis_for_selection(df_sel):
    total_pop = int(df_sel['poblacion_5_16'].sum(skipna=True))

    if df_sel['poblacion_5_16'].sum() > 0:
        weighted_desercion = (df_sel['desercion'] * df_sel['poblacion_5_16']).sum() / df_sel['poblacion_5_16'].sum()
    else:
        weighted_desercion = df_sel['desercion'].mean()

    dept_with_max = df_sel.loc[df_sel['desercion'].idxmax()] if not df_sel['desercion'].isna().all() else None
    dept_with_min = df_sel.loc[df_sel['desercion'].idxmin()] if not df_sel['desercion'].isna().all() else None

    return {
    'total_pop': total_pop,
    'weighted_desercion': weighted_desercion,
    'max': dept_with_max,
    'min': dept_with_min
    }

available_departamentos = sorted(df['dpto_cnmbr'].replace(reemplazos_departamentos).unique())

# ---------------------------------------------------
# DASH APP
# ---------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

card_style = {"borderRadius": "8px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)", "padding": "14px"}

app.layout = dbc.Container([

    dbc.Row([dbc.Col(html.H2("Dashboard: Tasa de Deserción Escolar por Departamento"), md=10)], align="center", className="py-2"),

    # Filtros
    dbc.Row([
        dbc.Col([
            dbc.Label("Año"),
            dcc.Dropdown(
                id="year_dropdown",
                options=[{"label": str(y), "value": y} for y in available_years],
                value=available_years[-1],
                clearable=False
            )
        ], md=5),

        dbc.Col([
            dbc.Label("Departamento"),
            dcc.Dropdown(
    id="dept_dropdown",
    options=[{"label": c, "value": c} for c in available_departamentos],
    value=None,
    placeholder="Todos"
    )

        ], md=5),

    ], className="mb-3"),

    # KPIs
    dbc.Row([
        dbc.Col(dbc.Card([html.Div([html.H6("Población 5-16 (total)"), html.H3(id="kpi_pop")], style=card_style)]), md=3),
        dbc.Col(dbc.Card([html.Div([html.H6("Deserción promedio (ponderada)"), html.H3(id="kpi_desercion")], style=card_style)]), md=3),
        dbc.Col(dbc.Card([html.Div([html.H6("Mayor deserción"), html.Div(id="kpi_max", style={"fontWeight":"600"})], style=card_style)]), md=3),
        dbc.Col(dbc.Card([html.Div([html.H6("Menor deserción"), html.Div(id="kpi_min", style={"fontWeight":"600"})], style=card_style)]), md=3)
    ], className="mb-3"),

    # Gráficos
    dbc.Row([
    dbc.Col(dbc.Card([dcc.Graph(id="map_choropleth", config={"displayModeBar": False})], body=True), md=4),
    dbc.Col(dbc.Card([dcc.Graph(id="bar_levels", config={"displayModeBar": False})], body=True), md=5),
    dbc.Col(dbc.Card([dcc.Graph(id="line_trend", config={"displayModeBar": False})], body=True), md=3)
    ], className="mb-3"),

    # Tabla
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H6("Tabla por departamento"),
            dash_table.DataTable(
                id='table_dept',
                columns=[{'name': col, 'id': col} for col in [
                    'codigo_departamento', 'dpto_cnmbr', 'poblacion_5_16',
                    'desercion', 'desercion_transicion', 'desercion_primaria',
                    'desercion_secundaria', 'desercion_media'
                ] if col in df.columns],
                page_size=12,
                sort_action='native',
                filter_action='native',
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '6px'},
            )
        ], body=True), md=12)
    ])
], fluid=True)

# ---------------------------------------------------
# CALLBACKS
# ---------------------------------------------------
@app.callback(
    Output("kpi_pop", "children"),
    Output("kpi_desercion", "children"),
    Output("kpi_max", "children"),
    Output("kpi_min", "children"),
    Output("map_choropleth", "figure"),
    Output("bar_levels", "figure"),
    Output("line_trend", "figure"),
    Output("table_dept", "data"),
    Input("year_dropdown", "value"),
    Input("dept_dropdown", "value")
)
def update_dashboard(year, dept_name):
    import plotly.graph_objects as go  # import local para fallback si hace falta
    # --- Normalizar year ---
    year_int = int(year) if year is not None else None

    # --- Filtro principal ---
    df_sel = df.copy()
    if year_int is not None:
        df_sel = df_sel[df_sel['ano'] == year_int]
    if dept_name:
        df_sel = df_sel[df_sel['dpto_cnmbr'] == dept_name]

    # --- gdf para mapa ---
    gdf_year = gdf.copy()
    if year_int is not None:
        gdf_year = gdf_year[gdf_year['ano'] == year_int]
    if dept_name:
        gdf_year = gdf_year[gdf_year['dpto_cnmbr'] == dept_name]

    # --- KPIs (proteger contra df_sel vacío) ---
    if df_sel.empty:
        kpi_pop = "0"
        kpi_des = "N/A"
        max_text = "N/A"
        min_text = "N/A"
    else:
        # --- KPIs ---
        k = kpis_for_selection(df_sel)

        kpi_pop = f"{k['total_pop']:,}"
        kpi_des = f"{k['weighted_desercion']:.2%}" if not np.isnan(k['weighted_desercion']) else "N/A"

        # Reemplazar nombres de departamentos por versión "bonita"
        def nombre_amigable(depto):
            if depto in reemplazos_departamentos:
                return reemplazos_departamentos[depto]
            return depto

        max_text = f"{nombre_amigable(k['max']['dpto_cnmbr'])} — {k['max']['desercion']:.2%}" if k['max'] is not None else "N/A"
        min_text = f"{nombre_amigable(k['min']['dpto_cnmbr'])} — {k['min']['desercion']:.2%}" if k['min'] is not None else "N/A"

    # --- Mapa (con fallback si no hay datos) ---
    try:
        if not gdf_year.empty:
            gjson = json.loads(gdf_year.to_json())
            fig_map = px.choropleth_mapbox(
                gdf_year,
                geojson=gjson,
                locations="codigo_departamento",
                featureidkey="properties.codigo_departamento",
                color="desercion",
                hover_name="dpto_cnmbr",
                hover_data={"desercion":":.2%"},
                mapbox_style="open-street-map",
                center={"lat": 4.6, "lon": -74.1},
                zoom=4.2,
                opacity=0.7,
                labels={'desercion': 'Deserción'}
            )
            fig_map.update_layout(margin={"r":0,"t":10,"l":0,"b":0})
        else:
            # Figura vacía con mensaje
            fig_map = go.Figure()
            fig_map.update_layout(
                annotations=[dict(text="No hay datos para el año/departamento seleccionado",
                                  x=0.5, y=0.5, showarrow=False, font=dict(size=14))],
                margin={"r":0,"t":10,"l":0,"b":0}
            )
    except Exception:
        fig_map = go.Figure()
        fig_map.update_layout(
            annotations=[dict(text="Error generando el mapa", x=0.5, y=0.5, showarrow=False)],
            margin={"r":0,"t":10,"l":0,"b":0}
        )

    # Aplicar reemplazos al dataframe principal
    df["dpto_cnmbr"] = df["dpto_cnmbr"].replace(reemplazos_departamentos)
    gdf["dpto_cnmbr"] = gdf["dpto_cnmbr"].replace(reemplazos_departamentos)

    # --- BARRAS: Deserción por nivel educativo (nuevo) ---
    # Comportamiento:
    #  - Si hay un departamento seleccionado: x = año, color = nivel (evolución por nivel)
    #  - Si NO hay departamento (comparar departamentos en un año): x = nivel, color = departamento (con nombres truncados en la leyenda/axis)
    max_len = 15  # <-- ajustar aquí la longitud máxima permitida antes de truncar

    if dept_name:
        # Evolución temporal por niveles para el departamento seleccionado
        bar_df = df_sel[[
            'ano', 'desercion_transicion', 'desercion_primaria', 'desercion_secundaria', 'desercion_media'
        ]].copy()
        if bar_df.empty:
            fig_bar = go.Figure()
            fig_bar.update_layout(annotations=[dict(text="No hay datos para este departamento", x=0.5, y=0.5, showarrow=False)])
        else:
            bar_df = bar_df.melt(id_vars='ano', var_name='nivel_educativo', value_name='tasa_desercion')
            bar_df['nivel_educativo'] = bar_df['nivel_educativo'].str.replace('desercion_', '').str.capitalize()
            fig_bar = px.bar(
                bar_df,
                x='ano',
                y='tasa_desercion',
                color='nivel_educativo',
                barmode='group',
                title=f"Evolución de la deserción por nivel educativo — {dept_name.title()}",
                labels={"ano": "Año", "tasa_desercion": "Tasa de deserción", "nivel_educativo": "Nivel educativo"}
            )
            fig_bar.update_layout(yaxis_tickformat=".0%", xaxis_title="Año", legend_title="Nivel educativo", margin=dict(t=35,b=20))
            fig_bar.update_traces(texttemplate='%{y:.1%}', textposition='outside')

    else:
        # Comparación por nivel entre departamentos (año seleccionado)
        bar_df = df_sel[[
            'dpto_cnmbr', 'desercion_transicion', 'desercion_primaria', 'desercion_secundaria', 'desercion_media'
        ]].copy()
        if bar_df.empty:
            fig_bar = go.Figure()
            fig_bar.update_layout(annotations=[dict(text="No hay datos para el año seleccionado", x=0.5, y=0.5, showarrow=False)])
        else:
            bar_df = bar_df.melt(id_vars='dpto_cnmbr', var_name='nivel_educativo', value_name='tasa_desercion')
            bar_df['nivel_educativo'] = bar_df['nivel_educativo'].str.replace('desercion_', '').str.capitalize()
            # guardamos nombre completo y creamos versión truncada para mostrar en la leyenda/ejes
            bar_df['dpto_cnmbr_full'] = bar_df['dpto_cnmbr'].astype(str)
            def _truncate(s, n=max_len):
                return s if len(s) <= n else s[:n-3] + '...'
            bar_df['dpto_cnmbr_trunc'] = bar_df['dpto_cnmbr_full'].apply(_truncate)

            # Aplicar nombres “bonitos” a la columna de departamentos
            bar_df['dpto_cnmbr'] = bar_df['dpto_cnmbr'].replace(reemplazos_departamentos)

            # Mantener nombre completo y versión truncada para mostrar en la leyenda/ejes
            bar_df['dpto_cnmbr_full'] = bar_df['dpto_cnmbr']
            def _truncate(s, n=max_len):
                return s if len(s) <= n else s[:n-3] + '...'
            bar_df['dpto_cnmbr_trunc'] = bar_df['dpto_cnmbr_full'].apply(_truncate)

            # Usamos la versión truncada como color/leyenda; el hover mostrará el nombre completo
            fig_bar = px.bar(
                bar_df,
                x='nivel_educativo',
                y='tasa_desercion',
                color='dpto_cnmbr_trunc',
                barmode='group',
                title=f"Comparación por nivel educativo — Año {year_int if year_int is not None else ''}",
                labels={"nivel_educativo": "Nivel educativo", 
                        "tasa_desercion": "Tasa de deserción", 
                        "dpto_cnmbr_trunc": "Departamento"
                        }
            )
            # Mostrar nombre completo en hover y formato %
            fig_bar.update_traces(hovertemplate='<b>%{customdata[0]}</b><br>Nivel: %{x}<br>Deserción: %{y:.2%}',
                                  customdata=bar_df[['dpto_cnmbr_full']].values,
                                  texttemplate='%{y:.1%}', textposition='outside')
            fig_bar.update_layout(yaxis_tickformat=".2%", legend_title="Departamento", margin=dict(t=35,b=20))

    # --- Tendencia de deserción --- 
    if dept_name:
        # Mostrar evolución por departamento seleccionado en todos los años disponibles
        trend = df[df['dpto_cnmbr'] == dept_name].groupby('ano').apply(
            lambda g: (g['desercion'] * g['poblacion_5_16']).sum() / g['poblacion_5_16'].sum()
        ).reset_index(name='desercion_ponderada')
    else:
        # Tendencia nacional (todos los departamentos) a lo largo de los años
        trend = df.groupby('ano').apply(
            lambda g: (g['desercion'] * g['poblacion_5_16']).sum() / g['poblacion_5_16'].sum()
        ).reset_index(name='desercion_ponderada')

    # Crear figura con nombres de ejes correctos y porcentaje con 2 decimales
    fig_line = px.line(
        trend,
        x="ano",
        y="desercion_ponderada",
        markers=True,
        labels={"ano": "Año", "desercion_ponderada": "Deserción ponderada"}
    )
    fig_line.update_layout(
        yaxis_tickformat=".2%",
        xaxis=dict(
            tickmode="array",
            tickvals=trend["ano"].tolist(),
            ticktext=[str(a) for a in trend["ano"].tolist()]
        )
    )


    # --- Tabla ---
    table_df = df_sel[[ 
        'codigo_departamento', 'dpto_cnmbr', 'poblacion_5_16',
        'desercion', 'desercion_transicion', 'desercion_primaria',
        'desercion_secundaria', 'desercion_media'
    ]].fillna("")
    table_data = table_df.to_dict('records')

    # --- Return ---
    return kpi_pop, kpi_des, max_text, min_text, fig_map, fig_bar, fig_line, table_data

# ---------------------------------------------------
# RUN
# ---------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=True, host="0.0.0.0", port=port)
