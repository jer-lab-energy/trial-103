import os
import streamlit as st
import pandas as pd
import geopandas as gpd
import leafmap.foliumap as leafmap
try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except ImportError:
    HAS_ST_FOLIUM = False
from shapely.geometry import Point
import base64
from io import BytesIO

# --- Page config ---
st.set_page_config(layout="wide", page_title="Dynamic Geo Dashboard")

st.title("Dynamic Geo Dashboard")
st.markdown("Upload Excel/CSV and GeoJSON/KMZ layers to visualize them interactively. " \
"Upload smaller files first and the largest ones last for better performance :)")


# --- Constants & helpers for multi-map support ---
MAX_MAPS = 6
KMZ_COLOR_CYCLE = [
    "#3388ff",  # blue (current default)
    "#2ecc71",  # green
    "#e74c3c",  # red
    "#f1c40f",  # yellow
    "#9b59b6",  # purple
    "#ff7f0e",  # orange
]


def map_key(key: str, map_idx: int) -> str:
    """Create a unique key for widgets/state per map instance."""
    return f"{key}_map_{map_idx}"


def ensure_base_state():
    if "map_count" not in st.session_state:
        st.session_state.map_count = 1
    st.session_state.setdefault("cached_excel_id", None)
    st.session_state.setdefault("cached_excel_df", None)
    st.session_state.setdefault("geojson_cache", {})
    st.session_state.setdefault("export_mode_global", False)
    st.session_state.setdefault("kmz_color_assignments", {})


def get_kmz_color_for_file(file_id: str) -> str:
    """Assign a stable color per KMZ file (cycles the palette)."""
    assignments = st.session_state.setdefault("kmz_color_assignments", {})
    if file_id in assignments:
        return assignments[file_id]
    color = KMZ_COLOR_CYCLE[len(assignments) % len(KMZ_COLOR_CYCLE)]
    assignments[file_id] = color
    st.session_state["kmz_color_assignments"] = assignments
    return color


def ensure_map_defaults(map_idx: int, df: pd.DataFrame):
    """Seed sensible defaults for a map based on the uploaded dataframe."""
    label_default = df.columns[0]

    st.session_state.setdefault(map_key("name", map_idx), f"Map {map_idx + 1}")
    st.session_state.setdefault(map_key("label_col", map_idx), label_default)
    st.session_state.setdefault(map_key("filter_col", map_idx), "(no filter)")
    st.session_state.setdefault(map_key("color_mode", map_idx), "Single color")
    st.session_state.setdefault(map_key("category_col", map_idx), None)
    st.session_state.setdefault(map_key("marker_color", map_idx), "#3388ff")
    st.session_state.setdefault(map_key("marker_shape", map_idx), "Circle")
    st.session_state.setdefault(map_key("marker_radius", map_idx), 6)
    st.session_state.setdefault(map_key("marker_opacity", map_idx), 0.8)
    st.session_state.setdefault(map_key("outline_color", map_idx), "#000000")
    st.session_state.setdefault(map_key("heatmap_toggle", map_idx), False)
    st.session_state.setdefault(map_key("heatmap_radius", map_idx), 20)
    st.session_state.setdefault(map_key("heatmap_blur", map_idx), 15)
    st.session_state.setdefault(map_key("heatmap_opacity", map_idx), 0.4)
    st.session_state.setdefault(map_key("category_colors", map_idx), {})
    st.session_state.setdefault(map_key("category_icons", map_idx), {})
    st.session_state.setdefault(map_key("labels_on", map_idx), True)
    st.session_state.setdefault(map_key("label_offsets", map_idx), {})
    st.session_state.setdefault(map_key("custom_icon_bytes", map_idx), None)
    st.session_state.setdefault(map_key("custom_icon_dims", map_idx), None)
    st.session_state.setdefault(map_key("custom_icon_size", map_idx), 24)
    st.session_state.setdefault(map_key("map_height", map_idx), calc_map_height(st.session_state.map_count))
    st.session_state.setdefault(map_key("map_zoom", map_idx), 6)
    st.session_state.setdefault(map_key("center_lat", map_idx), 54.5)
    st.session_state.setdefault(map_key("center_lon", map_idx), -2.0)
    st.session_state.setdefault(map_key("last_view", map_idx), {})
    st.session_state.setdefault(map_key("kmz_type_labels", map_idx), {"Point": "Point", "LineString": "Line", "Polygon": "Polygon"})

    # Keep selections valid if columns change
    for col_key, fallback in [
        ("label_col", label_default),
        ("filter_col", "(no filter)"),
        ("category_col", None),
    ]:
        session_key = map_key(col_key, map_idx)
        if st.session_state[session_key] not in df.columns and col_key != "filter_col":
            st.session_state[session_key] = fallback
    # Avoid empty map names
    if not st.session_state.get(map_key("name", map_idx)):
        st.session_state[map_key("name", map_idx)] = f"Map {map_idx + 1}"


def ensure_shared_coords(df: pd.DataFrame):
    """Latitude/longitude selection is shared across all maps."""
    lat_default = df.columns[0]
    lon_default = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    st.session_state.setdefault("shared_lat_col", lat_default)
    st.session_state.setdefault("shared_lon_col", lon_default)


def ensure_layer_settings(map_idx: int, geo_layers):
    """
    Seed per-layer styling/labels for KMZ/GeoJSON layers on a per-map basis.
    """
    key = map_key("layer_settings", map_idx)
    layer_state = st.session_state.setdefault(key, {})

    for layer in geo_layers:
        name = layer["name"]
        gdf = layer.get("gdf")
        geom_types = set(gdf.geometry.geom_type.unique()) if gdf is not None else set()
        default_color = layer.get("kmz_color") or "#3388ff"

        if name not in layer_state:
            layer_state[name] = {
                "color": default_color,
                "outline_color": "#000000",
                "weight": 2,
                "fill_opacity": 0.6,
                "marker_shape": "Circle",
                "marker_radius": 6,
                "marker_opacity": 0.9,
                "labels": {gt: gt for gt in geom_types if gt in {"Point", "LineString", "Polygon"}},
                "custom_icon_bytes": None,
                "custom_icon_dims": None,
                "custom_icon_size": 24,
            }
        else:
            # backfill any new keys
            layer_state[name].setdefault("color", default_color)
            layer_state[name].setdefault("outline_color", "#000000")
            layer_state[name].setdefault("weight", 2)
            layer_state[name].setdefault("fill_opacity", 0.6)
            layer_state[name].setdefault("marker_shape", "Circle")
            layer_state[name].setdefault("marker_radius", 6)
            layer_state[name].setdefault("marker_opacity", 0.9)
            layer_state[name].setdefault("labels", {gt: gt for gt in geom_types if gt in {"Point", "LineString", "Polygon"}})
            layer_state[name].setdefault("custom_icon_bytes", None)
            layer_state[name].setdefault("custom_icon_dims", None)
            layer_state[name].setdefault("custom_icon_size", 24)

    st.session_state[key] = layer_state


def get_image_dims(image_bytes):
    """Return (width, height) for an image byte payload if Pillow is available."""
    try:
        from PIL import Image
        with Image.open(BytesIO(image_bytes)) as img:
            return img.size
    except Exception:
        return None


def get_scaled_icon_size(image_bytes, target_px: int, stored_dims=None):
    """
    Scale an icon to target_px on its longest side while preserving aspect ratio.
    Falls back to square if dimensions cannot be read.
    """
    dims = stored_dims or get_image_dims(image_bytes)
    if not dims or dims[0] <= 0 or dims[1] <= 0:
        return target_px, target_px
    w, h = dims
    scale = target_px / max(w, h)
    sw = max(1, int(round(w * scale)))
    sh = max(1, int(round(h * scale)))
    return sw, sh


def ensure_legend_settings(map_idx: int):
    key = map_key("legend_settings", map_idx)
    st.session_state.setdefault(key, {
        "corner": "bottom_right",
        "offset_x": 30,
        "offset_y": 30,
        "width": 240,
        "max_height": 500,
        "title_font": "Segoe UI",
        "text_font": "Segoe UI",
        "title_size": 13,
        "text_size": 12,
        "logo_data": None,
        "text_offset": 6,
        "entry_spacing": 4,
        "custom_icon_size": 18,
    })


def render_legend_settings_controls(map_idx: int):
    """Legend styling controls (shared across Excel/Geo/KMZ)."""
    ensure_legend_settings(map_idx)
    legend_settings = st.session_state[map_key("legend_settings", map_idx)]

    st.sidebar.subheader("Legend Settings")

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        legend_settings["width"] = st.sidebar.slider("Width (px)", 160, 420, int(legend_settings.get("width", 240)), key=map_key("legend_width", map_idx))
        legend_settings["max_height"] = st.sidebar.slider("Max height (px)", 200, 900, int(legend_settings.get("max_height", 500)), key=map_key("legend_height", map_idx))
        legend_settings["title_size"] = st.sidebar.slider("Title size (px)", 10, 24, int(legend_settings.get("title_size", 13)), key=map_key("legend_title_size", map_idx))
        legend_settings["text_size"] = st.sidebar.slider("Text size (px)", 8, 20, int(legend_settings.get("text_size", 12)), key=map_key("legend_text_size", map_idx))
        legend_settings["text_offset"] = st.sidebar.slider("Text offset from icon (px)", 0, 30, int(legend_settings.get("text_offset", 6)), key=map_key("legend_text_offset", map_idx))
        legend_settings["entry_spacing"] = st.sidebar.slider("Spacing between entries (px)", 0, 20, int(legend_settings.get("entry_spacing", 4)), key=map_key("legend_entry_spacing", map_idx))
    with col_b:
        corner_options = ["bottom_right", "bottom_left", "top_right", "top_left"]
        current_corner = str(legend_settings.get("corner", "bottom_right")).lower().replace(" ", "_")
        if current_corner not in corner_options:
            current_corner = "bottom_right"
            legend_settings["corner"] = current_corner
        legend_settings["corner"] = st.sidebar.selectbox(
            "Placement",
            corner_options,
            index=corner_options.index(current_corner),
            key=map_key("legend_corner", map_idx),
        )
        legend_settings["offset_x"] = st.sidebar.slider("Horizontal offset (px)", 0, 200, int(legend_settings.get("offset_x", 30)), key=map_key("legend_offset_x", map_idx))
        legend_settings["offset_y"] = st.sidebar.slider("Vertical offset (px)", 0, 200, int(legend_settings.get("offset_y", 30)), key=map_key("legend_offset_y", map_idx))
        legend_settings["title_font"] = st.sidebar.text_input("Title font", legend_settings.get("title_font", "Segoe UI"), key=map_key("legend_title_font", map_idx))
        legend_settings["text_font"] = st.sidebar.text_input("Text font", legend_settings.get("text_font", "Segoe UI"), key=map_key("legend_text_font", map_idx))
        legend_settings["custom_icon_size"] = st.sidebar.slider(
            "Legend custom icon size (px)",
            10,
            64,
            int(legend_settings.get("custom_icon_size", 18)),
            key=map_key("legend_icon_size", map_idx),
        )

    logo_upload = st.sidebar.file_uploader("Legend logo (optional)", type=["png", "jpg", "jpeg"], key=map_key("legend_logo", map_idx))
    if logo_upload:
        legend_settings["logo_data"] = "data:image/png;base64," + base64.b64encode(logo_upload.read()).decode("ascii")

def get_map_name(map_idx: int) -> str:
    return st.session_state.get(map_key("name", map_idx), f"Map {map_idx + 1}")


def load_excel_cached(uploaded_file):
    """Cache Excel/CSV load so reruns avoid re-reading the file."""
    if not uploaded_file:
        return None
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("cached_excel_id") == file_id and st.session_state.get("cached_excel_df") is not None:
        return st.session_state["cached_excel_df"]

    data = uploaded_file.getvalue()
    buffer = BytesIO(data)
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(buffer)
    else:
        df = pd.read_csv(buffer)

    st.session_state["cached_excel_id"] = file_id
    st.session_state["cached_excel_df"] = df
    return df


def calc_map_height(count: int) -> int:
    if count == 1:
        return 700
    if count <= 2:
        return 550
    if count <= 4:
        return 500
    return 420


def cols_for_count(count: int) -> int:
    if count <= 1:
        return 1
    if count <= 4:
        return 2
    return 3

# --- KMZ Legend Extraction --- #
def extract_kmz_styles(gdf, max_entries=2000, color_override=None):
    """
    Extract true KML/KMZ style information:
    - Point icons (iconHref)
    - Line stroke color/width
    - Polygon fill + stroke
    - Also keep a small sample geometry for rendering mini-polygons
    """
    entries = []

    if gdf is None or gdf.empty:
        return entries

    seen_types = set()

    def base_type(g):
        if g in ("Point", "MultiPoint"):   return "Point"
        if g in ("LineString", "MultiLineString"):   return "LineString"
        if g in ("Polygon", "MultiPolygon"):   return "Polygon"
        return g

    for _, row in gdf.head(max_entries).iterrows():
        geom = row.geometry
        if geom is None:
            continue

        gtype = base_type(geom.geom_type)
        if gtype in seen_types:
            continue
        seen_types.add(gtype)

        # KMZ/KML common styling fields
        icon_href = row.get("href") or row.get("icon_href") or row.get("Icon") or None
        stroke = row.get("stroke") or row.get("color") or "#555555"
        stroke_width = row.get("stroke-width") or row.get("width") or 2
        fill = row.get("fill") or stroke
        if color_override:
            stroke = color_override
            fill = color_override

        entries.append({
            "geom_type": gtype,
            "stroke": stroke,
            "fill": fill,
            "stroke_width": stroke_width,
            "icon_href": icon_href,
            "color": stroke,
            "sample_geom": geom,  # ⭐ keep real geometry for legend SVG
            "label": gtype,
        })

    return entries

from fastkml import kml
import zipfile
import base64

def extract_kml_styles_from_kmz(kmz_path):
    """
    Reads KMZ, parses KML, extracts styles into a dict:
    { style_id : { 'icon_href', 'poly_color', 'line_color', 'line_width' } }
    """
    styles = {}

    with zipfile.ZipFile(kmz_path, "r") as z:
        # find KML file inside KMZ
        kml_name = [n for n in z.namelist() if n.lower().endswith(".kml")][0]
        kml_data = z.read(kml_name).decode("utf-8")

    doc = kml.KML()
    doc.from_string(kml_data)

    for feature in doc.features():
        for s in feature.styles():
            sid = s.id

            poly_color = None
            line_color = None
            line_width = 2
            icon_href = None

            if s.polystyle:
                # KML color format: aabbggrr → convert to CSS #rrggbbaa
                kml_col = s.polystyle.color
                if kml_col:
                    rr = kml_col[6:8]
                    gg = kml_col[4:6]
                    bb = kml_col[2:4]
                    aa = kml_col[0:2]
                    poly_color = f"#{rr}{gg}{bb}{aa}"

            if s.linestyle:
                kml_col = s.linestyle.color
                if kml_col:
                    rr = kml_col[6:8]
                    gg = kml_col[4:6]
                    bb = kml_col[2:4]
                    aa = kml_col[0:2]
                    line_color = f"#{rr}{gg}{bb}{aa}"
                if s.linestyle.width:
                    line_width = s.linestyle.width

            if s.iconstyle and s.iconstyle.icon and s.iconstyle.icon.href:
                icon_href = s.iconstyle.icon.href

            styles[sid] = {
                "poly_color": poly_color,
                "line_color": line_color,
                "line_width": line_width,
                "icon_href": icon_href,
            }

    return styles

def attach_kml_styles_to_gdf(gdf, kmz_path):
    styles = extract_kml_styles_from_kmz(kmz_path)

    if "styleUrl" not in gdf.columns:
        return gdf  # nothing to map

    def apply(row):
        sid = str(row.get("styleUrl", "")).replace("#", "")
        s = styles.get(sid, {})

        if "poly_color" in s:
            row["fill"] = s["poly_color"]
        if "line_color" in s:
            row["stroke"] = s["line_color"]
        if "line_width" in s:
            row["stroke-width"] = s["line_width"]
        if "icon_href" in s:
            row["icon_href"] = s["icon_href"]

        return row

    return gdf.apply(apply, axis=1)

# --- Mini Symbol Generator --- #
import shapely
from shapely.geometry import Polygon, LineString, MultiPolygon

def kmz_symbol_html(entry):
    """
    Draws a mini SVG version of the actual KMZ geometry and style.
    """
    g = entry.get("sample_geom")
    color = entry.get("color") or entry.get("stroke") or "#3388ff"
    stroke = color or entry.get("stroke") or "#3388ff"
    fill = entry.get("fill") or color
    width = entry.get("stroke_width", 2)
    icon = entry.get("icon_href")

    # # ---------- POINTS ----------
    # if entry["geom_type"] == "Point":
    #     if icon:
    #         return f"<img src='{icon}' style='width:20px;height:20px;object-fit:contain;'/>"
    #     return (
    #         f"<div style='width:14px;height:14px;border-radius:50%;"
    #         f"background:{fill};border:2px solid {stroke};'></div>"
    #     )
    
        # ---------- POINTS ----------
    if entry["geom_type"] == "Point":
        shape = entry.get("marker_shape", "Marker")
        outline = entry.get("outline_color") or stroke
        custom_size = int(entry.get("custom_icon_size", 18))
        if shape == "Circle":
            return (
                f"<div style='width:14px;height:14px;border-radius:50%;"
                f"background:{color};border:2px solid {outline};'></div>"
            )
        if shape == "Square":
            return (
                f"<div style='width:14px;height:14px;background:{color};"
                f"border:2px solid {outline};'></div>"
            )
        if shape == "Triangle":
            return (
                "<svg width='16' height='14' viewBox='0 0 16 14'>"
                f"<polygon points='8,0 16,14 0,14' style='fill:{color};stroke:{outline};stroke-width:1'/>"
                "</svg>"
            )
        if shape == "Custom Image" and entry.get("custom_icon_data"):
            icon_opacity = entry.get("icon_opacity", 1.0)
            return (
                f"<img src='{entry['custom_icon_data']}' style='width:{custom_size}px;height:{custom_size}px;object-fit:contain;opacity:{icon_opacity};'>"
            )
        # Default: Leaflet marker silhouette tinted to the layer color with white dot
        marker_url = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png"
        return (
            f"<div style='position:relative;width:14px;height:22px;'>"
            f"  <div style='width:14px;height:22px;background:{color};"
            f"  -webkit-mask:url({marker_url}) center / contain no-repeat;"
            f"  mask:url({marker_url}) center / contain no-repeat;'></div>"
            f"  <div style='position:absolute;top:7px;left:7px;width:4px;height:4px;"
            f"  transform:translate(-50%,-50%);border-radius:50%;background:white;'></div>"
            f"</div>"
        )

    # ---------- LINESTRINGS ----------
    if entry["geom_type"] == "LineString":
        return (
            f"<svg width='30' height='12'>"
            f"<line x1='2' y1='6' x2='28' y2='6' "
            f"stroke='{stroke}' stroke-width='{width}'/>"
            f"</svg>"
        )

    # ---------- POLYGONS ----------
    if entry["geom_type"] == "Polygon":
        try:
            poly = g if isinstance(g, Polygon) else list(g.geoms)[0]
            xs, ys = poly.exterior.xy

            # scale coords into 20x20 viewport
            minx, miny, maxx, maxy = poly.bounds
            scale = 18 / max(maxx - minx, maxy - miny)

            pts = []
            for x, y in zip(xs, ys):
                sx = (x - minx) * scale + 1
                sy = (maxy - y) * scale + 1
                pts.append(f"{sx},{sy}")

            pts_str = " ".join(pts)

            return (
                f"<svg width='22' height='22'>"
                f"<polygon points='{pts_str}' fill='none' stroke='{stroke}' stroke-width='1'/>"
                f"</svg>"
            )

        except Exception:
            # fallback simple polygon
            return (
                f"<div style='width:18px;height:18px;background:{fill or stroke};"
                f"border:1px solid {stroke};opacity:0.8;'></div>"
            )

    # fallback
    return f"<div style='width:16px;height:16px;background:{stroke};'></div>"


def render_point_marker(map_obj, lat, lon, settings):
    """Render a single point using per-layer settings."""
    import folium

    color = settings.get("color", "#3388ff")
    outline = settings.get("outline_color", "#000000")
    shape = settings.get("marker_shape", "Circle")
    radius = settings.get("marker_radius", 6)
    opacity = settings.get("marker_opacity", 0.9)

    if shape == "Circle":
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=outline,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=opacity,
            opacity=opacity,
        ).add_to(map_obj)
    elif shape == "Marker":
        marker_url = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png"
        html = (
            f"<div style='position:relative;width:20px;height:32px;'>"
            f"  <div style='width:20px;height:32px;background:{color};"
            f"  -webkit-mask:url({marker_url}) center / contain no-repeat;"
            f"  mask:url({marker_url}) center / contain no-repeat;'></div>"
            f"  <div style='position:absolute;top:11px;left:10px;width:5px;height:5px;"
            f"  transform:translate(-50%,-50%);border-radius:50%;background:white;'></div>"
            f"</div>"
        )
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(html=html, icon_size=(20, 32), icon_anchor=(10, 32)),
        ).add_to(map_obj)
    elif shape == "Square":
        size = radius * 2
        html = (
            f"<div style='width:{size}px;height:{size}px;"
            f"background:{color};border:2px solid {outline};"
            f"margin-left:-{size/2}px;margin-top:-{size/2}px;'></div>"
        )
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(html=html, icon_size=(size, size), icon_anchor=(size / 2, size / 2)),
        ).add_to(map_obj)
    elif shape == "Triangle":
        size = radius * 3
        html = (
            f"<svg width='{size}' height='{size}'>"
            f"<polygon points='{radius},{0} {radius*2},{size} 0,{size}' "
            f"style='fill:{color};stroke:{outline};stroke-width:1'/>"
            f"</svg>"
        )
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(html=html, icon_size=(size, size), icon_anchor=(radius, size)),
        ).add_to(map_obj)
    elif shape == "Custom Image" and settings.get("custom_icon_bytes"):
        import base64
        b64_icon = base64.b64encode(settings["custom_icon_bytes"]).decode("ascii")
        data_url = f"data:image/png;base64,{b64_icon}"
        icon_px = int(settings.get("custom_icon_size", 24))
        w, h = get_scaled_icon_size(settings["custom_icon_bytes"], icon_px, settings.get("custom_icon_dims"))
        folium.Marker(
            location=[lat, lon],
            icon=folium.CustomIcon(
                data_url,
                icon_size=(w, h),
                icon_anchor=(w / 2, h),
            ),
        ).add_to(map_obj)


# --- Sidebar: upload + basemap ---
st.sidebar.header("Upload your files")
excel_file = st.sidebar.file_uploader("Upload Excel/CSV file", type=["xlsx", "csv"])
geojson_files = st.sidebar.file_uploader(
    "Upload GeoJSON/KMZ files",
    type=["geojson", "json", "kmz", "kml", "application/vnd.google-earth.kmz"],
    accept_multiple_files=True,
)

st.sidebar.subheader("Map Style")
basemap_options = {
    "OpenStreetMap (Detailed)": "OpenStreetMap",
    "Esri Satellite": "Esri.WorldImagery",
    "Esri Terrain": "Esri.WorldTopoMap",
    "CartoDB Voyager": "CartoDB.Voyager",
    "CartoDB Positron (Light)": "CartoDB.Positron",
    "CartoDB Dark Matter": "CartoDB.DarkMatter",
    "OpenTopoMap": "OpenTopoMap",
    "Esri Streets": "Esri.WorldStreetMap",
}

selected_basemap_name = st.sidebar.selectbox(
    "Select a basemap style:",
    options=list(basemap_options.keys()),
    index=3,
    key="basemap_selector",
)

if "current_basemap" not in st.session_state:
    st.session_state.current_basemap = basemap_options[selected_basemap_name]

if st.session_state.current_basemap != basemap_options[selected_basemap_name]:
    st.session_state.current_basemap = basemap_options[selected_basemap_name]
    try:
        st.rerun()
    except AttributeError:
        st.rerun()


def make_map(map_idx: int, zoom: int = None) -> leafmap.Map:
    zoom_val = zoom if zoom is not None else st.session_state.get(map_key("map_zoom", map_idx), 6)
    center_lat = st.session_state.get(map_key("center_lat", map_idx), 54.5)
    center_lon = st.session_state.get(map_key("center_lon", map_idx), -2.0)
    return leafmap.Map(center=[center_lat, center_lon], zoom=zoom_val, tiles=st.session_state.current_basemap)


# --- Helper: Add floating non-overlapping labels for small datasets ---
def add_floating_labels(
    map_obj,
    data,
    label_field="name",
    source="Excel",
    max_labels=100,
    sidebar_prefix="Label",
):
    """
    Adds floating labels above points with configurable style and consistent vertical spacing.
    """
    import folium

    if data is None or len(data) == 0:
        return

    st.sidebar.subheader(f"{sidebar_prefix} Style Controls")

    box_width = st.sidebar.slider("Label box width (px)", 60, 300, 120, key=f"{sidebar_prefix}_box")
    text_size = st.sidebar.slider("Text size (px)", 8, 24, 11, key=f"{sidebar_prefix}_txt")
    text_color = st.sidebar.color_picker("Text colour", "#111111", key=f"{sidebar_prefix}_txt_color")
    bg_color = st.sidebar.color_picker("Box background", "#FFFFFF", key=f"{sidebar_prefix}_bg")
    line_color = st.sidebar.color_picker("Connector line colour", "#333333", key=f"{sidebar_prefix}_line")
    vertical_offset_m = st.sidebar.slider("Vertical offset above point (m)", 10, 15000, 40, key=f"{sidebar_prefix}_offset")
    label_spacing_m = st.sidebar.slider("Extra spacing between labels (m)", 0, 20000, 25, key=f"{sidebar_prefix}_spacing")

    deg_per_m = 1 / 111_000
    v_off = vertical_offset_m * deg_per_m
    spacing_deg = label_spacing_m * deg_per_m

    for i, row in enumerate(data[:max_labels]):
        lat, lon, label = row["lat"], row["lon"], str(row["label"])
        if pd.isna(lat) or pd.isna(lon) or not label.strip():
            continue

        lat_off = lat + v_off + (i * spacing_deg)
        lon_off = lon

        folium.PolyLine(
            locations=[[lat, lon], [lat_off, lon_off]],
            color=line_color,
            weight=1.5,
            opacity=0.7,
        ).add_to(map_obj)

        icon_html = (
            f'<div style="'
            f'background-color:{bg_color};'
            f'padding:2px 6px;border-radius:4px;'
            f'min-width:{box_width}px;max-width:{box_width}px;'
            f'font-size:{text_size}px;font-weight:600;color:{text_color};'
            f'text-align:center;white-space:normal;'
            f'box-shadow:0 0 3px rgba(0,0,0,0.25);">'
            f'{label}</div>'
        )

        folium.Marker(
            location=[lat_off, lon_off],
            icon=folium.DivIcon(
                html=icon_html,
                icon_size=(box_width, 24),
                icon_anchor=(box_width / 2, 0),
            ),
        ).add_to(map_obj)


# --- Helper: Optimize heavy GeoJSONs before mapping ---
def optimize_gdf(gdf, file_name: str = None, max_points: int = 200000, tolerance: float = 0.001):
    """
    Simplify and thin large GeoDataFrames for visualization.
    - max_points: maximum total vertices before simplification triggers
    - tolerance: simplification tolerance in degrees (~0.001 ≈ 100 m)
    """
    import shapely
    total_points = 0

    try:
        for geom in gdf.geometry:
            if geom and hasattr(geom, "coords"):
                total_points += len(geom.coords)
            elif geom and geom.geom_type.startswith("Multi"):
                for part in geom.geoms:
                    total_points += len(part.coords)
    except Exception:
        total_points = 0

    if total_points > max_points:
        st.warning(f"{file_name or 'Layer'} has {total_points:,} vertices – simplifying for better performance.")
        try:
            gdf["geometry"] = gdf["geometry"].simplify(tolerance=tolerance, preserve_topology=True)
        except Exception as e:
            st.warning(f"Could not simplify {file_name or 'layer'}: {e}")

    if len(gdf) > 8000:
        st.info(f"Displaying only first 8,000 features for {file_name or 'layer'} (of {len(gdf):,}).")
        gdf = gdf.head(8000)

    try:
        gdf["geometry"] = gdf["geometry"].apply(
            lambda g: shapely.set_precision(g, grid_size=0.0001) if g else g
        )
    except Exception:
        pass

    return gdf


# --- Helper: Ensure all non-geometry columns are JSON-serialisable for Folium/leafmap ---
def make_json_serialisable_gdf(gdf):
    import pandas as pd
    from datetime import datetime, date
    import numpy as np

    if gdf is None or gdf.empty:
        return gdf

    for col in gdf.columns:
        if col == "geometry":
            continue
        s = gdf[col]

        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_datetime64tz_dtype(s):
            gdf[col] = pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S%z").fillna("")
            continue

        if pd.api.types.is_timedelta64_dtype(s) or pd.api.types.is_period_dtype(s) or pd.api.types.is_categorical_dtype(s):
            gdf[col] = s.astype(str)
            continue

        if pd.api.types.is_object_dtype(s):
            def _coerce(v):
                if isinstance(v, (str, int, float, bool, type(None), list, dict)):
                    return v
                if isinstance(v, (pd.Timestamp, datetime, date)):
                    try:
                        return v.isoformat()
                    except Exception:
                        return str(v)
                if isinstance(v, (np.integer, np.floating, np.bool_)):
                    return v.item()
                return str(v)
            gdf[col] = s.apply(_coerce)

    return gdf

import zipfile
import geopandas as gpd
from fastkml import kml

def _extract_kml_bytes_from_kmz_bytes(kmz_bytes: bytes) -> bytes:
    with zipfile.ZipFile(BytesIO(kmz_bytes), "r") as z:
        kml_candidates = [n for n in z.namelist() if n.lower().endswith(".kml")]
        if not kml_candidates:
            raise ValueError("No .kml file found inside KMZ.")
        # pick first; you can get fancier if needed
        return z.read(kml_candidates[0])

def _iter_features(feature):
    """Recursively iterate through KML containers (Document/Folder) to Placemarks."""
    if hasattr(feature, "features"):
        for f in feature.features():
            yield from _iter_features(f)
    else:
        yield feature

def _kml_bytes_to_gdf(kml_bytes: bytes) -> gpd.GeoDataFrame:
    doc = kml.KML()
    # fastkml expects bytes in many versions; give it bytes directly
    doc.from_string(kml_bytes)

    records = []
    geoms = []

    for top in doc.features():
        for feat in _iter_features(top):
            # Most leaf features are Placemarks
            geom = getattr(feat, "geometry", None)
            if geom is None:
                continue

            name = getattr(feat, "name", None)
            desc = getattr(feat, "description", None)
            style_url = getattr(feat, "styleUrl", None)

            rec = {
                "name": name,
                "description": desc,
                "styleUrl": style_url,
            }

            # ExtendedData (if present)
            ext = getattr(feat, "extended_data", None)
            if ext and getattr(ext, "elements", None):
                for el in ext.elements:
                    # el.name / el.value in many fastkml versions
                    k = getattr(el, "name", None)
                    v = getattr(el, "value", None)
                    if k:
                        rec[str(k)] = v

            records.append(rec)
            geoms.append(geom)

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:4326")
    return gdf

def write_uploaded_kmz_to_temp_kml(uploaded_file):
    """KMZ UploadedFile -> extract first KML -> write to temp .kml path"""
    import tempfile
    import zipfile

    kmz_bytes = uploaded_file.getvalue()  # ✅ DO NOT use .read()
    with zipfile.ZipFile(BytesIO(kmz_bytes), "r") as z:
        kml_files = [n for n in z.namelist() if n.lower().endswith(".kml")]
        if not kml_files:
            return None
        kml_name = next((n for n in kml_files if n.lower().endswith("doc.kml")), kml_files[0])

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".kml")
        with tmp as f:
            f.write(z.read(kml_name))
        return tmp.name


# --- Load and add GeoJSON layers (safe + validated) ---


def safe_read_geojson(uploaded_file):
    """Safely read a GeoJSON/KML/KMZ upload and repair invalid geometries if needed.

    Streamlit Cloud often cannot read KML via GDAL/Fiona drivers, so:
    - .kmz/.kml are parsed with fastkml -> GeoDataFrame (driver-free)
    - .geojson/.json uses geopandas.read_file as usual
    """
    import geopandas as gpd
    import zipfile
    from fastkml import kml as fastkml_kml
    from shapely.errors import GEOSException
    from shapely.geometry.base import BaseGeometry
    from shapely.ops import transform as shapely_transform

    def _extract_kml_bytes_from_kmz(uploaded_kmz) -> bytes:
        kmz_bytes = uploaded_kmz.getvalue()
        with zipfile.ZipFile(BytesIO(kmz_bytes), "r") as z:
            kml_files = [n for n in z.namelist() if n.lower().endswith(".kml")]
            if not kml_files:
                raise ValueError("No .kml found inside KMZ.")
            # Prefer doc.kml if present
            kml_name = next((n for n in kml_files if n.lower().endswith("doc.kml")), kml_files[0])
            return z.read(kml_name)


def _walk_fastkml_features(feat):
    """
    fastkml compatibility:
    - some versions use feat.features() (callable)
    - some versions store children in feat.features (list)
    """
    children = None

    if hasattr(feat, "features"):
        children = getattr(feat, "features")

        # If it's a function -> call it
        if callable(children):
            children = children()

    if children:
        for sub in children:
            yield from _walk_fastkml_features(sub)
    else:
        yield feat

    def _kml_bytes_to_gdf(kml_bytes: bytes) -> gpd.GeoDataFrame:
        doc = fastkml_kml.KML()

        # fastkml expects bytes in many cases; but tolerate str too.
        try:
            doc.from_string(kml_bytes)
        except Exception:
            try:
                txt = kml_bytes.decode("utf-8", errors="replace")
                doc.from_string(txt)
            except Exception:
                # last resort
                doc.from_string(kml_bytes.decode("utf-8", errors="ignore"))

        records, geoms = [], []

        for top in doc.features():
            for f in _walk_fastkml_features(top):
                geom = getattr(f, "geometry", None)
                if geom is None:
                    continue
                records.append(
                    {
                        "name": getattr(f, "name", None),
                        "description": getattr(f, "description", None),
                        "styleUrl": getattr(f, "styleUrl", None),
                    }
                )
                geoms.append(geom)

        gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:4326")
        return gdf

    def _drop_z(geom: BaseGeometry):
        if geom is None:
            return None
        if not hasattr(geom, "has_z") or not geom.has_z:
            return geom
        try:
            return shapely_transform(lambda x, y, z=None: (x, y), geom)
        except Exception:
            return geom

    # ----------------------------
    # Read file
    # ----------------------------
    try:
        name = (uploaded_file.name or "").lower()
    
        if name.endswith(".kmz"):
            kml_bytes = _extract_kml_bytes_from_kmz(uploaded_file)
            gdf = _kml_bytes_to_gdf(kml_bytes)
    
            # Attach styles from a temp KML path
            kml_path = None
            try:
                kml_path = write_uploaded_kmz_to_temp_kml(uploaded_file)
                if kml_path:
                    gdf = attach_kml_styles_to_gdf(gdf, kml_path)
            except Exception as e:
                st.warning(f"KMZ styles could not be fully extracted: {e}")
            finally:
                if kml_path:
                    try:
                        import os
                        os.remove(kml_path)
                    except Exception:
                        pass
    
        elif name.endswith(".kml"):
            kml_bytes = uploaded_file.getvalue()
            gdf = _kml_bytes_to_gdf(kml_bytes)
    
            # Attach styles from a temp KML path
            kml_path = None
            try:
                import tempfile
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".kml")
                kml_path = tmp.name
                tmp.close()
                with open(kml_path, "wb") as f:
                    f.write(kml_bytes)
    
                gdf = attach_kml_styles_to_gdf(gdf, kml_path)
            except Exception as e:
                st.warning(f"KML styles could not be fully extracted: {e}")
            finally:
                if kml_path:
                    try:
                        import os
                        os.remove(kml_path)
                    except Exception:
                        pass
    
        else:
            # GeoJSON / JSON and other formats supported by geopandas
            gdf = gpd.read_file(uploaded_file)
    
    except GEOSException as e:
        st.warning(
            f"{uploaded_file.name} contains invalid geometries ({e}). Attempting to read attributes only..."
        )
        try:
            gdf = gpd.read_file(uploaded_file, ignore_geometry=True)
            if "geometry" in gdf.columns:
                gdf["geometry"] = None
        except Exception as inner_e:
            st.error(f"Could not load {uploaded_file.name}: {inner_e}")
            return None
    
    except Exception as e:
        st.error(f"Failed to read {uploaded_file.name}: {e}")
        return None


    # ----------------------------
    # Geometry cleanup / repair
    # ----------------------------
    if gdf is not None and "geometry" in gdf.columns:
        try:
            gdf["geometry"] = gdf["geometry"].apply(_drop_z)
            gdf = gdf[gdf.geometry.notnull()]
            gdf = gdf[~gdf.geometry.is_empty]

            allowed = {
                "Point",
                "MultiPoint",
                "LineString",
                "MultiLineString",
                "Polygon",
                "MultiPolygon",
            }
            gdf = gdf[gdf.geometry.geom_type.isin(allowed)]

            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326, allow_override=True)
            if gdf.crs and gdf.crs.to_string().upper() not in ("EPSG:4326", "WGS84"):
                try:
                    gdf = gdf.to_crs(epsg=4326)
                except Exception:
                    st.warning(
                        f"{uploaded_file.name}: could not reproject CRS {gdf.crs}; continuing as-is."
                    )

            invalid_count = (~gdf.is_valid).sum()
            if invalid_count > 0:
                st.info(f"Repairing {invalid_count} invalid geometries in {uploaded_file.name}...")
                gdf["geometry"] = gdf["geometry"].buffer(0)

            gdf = gdf[gdf.is_valid].reset_index(drop=True)

        except Exception as e:
            st.warning(f"Could not fully repair geometries in {uploaded_file.name}: {e}")

    return gdf


def preprocess_geojson_layers(files):
    layers = []
    if not files:
        return layers
    cache = st.session_state.get("geojson_cache", {})

    st.sidebar.header("Loaded GeoJSON layers")
    for geo in files:
        file_id = f"{geo.name}_{getattr(geo, 'size', 0)}"
        if file_id in cache:
            cached_layer = cache[file_id]
            if cached_layer.get("is_kmz") and "kmz_color" not in cached_layer:
                cached_layer["kmz_color"] = get_kmz_color_for_file(file_id)
            layers.append(cached_layer)
            st.sidebar.success(f"Added {geo.name} (cached)")
            continue

        gdf_zone = safe_read_geojson(geo)
        if gdf_zone is None or gdf_zone.empty:
            st.sidebar.error(f"Skipped {geo.name} (no valid geometries).")
            continue

        gdf_zone = optimize_gdf(gdf_zone, file_name=geo.name)
        gdf_zone = make_json_serialisable_gdf(gdf_zone)
        if "feature_id" not in gdf_zone.columns:
            gdf_zone["feature_id"] = gdf_zone.index.astype(str)

        kmz_color = get_kmz_color_for_file(file_id) if geo.name.lower().endswith((".kmz", ".kml")) else None
        if kmz_color:
            gdf_zone["stroke"] = kmz_color
            gdf_zone["color"] = kmz_color
            gdf_zone["fill"] = kmz_color
            gdf_zone["marker-color"] = kmz_color
        layer_record = {
            "name": geo.name,
            "gdf": gdf_zone,
            "labels": [],
            "is_kmz": geo.name.lower().endswith((".kmz", ".kml")),
            "kmz_color": kmz_color,
        }
        geom_types = list(gdf_zone.geometry.geom_type.unique())
        st.sidebar.caption(f"{geo.name}: geom types {geom_types}")

        try:
            if len(gdf_zone) <= 50 and set(gdf_zone.geometry.geom_type) <= {"Point", "MultiPoint"}:
                label_field = None
                for cand in ["name", "Name", "label", "Label"]:
                    if cand in gdf_zone.columns:
                        label_field = cand
                        break
                if label_field:
                    for _, r in gdf_zone.iterrows():
                        geom = r.geometry
                        if geom is None:
                            continue
                        if geom.geom_type == "Point":
                            layer_record["labels"].append(
                                {"lat": geom.y, "lon": geom.x, "label": str(r[label_field])}
                            )
            st.sidebar.success(f"Added {geo.name} ({len(gdf_zone)} features)")
        except Exception as e:
            st.sidebar.warning(f"Could not generate labels for {geo.name}: {e}")

        cache[file_id] = layer_record
        layers.append(layer_record)

    st.session_state["geojson_cache"] = cache

    return layers


# --- Excel helpers ---
def prepare_map_dataframe(map_idx: int, base_df: pd.DataFrame):
    df = base_df.copy()
    lat_col = st.session_state.get("shared_lat_col", base_df.columns[0])
    lon_col = st.session_state.get("shared_lon_col", base_df.columns[1] if len(base_df.columns) > 1 else base_df.columns[0])
    label_col = st.session_state.get(map_key("label_col", map_idx), base_df.columns[0])

    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col])

    filter_col = st.session_state.get(map_key("filter_col", map_idx), "(no filter)")
    if filter_col != "(no filter)" and filter_col in df.columns:
        col_data = df[filter_col].dropna()
        if pd.api.types.is_numeric_dtype(col_data):
            range_key = map_key("filter_range", map_idx)
            min_val, max_val = float(col_data.min()), float(col_data.max())
            selected_range = st.session_state.get(range_key, (min_val, max_val))
            df = df[(df[filter_col] >= selected_range[0]) & (df[filter_col] <= selected_range[1])]
        else:
            values_key = map_key("filter_values", map_idx)
            unique_vals = sorted(col_data.astype(str).unique())
            selected_vals = st.session_state.get(values_key, unique_vals)
            df = df[df[filter_col].astype(str).isin(selected_vals)]

    return df, lat_col, lon_col, label_col


def add_excel_points(map_obj, df, map_idx: int, lat_col: str, lon_col: str, label_col: str):
    import folium

    color_mode = st.session_state.get(map_key("color_mode", map_idx), "Single color")
    marker_color = st.session_state.get(map_key("marker_color", map_idx), "#3388ff")
    marker_shape = st.session_state.get(map_key("marker_shape", map_idx), "Circle")
    marker_radius = st.session_state.get(map_key("marker_radius", map_idx), 6)
    marker_opacity = st.session_state.get(map_key("marker_opacity", map_idx), 0.8)
    outline_color = st.session_state.get(map_key("outline_color", map_idx), "#000000")
    category_col = st.session_state.get(map_key("category_col", map_idx))
    category_colors = st.session_state.get(map_key("category_colors", map_idx), {})
    category_icons = st.session_state.get(map_key("category_icons", map_idx), {})
    custom_icon_bytes = st.session_state.get(map_key("custom_icon_bytes", map_idx))
    custom_icon_size = st.session_state.get(map_key("custom_icon_size", map_idx), 24)
    custom_icon_dims = st.session_state.get(map_key("custom_icon_dims", map_idx))

    if color_mode == "Single color":
        df["_color"] = marker_color
    else:
        if category_col and category_col in df.columns:
            categories = sorted(df[category_col].dropna().astype(str).unique())
            default_palette = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf"
            ]
            # Keep only colors for categories still present after filtering
            category_colors = {k: v for k, v in category_colors.items() if k in categories}
            for i, cat in enumerate(categories):
                if cat not in category_colors:
                    category_colors[cat] = default_palette[i % len(default_palette)]
            st.session_state[map_key("category_colors", map_idx)] = category_colors
            df["_color"] = df[category_col].astype(str).map(category_colors)
        else:
            df["_color"] = marker_color

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")

    fg = folium.FeatureGroup(name="Excel Points")
    legend_entries = []
    if color_mode == "Color by category" and category_col and category_colors:
        legend_entries = [
            {
                "label": cat,
                "color": col,
                "geom_type": "Point",
                "title": category_col,
                "marker_shape": marker_shape,
                "stroke": outline_color,
                "stroke_width": 1,
                "fill": col,
            }
            for cat, col in category_colors.items()
        ]
        for entry in legend_entries:
            cat = entry["label"]
            icon_info = category_icons.get(cat, {})
            if icon_info.get("bytes"):
                entry["custom_icon_data"] = "data:image/png;base64," + base64.b64encode(icon_info["bytes"]).decode("ascii")
                entry["custom_icon_size"] = int(icon_info.get("size", 24))
                entry["icon_opacity"] = float(icon_info.get("opacity", 1.0))
    else:
        legend_entries = [{
            "label": "All points",
            "color": marker_color,
            "geom_type": "Point",
            "title": "Excel Points",
            "marker_shape": marker_shape,
            "stroke": outline_color,
            "stroke_width": 1,
            "fill": marker_color,
        }]
        if marker_shape == "Custom Image" and custom_icon_bytes:
            legend_entries[0]["custom_icon_data"] = "data:image/png;base64," + base64.b64encode(custom_icon_bytes).decode("ascii")

    for _, row in gdf.iterrows():
        lat, lon = row[lat_col], row[lon_col]
        label = str(row[label_col]) if label_col else ""
        color_val = row["_color"] if "_color" in row else marker_color
        cat_val = str(row[category_col]) if category_col and category_col in row else None
        icon_info = category_icons.get(cat_val, {}) if cat_val else {}

        if marker_shape == "Circle":
            folium.CircleMarker(
                location=[lat, lon],
                radius=marker_radius,
                color=outline_color,
                weight=1,
                fill=True,
                fill_color=color_val,
                fill_opacity=marker_opacity,
                popup=label,
            ).add_to(fg)
        elif marker_shape == "Marker":
            folium.Marker(location=[lat, lon], popup=label, icon=folium.Icon(color="blue")).add_to(fg)
        elif marker_shape == "Square":
            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(
                    html=f'<div style="width:{marker_radius*2}px;height:{marker_radius*2}px;'
                    f'background-color:{color_val};border:2px solid {outline_color};'
                    f'margin-left:-{marker_radius}px;margin-top:-{marker_radius}px;"></div>'
                ),
                popup=label,
            ).add_to(fg)
        elif marker_shape == "Triangle":
            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(
                    html=(
                        f'<svg width="{marker_radius*3}" height="{marker_radius*3}">'
                        f'<polygon points="{marker_radius},{0} {marker_radius*2},{marker_radius*3} 0,{marker_radius*3}" '
                        f'style="fill:{color_val};stroke:{outline_color};stroke-width:1"/>'
                        "</svg>"
                    )
                ),
                popup=label,
            ).add_to(fg)
        elif marker_shape == "Custom Image":
            icon_bytes = icon_info.get("bytes") or custom_icon_bytes
            icon_dims = icon_info.get("dims") or custom_icon_dims
            icon_size = icon_info.get("size", custom_icon_size)
            icon_opacity = float(icon_info.get("opacity", marker_opacity))
            if icon_bytes:
                b64_icon = base64.b64encode(icon_bytes).decode("ascii")
                data_url = f"data:image/png;base64,{b64_icon}"
                icon_px = int(icon_size)
                icon_w, icon_h = get_scaled_icon_size(icon_bytes, icon_px, icon_dims)
                folium.Marker(
                    location=[lat, lon],
                    popup=label,
                    icon=folium.DivIcon(
                        html=(
                            f"<img src='{data_url}' style='width:{icon_w}px;height:{icon_h}px;"
                            f"object-fit:contain;opacity:{icon_opacity};'>"
                        ),
                        icon_size=(icon_w, icon_h),
                        icon_anchor=(icon_w // 2, icon_h // 2),
                    ),
                ).add_to(fg)

    fg.add_to(map_obj)
    return legend_entries


def add_labels_from_file(map_obj, df, map_idx: int, lat_col: str, lon_col: str, label_col: str):
    import folium

    if label_col not in df.columns:
        return

    offsets = st.session_state.get(map_key("label_offsets", map_idx), {})
    deg_per_m = 1 / 111_000  # approx conversion at mid-latitudes

    for _, r in df.iterrows():
        label_text = str(r[label_col]).strip()
        if not label_text:
            continue

        coord_key = f"{round(r[lat_col], 6)}_{round(r[lon_col], 6)}"
        # Prefer coordinate-based offset so it survives label column changes; fall back to legacy label-based key
        offset_km = float(offsets.get(coord_key, offsets.get(label_text, 0)))
        offset_deg = (offset_km * 1000) * deg_per_m

        label_lat = r[lat_col] + offset_deg
        label_lon = r[lon_col]

        folium.PolyLine(
            locations=[[r[lat_col], r[lon_col]], [label_lat, label_lon]],
            color="#444444",
            weight=1,
            opacity=0.7,
        ).add_to(map_obj)

        text_len = len(label_text)
        box_min = max(60, text_len * 6)
        box_max = min(250, text_len * 8)

        label_html = (
            f'<div style="'
            f'font-size:11px; font-weight:600; '
            f'background:rgba(255,255,255,0.9); '
            f'border:1px solid #555; border-radius:4px; '
            f'padding:3px 6px; '
            f'min-width:{box_min}px; max-width:{box_max}px; '
            f'text-align:center; white-space:normal; word-break:break-word; '
            f'box-shadow:0 1px 2px rgba(0,0,0,0.25);">'
            f'{label_text}</div>'
        )

        folium.Marker(
            location=[label_lat, label_lon],
            icon=folium.DivIcon(
                html=label_html,
                icon_size=(box_max, 18),
                icon_anchor=(box_max / 2, 0),
            ),
        ).add_to(map_obj)


def maybe_add_heatmap(map_obj, points, map_idx: int):
    toggle = st.session_state.get(map_key("heatmap_toggle", map_idx), False)
    if not toggle or not points:
        return

    heatmap_data = [p for p in points if p and len(p) == 2 and p[0] is not None and p[1] is not None]
    if len(heatmap_data) < 2:
        return

    try:
        map_obj.add_heatmap(
            data=heatmap_data,
            radius=st.session_state.get(map_key("heatmap_radius", map_idx), 20),
            blur=st.session_state.get(map_key("heatmap_blur", map_idx), 15),
            min_opacity=st.session_state.get(map_key("heatmap_opacity", map_idx), 0.4),
            name="Heatmap",
        )
    except Exception as e:
        st.sidebar.error(f"Heatmap could not be added: {e}")


# --- Sidebar controls per map (expander) ---
def render_map_controls(map_idx: int, df: pd.DataFrame, geo_layers):
    has_df = df is not None
    if has_df:
        ensure_map_defaults(map_idx, df)
    else:
        st.session_state.setdefault(map_key("name", map_idx), f"Map {map_idx + 1}")
        st.session_state.setdefault(map_key("labels_on", map_idx), True)
        st.session_state.setdefault(map_key("marker_shape", map_idx), "Circle")
        st.session_state.setdefault(map_key("marker_radius", map_idx), 6)
        st.session_state.setdefault(map_key("marker_opacity", map_idx), 0.8)
        st.session_state.setdefault(map_key("outline_color", map_idx), "#000000")
        st.session_state.setdefault(map_key("map_height", map_idx), calc_map_height(st.session_state.map_count))
        st.session_state.setdefault(map_key("heatmap_toggle", map_idx), False)
        st.session_state.setdefault(map_key("heatmap_radius", map_idx), 20)
        st.session_state.setdefault(map_key("heatmap_blur", map_idx), 15)
        st.session_state.setdefault(map_key("heatmap_opacity", map_idx), 0.4)
    ensure_layer_settings(map_idx, geo_layers)
    with st.sidebar.expander(f"Map {map_idx + 1} controls", expanded=(map_idx == 0)):
        st.text(f"Name: {get_map_name(map_idx)}")
        st.toggle("Show labels on this map", value=st.session_state.get(map_key("labels_on", map_idx), True), key=map_key("labels_on", map_idx))

        render_legend_settings_controls(map_idx)

        if has_df:
            st.selectbox("Label column", options=df.columns, key=map_key("label_col", map_idx))

            st.subheader("Filters")
            filter_options = ["(no filter)"] + [str(c) for c in df.columns]
            st.selectbox(
                "Select a column to filter",
                options=filter_options,
                index=filter_options.index(st.session_state.get(map_key("filter_col", map_idx), "(no filter)")),
                key=map_key("filter_col", map_idx),
            )

            filter_col = st.session_state.get(map_key("filter_col", map_idx), "(no filter)")
            if filter_col != "(no filter)":
                col_data = df[filter_col].dropna()
                if pd.api.types.is_numeric_dtype(col_data):
                    min_val, max_val = float(col_data.min()), float(col_data.max())
                    st.slider(
                        f"Filter {filter_col} range",
                        min_value=min_val,
                        max_value=max_val,
                        value=st.session_state.get(map_key("filter_range", map_idx), (min_val, max_val)),
                        key=map_key("filter_range", map_idx),
                    )
                else:
                    unique_vals = sorted(col_data.astype(str).unique())
                    st.multiselect(
                        f"Filter {filter_col} values",
                        options=unique_vals,
                        default=st.session_state.get(map_key("filter_values", map_idx), unique_vals),
                        key=map_key("filter_values", map_idx),
                    )

            # Apply current filters so downstream controls see filtered categories
            filtered_df, lat_col, lon_col, label_col = prepare_map_dataframe(map_idx, df)

            st.subheader("Marker & Category Colors")
            st.radio(
                "Color mode:",
                ["Single color", "Color by category"],
                index=0 if st.session_state.get(map_key("color_mode", map_idx), "Single color") == "Single color" else 1,
                key=map_key("color_mode", map_idx),
                horizontal=True,
            )

            if st.session_state[map_key("color_mode", map_idx)] == "Single color":
                st.color_picker(
                    "Marker color",
                    st.session_state.get(map_key("marker_color", map_idx), "#3388ff"),
                    key=map_key("marker_color", map_idx),
                )
            else:
                category_col = st.selectbox(
                    "Select category column",
                    options=[None] + list(df.columns),
                    index=0,
                    key=map_key("category_col", map_idx),
                )
                if category_col:
                    categories = sorted(filtered_df[category_col].dropna().astype(str).unique())
                    stored_colors = st.session_state.get(map_key("category_colors", map_idx), {})
                    stored_icons = st.session_state.get(map_key("category_icons", map_idx), {})
                    # keep only current cats
                    stored_colors = {k: v for k, v in stored_colors.items() if k in categories}
                    stored_icons = {k: v for k, v in stored_icons.items() if k in categories}

                    st.subheader("Marker style")
                    cat_marker_shape = st.selectbox(
                        "Marker shape",
                        options=["Circle", "Marker", "Square", "Triangle", "Custom Image"],
                        index=["Circle", "Marker", "Square", "Triangle", "Custom Image"].index(
                            st.session_state.get(map_key("marker_shape", map_idx), "Circle")
                        ),
                        key=map_key("marker_shape", map_idx),
                    )

                    if cat_marker_shape == "Custom Image":
                        st.markdown("Upload an icon per category")
                        for cat in categories:
                            icon_upload = st.file_uploader(
                                f"{cat} icon (PNG/JPG)",
                                type=["png", "jpg", "jpeg"],
                                key=map_key(f"cat_icon_{cat}", map_idx),
                            )
                            stored_icons.setdefault(cat, {"bytes": None, "dims": None, "size": 24, "opacity": 1.0})
                            if icon_upload:
                                raw = icon_upload.read()
                                stored_icons[cat]["bytes"] = raw
                                stored_icons[cat]["dims"] = get_image_dims(raw)
                            stored_icons[cat]["size"] = st.slider(
                                f"{cat} icon size (px)",
                                min_value=12,
                                max_value=250,
                                value=int(stored_icons[cat].get("size", 24)),
                                key=map_key(f"cat_icon_size_{cat}", map_idx),
                            )
                            stored_icons[cat]["opacity"] = st.slider(
                                f"{cat} icon opacity",
                                min_value=0.1,
                                max_value=1.0,
                                value=float(stored_icons[cat].get("opacity", 1.0)),
                                key=map_key(f"cat_icon_opacity_{cat}", map_idx),
                            )
                        st.session_state[map_key("category_icons", map_idx)] = stored_icons
                        # hide per-category color pickers when using images
                        st.session_state[map_key("category_colors", map_idx)] = stored_colors
                    else:
                        st.markdown("Assign colors to each category")
                        default_palette = [
                            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                            "#bcbd22", "#17becf"
                        ]
                        for i, cat in enumerate(categories):
                            picker_key = map_key(f"cat_color_{cat}", map_idx)
                            stored_colors[cat] = st.color_picker(
                                f"{cat}",
                                stored_colors.get(cat, default_palette[i % len(default_palette)]),
                                key=picker_key,
                            )
                        st.session_state[map_key("category_colors", map_idx)] = stored_colors
                        # clear icons when not using them
                        st.session_state[map_key("category_icons", map_idx)] = {}

                    if cat_marker_shape != "Custom Image":
                        st.slider("Marker radius (px)", 2, 20, st.session_state.get(map_key("marker_radius", map_idx), 6), key=map_key("marker_radius", map_idx))
                        st.slider("Marker opacity", 0.1, 1.0, st.session_state.get(map_key("marker_opacity", map_idx), 0.8), key=map_key("marker_opacity", map_idx))
                        st.color_picker("Outline color", st.session_state.get(map_key("outline_color", map_idx), "#000000"), key=map_key("outline_color", map_idx))
                else:
                    st.warning("Select a column to color by category.")

        if has_df and st.session_state.get(map_key("color_mode", map_idx), "Single color") == "Single color":
            st.subheader("Marker Style")
            st.selectbox(
                "Marker shape",
                options=["Circle", "Marker", "Square", "Triangle", "Custom Image"],
                key=map_key("marker_shape", map_idx),
            )
            st.slider("Marker radius (px)", 2, 20, st.session_state.get(map_key("marker_radius", map_idx), 6), key=map_key("marker_radius", map_idx))
            st.slider("Marker opacity", 0.1, 1.0, st.session_state.get(map_key("marker_opacity", map_idx), 0.8), key=map_key("marker_opacity", map_idx))
            st.color_picker("Outline color", st.session_state.get(map_key("outline_color", map_idx), "#000000"), key=map_key("outline_color", map_idx))

            if st.session_state[map_key("marker_shape", map_idx)] == "Custom Image":
                uploaded_icon = st.file_uploader(
                    "Upload PNG/JPG icon",
                    type=["png", "jpg", "jpeg"],
                    key=map_key("custom_icon_upload", map_idx),
                )
                if uploaded_icon:
                    raw = uploaded_icon.read()
                    st.session_state[map_key("custom_icon_bytes", map_idx)] = raw
                    st.session_state[map_key("custom_icon_dims", map_idx)] = get_image_dims(raw)
                st.slider(
                    "Custom icon size (px)",
                    min_value=50,
                    max_value=150,
                    value=st.session_state.get(map_key("custom_icon_size", map_idx), 24),
                    key=map_key("custom_icon_size", map_idx),
                )

        st.slider(
            "Map height (px)",
            min_value=300,
            max_value=1200,
            value=st.session_state.get(map_key("map_height", map_idx), calc_map_height(st.session_state.map_count)),
            key=map_key("map_height", map_idx),
        )

        if has_df:
            st.subheader("Label Placement")
            label_choices = []
            if label_col in filtered_df.columns:
                for _, row in filtered_df.iterrows():
                    label_text = str(row[label_col])
                    coord_key = f"{round(row[lat_col], 6)}_{round(row[lon_col], 6)}"
                    display = f"{label_text} ({coord_key})"
                    label_choices.append((display, coord_key))

            if label_choices:
                display_options = [c[0] for c in label_choices][:200]
                coord_lookup = {c[0]: c[1] for c in label_choices[:200]}
                selected_display = st.selectbox(
                    "Choose a point to adjust",
                    options=display_options,
                    key=map_key("label_choice", map_idx),
                )
                selected_coord = coord_lookup[selected_display]
                slider_key = map_key(f"label_offset_slider_{selected_coord}", map_idx)
                offsets = st.session_state.get(map_key("label_offsets", map_idx), {})
                offset_val = st.slider(
                    "Vertical offset (km; negative = below)",
                    min_value=-100.0,
                    max_value=100.0,
                    step=0.5,
                    value=offsets.get(selected_coord, 0.0),
                    key=slider_key,
                )
                offsets[selected_coord] = offset_val
                st.session_state[map_key("label_offsets", map_idx)] = offsets
            else:
                st.info("No labels available to adjust for this map.")

        st.subheader("🔥Heatmap Options")
        heatmap_sources = []
        if has_df:
            filtered_df, lat_col, lon_col, label_col = prepare_map_dataframe(map_idx, df)
            if not filtered_df.empty and pd.api.types.is_numeric_dtype(filtered_df[lat_col]) and pd.api.types.is_numeric_dtype(filtered_df[lon_col]):
                heatmap_sources.append(("Excel", "Excel data"))
        # add geo layers with points
        for layer in geo_layers:
            gdf = layer.get("gdf")
            if gdf is None or gdf.empty:
                continue
            if set(gdf.geometry.geom_type.unique()) & {"Point", "MultiPoint"}:
                heatmap_sources.append((layer["name"], f"Geo: {layer['name']}"))

        if heatmap_sources:
            st.toggle("Enable Heatmap", value=st.session_state.get(map_key("heatmap_toggle", map_idx), False), key=map_key("heatmap_toggle", map_idx))
            source_labels = [label for _, label in heatmap_sources]
            source_keys = [key for key, _ in heatmap_sources]
            current_src = st.session_state.get(map_key("heatmap_source", map_idx), source_keys[0])
            st.session_state[map_key("heatmap_source", map_idx)] = st.selectbox(
                "Heatmap data source",
                options=source_keys,
                format_func=lambda k: dict(heatmap_sources).get(k, k),
                index=source_keys.index(current_src) if current_src in source_keys else 0,
                key=map_key("heatmap_source_select", map_idx),
            )
            st.slider("Heatmap radius", 5, 50, st.session_state.get(map_key("heatmap_radius", map_idx), 20), key=map_key("heatmap_radius", map_idx))
            st.slider("Heatmap blur", 5, 40, st.session_state.get(map_key("heatmap_blur", map_idx), 15), key=map_key("heatmap_blur", map_idx))
            st.slider("Minimum opacity", 0.1, 1.0, st.session_state.get(map_key("heatmap_opacity", map_idx), 0.4), key=map_key("heatmap_opacity", map_idx))
        else:
            st.info("No suitable point data available for heatmap.")

        if not has_df:
            st.info("Upload an Excel/CSV to enable data filters and category colors.")

        # --- KMZ / KML per-file controls ---
        kmz_layers = [l for l in geo_layers if l.get("is_kmz")]
        if kmz_layers:
            st.subheader("KMZ / KML Layers")
            kmz_names = [l["name"] for l in kmz_layers]
            selected_kmz = st.selectbox("Select KMZ/KML file", kmz_names, key=map_key("selected_kmz_layer", map_idx))
            layer_settings = st.session_state[map_key("layer_settings", map_idx)][selected_kmz]
            # Colors / style
            layer_settings["color"] = st.color_picker("Layer color", layer_settings.get("color", "#3388ff"), key=map_key(f"kmz_color_{selected_kmz}", map_idx))
            # Point marker options (if points exist)
            layer_geom_types = set(next(l for l in kmz_layers if l["name"] == selected_kmz)["gdf"].geometry.geom_type.unique())
            has_points = bool(layer_geom_types & {"Point", "MultiPoint"})
            has_polys = bool(layer_geom_types & {"Polygon", "MultiPolygon"})
            has_lines = bool(layer_geom_types & {"LineString", "MultiLineString"})

            if has_lines:
                layer_settings["weight"] = st.slider("Line weight", 1, 10, int(layer_settings.get("weight", 2)), key=map_key(f"kmz_weight_{selected_kmz}", map_idx))
            if has_points or has_polys:
                layer_settings["outline_color"] = st.color_picker("Outline color (shapes)", layer_settings.get("outline_color", "#000000"), key=map_key(f"kmz_outline_{selected_kmz}", map_idx))
            if has_polys:
                layer_settings["fill_opacity"] = st.slider("Fill opacity", 0.0, 1.0, float(layer_settings.get("fill_opacity", 0.6)), key=map_key(f"kmz_fill_{selected_kmz}", map_idx))

            if layer_geom_types & {"Point", "MultiPoint"}:
                layer_settings["marker_shape"] = st.selectbox(
                    "Point marker shape",
                    options=["Circle", "Marker", "Square", "Triangle", "Custom Image"],
                    index=["Circle", "Marker", "Square", "Triangle", "Custom Image"].index(layer_settings.get("marker_shape", "Circle")),
                    key=map_key(f"kmz_marker_shape_{selected_kmz}", map_idx),
                )
                layer_settings["marker_radius"] = st.slider("Point marker radius (px)", 2, 20, int(layer_settings.get("marker_radius", 6)), key=map_key(f"kmz_marker_radius_{selected_kmz}", map_idx))
                layer_settings["marker_opacity"] = st.slider("Point marker opacity", 0.1, 1.0, float(layer_settings.get("marker_opacity", 0.9)), key=map_key(f"kmz_marker_opacity_{selected_kmz}", map_idx))
                if layer_settings["marker_shape"] == "Custom Image":
                    uploaded_icon = st.file_uploader("Custom point icon (PNG/JPG)", type=["png", "jpg", "jpeg"], key=map_key(f"kmz_custom_icon_{selected_kmz}", map_idx))
                    if uploaded_icon:
                        raw = uploaded_icon.read()
                        layer_settings["custom_icon_bytes"] = raw
                        layer_settings["custom_icon_dims"] = get_image_dims(raw)
                    layer_settings["custom_icon_size"] = st.slider("Custom icon size (px)", 20, 150, int(layer_settings.get("custom_icon_size", 24)), key=map_key(f"kmz_custom_icon_size_{selected_kmz}", map_idx))

        # --- GeoJSON per-file controls ---
        geojson_layers_only = [l for l in geo_layers if not l.get("is_kmz")]
        if geojson_layers_only:
            st.subheader("GeoJSON Layers")
            gj_names = [l["name"] for l in geojson_layers_only]
            selected_gj = st.selectbox("Select GeoJSON file", gj_names, key=map_key("selected_geojson_layer", map_idx))
            layer_settings = st.session_state[map_key("layer_settings", map_idx)][selected_gj]
            layer_settings["color"] = st.color_picker("Layer color", layer_settings.get("color", "#3388ff"), key=map_key(f"gj_color_{selected_gj}", map_idx))
            layer_geom_types = set(next(l for l in geojson_layers_only if l["name"] == selected_gj)["gdf"].geometry.geom_type.unique())
            has_points = bool(layer_geom_types & {"Point", "MultiPoint"})
            has_polys = bool(layer_geom_types & {"Polygon", "MultiPolygon"})
            has_lines = bool(layer_geom_types & {"LineString", "MultiLineString"})

            if has_lines:
                layer_settings["weight"] = st.slider("Line weight", 1, 10, int(layer_settings.get("weight", 2)), key=map_key(f"gj_weight_{selected_gj}", map_idx))
            if has_points or has_polys:
                layer_settings["outline_color"] = st.color_picker("Outline color (shapes)", layer_settings.get("outline_color", "#000000"), key=map_key(f"gj_outline_{selected_gj}", map_idx))
            if has_polys:
                layer_settings["fill_opacity"] = st.slider("Fill opacity", 0.0, 1.0, float(layer_settings.get("fill_opacity", 0.6)), key=map_key(f"gj_fill_{selected_gj}", map_idx))
            if layer_geom_types & {"Point", "MultiPoint"}:
                layer_settings["marker_shape"] = st.selectbox(
                    "Point marker shape",
                    options=["Circle", "Marker", "Square", "Triangle", "Custom Image"],
                    index=["Circle", "Marker", "Square", "Triangle", "Custom Image"].index(layer_settings.get("marker_shape", "Circle")),
                    key=map_key(f"gj_marker_shape_{selected_gj}", map_idx),
                )
                layer_settings["marker_radius"] = st.slider("Point marker radius (px)", 2, 20, int(layer_settings.get("marker_radius", 6)), key=map_key(f"gj_marker_radius_{selected_gj}", map_idx))
                layer_settings["marker_opacity"] = st.slider("Point marker opacity", 0.1, 1.0, float(layer_settings.get("marker_opacity", 0.9)), key=map_key(f"gj_marker_opacity_{selected_gj}", map_idx))
                if layer_settings["marker_shape"] == "Custom Image":
                    uploaded_icon = st.file_uploader("Custom point icon (PNG/JPG)", type=["png", "jpg", "jpeg"], key=map_key(f"gj_custom_icon_{selected_gj}", map_idx))
                    if uploaded_icon:
                        raw = uploaded_icon.read()
                        layer_settings["custom_icon_bytes"] = raw
                        layer_settings["custom_icon_dims"] = get_image_dims(raw)
                    layer_settings["custom_icon_size"] = st.slider("Custom icon size (px)", 20, 150, int(layer_settings.get("custom_icon_size", 24)), key=map_key(f"gj_custom_icon_size_{selected_gj}", map_idx))
            st.markdown("Legend labels")
            for gt in ["Point", "LineString", "Polygon"]:
                if gt in layer_geom_types:
                    layer_settings["labels"][gt] = st.text_input(f"{gt} label", value=layer_settings["labels"].get(gt, gt), key=map_key(f"gj_label_{selected_gj}_{gt}", map_idx))

    return prepare_map_dataframe(map_idx, df)[0] if has_df else None

# --- KMZ Legend Editor (Streamlit UI) --- #
def editable_kmz_legend(kmz_entries, map_idx):
    """
    Legend settings + editable KMZ legend labels.
    One input per (file, geom_type), with stable keys.
    In export mode, no widgets are rendered but saved labels are applied.
    """

    if not kmz_entries:
        return kmz_entries

    key_base = map_key("kmz_legend_labels", map_idx)
    ensure_legend_settings(map_idx)
    legend_settings = st.session_state[map_key("legend_settings", map_idx)]

    # Initialize session storage: { (file, type): label }
    if key_base not in st.session_state:
        st.session_state[key_base] = {
            (e["source_file"], e["geom_type"]): e.get("label", e["geom_type"])
            for e in kmz_entries
        }

    label_map = st.session_state[key_base]

    # 🚫 EXPORT MODE — do NOT render widgets, but DO apply saved labels
    if st.session_state.get("export_mode_global", False):
        for e in kmz_entries:
            key = (e["source_file"], e["geom_type"])
            e["user_label"] = label_map.get(key, e.get("label"))
        return kmz_entries

    # NORMAL MODE — render KMZ label overrides (legend styling controls are global elsewhere)
    st.sidebar.subheader("KMZ Legend Labels")

    for e in kmz_entries:
        fname = e["source_file"]
        gtype = e["geom_type"]
        key = (fname, gtype)

        default_label = label_map.get(key, e.get("label"))

        widget_key = f"{key_base}_{fname}_{gtype}"

        new_label = st.sidebar.text_input(
            f"{fname} – {gtype} label",
            value=default_label,
            key=widget_key
        )

        label_map[key] = new_label
        e["user_label"] = new_label

    return kmz_entries





def render_single_map(map_idx: int, df: pd.DataFrame, geo_layers):
    zoom_level = st.session_state.get(map_key("map_zoom", map_idx), 6)
    center_lat = st.session_state.get(map_key("center_lat", map_idx), 54.5)
    center_lon = st.session_state.get(map_key("center_lon", map_idx), -2.0)
    m = leafmap.Map(center=[center_lat, center_lon], zoom=zoom_level, tiles=st.session_state.current_basemap)
    labels_on = st.session_state.get(map_key("labels_on", map_idx), True)
    ensure_layer_settings(map_idx, geo_layers)

    # Excel data
    category_entries = []
    if df is not None:
        filtered_df, lat_col, lon_col, label_col = prepare_map_dataframe(map_idx, df)
        category_entries = add_excel_points(m, filtered_df, map_idx, lat_col, lon_col, label_col)
        if labels_on:
            add_labels_from_file(m, filtered_df, map_idx, lat_col, lon_col, label_col)

    # Prepare heatmap points based on selected source
    heatmap_points = []
    selected_heatmap_source = st.session_state.get(map_key("heatmap_source", map_idx), "Excel")
    if selected_heatmap_source == "Excel" and df is not None:
        heatmap_points = filtered_df[[lat_col, lon_col]].dropna().values.tolist()

    # GeoJSON layers (shared source)
    kmz_entries = []
    geo_entries = []
    layer_points_cache = {}
    for layer in geo_layers:
        try:
            if "feature_id" not in layer["gdf"].columns:
                layer["gdf"]["feature_id"] = layer["gdf"].index.astype(str)

            settings = st.session_state[map_key("layer_settings", map_idx)].get(layer["name"], {})
            layer_color = settings.get("color") or layer.get("kmz_color") or "#3388ff"
            outline = settings.get("outline_color", "#000000")
            weight = settings.get("weight", 2)
            fill_opacity = settings.get("fill_opacity", 0.6)

            gdf_all = layer["gdf"]
            point_mask = gdf_all.geometry.geom_type.isin(["Point", "MultiPoint"])
            gdf_points = gdf_all[point_mask]
            gdf_non_points = gdf_all[~point_mask]

            # Lines/Polygons with per-layer color
            if not gdf_non_points.empty:
                style_kwargs = {
                    "color": layer_color,
                    "fillColor": layer_color,
                    "fillOpacity": fill_opacity,
                    "weight": weight,
                }
                m.add_gdf(gdf_non_points, layer_name=layer["name"], zoom_to_layer=False, style=style_kwargs)

            # Points with per-layer marker styling
            if not gdf_points.empty:
                layer_points_cache[layer["name"]] = []
                for _, row in gdf_points.iterrows():
                    geom = row.geometry
                    if geom is None:
                        continue
                    if geom.geom_type == "Point":
                        coords = [(geom.y, geom.x)]
                    elif geom.geom_type == "MultiPoint":
                        coords = [(pt.y, pt.x) for pt in geom.geoms]
                    else:
                        continue
                    for lat, lon in coords:
                        layer_points_cache[layer["name"]].append([lat, lon])
                        render_point_marker(m, lat, lon, {**settings, "color": layer_color, "outline_color": outline})

            # Floating labels
            if labels_on and layer["labels"]:
                add_floating_labels(
                    m,
                    layer["labels"],
                    label_field="label",
                    source=f"GeoJSON: {layer['name']}",
                    sidebar_prefix=f"{layer['name']}_labels",
                )

            # Extract KMZ style entries
            if layer.get("is_kmz"):
                entries = extract_kmz_styles(layer["gdf"], color_override=layer_color)
                for e in entries:
                    e["source_file"] = layer["name"]   # <-- Required for grouping + editing
                    # Apply label override if provided
                    e["label"] = settings.get("labels", {}).get(e["geom_type"], e.get("label"))
                    if e["geom_type"] == "Point":
                        e["marker_shape"] = settings.get("marker_shape", "Marker")
                        e["outline_color"] = settings.get("outline_color")
                        if settings.get("marker_shape") == "Custom Image" and settings.get("custom_icon_bytes"):
                            e["custom_icon_data"] = "data:image/png;base64," + base64.b64encode(settings["custom_icon_bytes"]).decode("ascii")
                kmz_entries.extend(entries)
            else:
                # GeoJSON legend entries, same structure for reuse
                entries = extract_kmz_styles(layer["gdf"], color_override=layer_color)
                for e in entries:
                    e["source_file"] = layer["name"]
                    e["label"] = settings.get("labels", {}).get(e["geom_type"], e.get("label"))
                    if e["geom_type"] == "Point":
                        e["marker_shape"] = settings.get("marker_shape", "Marker")
                        e["outline_color"] = settings.get("outline_color")
                        if settings.get("marker_shape") == "Custom Image" and settings.get("custom_icon_bytes"):
                            e["custom_icon_data"] = "data:image/png;base64," + base64.b64encode(settings["custom_icon_bytes"]).decode("ascii")
                geo_entries.extend(entries)
        except Exception as e:
            st.sidebar.error(f"Failed to add {layer['name']} to map: {e}")

    # Add heatmap if selected source is a geo layer
    if selected_heatmap_source != "Excel":
        heatmap_points = layer_points_cache.get(selected_heatmap_source, [])
    maybe_add_heatmap(m, heatmap_points, map_idx)

    # User edits labels once per type (KMZ only, legacy UI)
    kmz_entries = editable_kmz_legend(kmz_entries, map_idx)
    for e in kmz_entries:
        e["label"] = e.get("user_label", e.get("label"))

    combined_geo_entries = kmz_entries + geo_entries
    geo_types_present = sorted({e["geom_type"] for e in combined_geo_entries})

    if geo_types_present:
        st.session_state[map_key("kmz_types_present", map_idx)] = geo_types_present
        render_combined_legend(m, category_entries, combined_geo_entries, map_idx)
        st.caption(f"Geo legend entries: {len(combined_geo_entries)} types -> {', '.join(geo_types_present)}")
    else:
        st.session_state[map_key("kmz_types_present", map_idx)] = []
        render_combined_legend(m, category_entries, [], map_idx)

    add_layer_selector(m)
    apply_fine_zoom(m)
    return m


def apply_fine_zoom(map_obj, snap: float = 0.1, delta: float = 0.1):
    """Allow finer-grain zoom steps via Leaflet zoomSnap/zoomDelta."""
    import folium

    # Set options on the folium map object so they render into the leaflet config
    map_obj.options["zoomSnap"] = snap
    map_obj.options["zoomDelta"] = delta

    map_var = map_obj.get_name()
    script = f"""
    <script>
      (function() {{
        var m = {map_var};
        if (!m) return;
        m.options.zoomSnap = {snap};
        m.options.zoomDelta = {delta};
        if (m.zoomControl) {{ m.zoomControl.options.zoomDelta = {delta}; }}
        if (m.scrollWheelZoom) {{ m.scrollWheelZoom._delta = {delta}; }}
        m.on('load', function() {{
          this.options.zoomSnap = {snap};
          this.options.zoomDelta = {delta};
          if (this.zoomControl) {{ this.zoomControl.options.zoomDelta = {delta}; }}
          if (this.scrollWheelZoom) {{ this.scrollWheelZoom._delta = {delta}; }}
        }});
        // Force refresh current zoom to respect new snap
        m.setZoom(m.getZoom());
      }})();
    </script>
    """
    map_obj.get_root().html.add_child(folium.Element(script))


def render_kmz_type_legend(map_obj, types_present):
    """Render a minimal KMZ legend based on geometry types present."""
    if not types_present:
        return
    import folium

    def icon_html(label: str, color: str, shape: str):
        if shape == "point":
            mark = f"<div style='width:12px;height:12px;border-radius:50%;background:{color};'></div>"
        elif shape == "line":
            mark = f"<div style='width:18px;height:2px;background:{color};margin-top:6px;'></div>"
        else:  # polygon
            mark = f"<div style='width:14px;height:14px;background:{color};opacity:0.7;border:1px solid {color};'></div>"
        return (
            f"<div style='margin:4px 0; display:flex; align-items:center;'>"
            f"<span style='display:inline-block;width:20px;height:14px;margin-right:6px;'>{mark}</span>"
            f"<span>{label}</span>"
            f"</div>"
        )

    items_html = ""
    if "Point" in types_present:
        items_html += icon_html("Point", "#3388ff", "point")
    if "LineString" in types_present:
        items_html += icon_html("Line", "#ff7f0e", "line")
    if "Polygon" in types_present:
        items_html += icon_html("Polygon", "#2ca02c", "polygon")

    legend_html = (
        "<div style='position: fixed; bottom: 30px; right: 30px; z-index:9999; "
        "background-color: white; border:2px solid gray; border-radius:8px; "
        "padding:10px; font-size:12px;'>"
        "<b>KMZ Legend</b><br>"
        f"{items_html}"
        "</div>"
    )
    map_obj.get_root().html.add_child(folium.Element(legend_html))


def render_combined_legend(map_obj, category_entries, kmz_entries, map_idx):
    if not category_entries and not kmz_entries:
        return
    import folium
    legend_settings = st.session_state.get(map_key("legend_settings", map_idx), {})
    width = int(legend_settings.get("width", 240))
    max_h = int(legend_settings.get("max_height", 500))
    corner = legend_settings.get("corner", "bottom_right")
    offset_x = int(legend_settings.get("offset_x", 30))
    offset_y = int(legend_settings.get("offset_y", 30))
    title_font = legend_settings.get("title_font", "Segoe UI")
    text_font = legend_settings.get("text_font", "Segoe UI")
    title_size = int(legend_settings.get("title_size", 13))
    text_size = int(legend_settings.get("text_size", 12))
    logo_data = legend_settings.get("logo_data")
    text_offset = int(legend_settings.get("text_offset", 6))
    entry_spacing = int(legend_settings.get("entry_spacing", 4))
    icon_size_override = int(legend_settings.get("custom_icon_size", 18))
    icon_box = max(20, icon_size_override + 6)

    pos_style = ""
    if corner == "bottom_right":
        pos_style = f"bottom:{offset_y}px; right:{offset_x}px;"
    elif corner == "bottom_left":
        pos_style = f"bottom:{offset_y}px; left:{offset_x}px;"
    elif corner == "top_right":
        pos_style = f"top:{offset_y}px; right:{offset_x}px;"
    else:
        pos_style = f"top:{offset_y}px; left:{offset_x}px;"

    html = (
        f"<div style='position: fixed; {pos_style} z-index:9999; "
        f"background-color: white; border:2px solid gray; border-radius:8px; "
        f"padding:10px; width:{width}px; max-height:{max_h}px; overflow-y:auto; "
        f"font-family:{text_font}; font-size:{text_size}px; "
        f"box-shadow:0 2px 6px rgba(0,0,0,0.2);'>"
    )

    # Excel category part
    if category_entries:
        title = category_entries[0].get("title") or "Categories"
        html += f"<div style='font-weight:bold;font-size:{title_size}px;font-family:{title_font};'>{title}</div>"
        for entry in category_entries:
            if entry.get("custom_icon_data"):
                entry = {**entry, "custom_icon_size": icon_size_override}
            sym = kmz_symbol_html(entry) if entry.get("geom_type") == "Point" else (
                f"<span style='width:{icon_box}px;height:{icon_box}px;background:{entry['color']};"
                f"margin-right:6px;border-radius:3px;display:inline-flex;align-items:center;justify-content:center;'></span>"
            )
            html += (
                f"<div style='display:flex;align-items:center;margin:{entry_spacing}px 0;'>"
                f"<span style='width:{icon_box}px;height:{icon_box}px;margin-right:6px;display:flex;align-items:center;justify-content:center;'>{sym}</span>"
                f"<span style='margin-left:{text_offset}px;'>{entry['label']}</span></div>"
            )
        html += "<hr>"

    # Geo legend (KMZ/GeoJSON)
    if kmz_entries:
        html += f"<div style='font-weight:bold;font-size:{title_size}px;font-family:{title_font};'>Geo Layers</div>"
        for e in kmz_entries:
            if e.get("custom_icon_data"):
                e = {**e, "custom_icon_size": icon_size_override}
            sym = kmz_symbol_html(e)
            label = e.get("user_label") or e.get("label")
            html += (
                f"<div style='display:flex;align-items:center;margin:{entry_spacing}px 0;'>"
                f"<span style='width:{icon_box}px;height:{icon_box}px;margin-right:6px;display:flex;align-items:center;justify-content:center;'>{sym}</span>"
                f"<span style='margin-left:{text_offset}px;'>{label}</span></div>"
            )

    if logo_data:
        html += (
            f"<div style='margin-top:8px;text-align:center;'>"
            f"<img src='{logo_data}' style='max-width:100%;height:auto;'/>"
            f"</div>"
        )

    html += "</div>"

    map_obj.get_root().html.add_child(folium.Element(html))



def build_kmz_type_entries(gdf, max_entries: int = 1000):
    """Build legend entries for KMZ/KML by geometry type with colors."""
    entries = []
    if gdf is None or gdf.empty:
        return entries

    def base_type(gtype: str) -> str:
        if gtype in ("Point", "MultiPoint"):
            return "Point"
        if gtype in ("LineString", "MultiLineString"):
            return "LineString"
        if gtype in ("Polygon", "MultiPolygon"):
            return "Polygon"
        return gtype

    for _, row in gdf.head(max_entries).iterrows():
        geom = row.get("geometry")
        if geom is None:
            continue
        gtype = base_type(getattr(geom, "geom_type", "") or str(row.get("geom_type", "")))
        if gtype not in {"Point", "LineString", "Polygon"}:
            continue
        color = row.get("stroke") or row.get("color") or row.get("fill") or "#666"
        entries.append(
            {
                "geom_type": gtype,
                "color": color,
                "label": gtype,
            }
        )
    return entries


def add_layer_selector(map_obj):
    """Ensure a Leaflet layer control is present regardless of renderer."""
    import folium

    # folium/leafmap add_gdf creates an overlay layer; ensure control exists
    try:
        map_obj.add_layer_control()
        return
    except Exception:
        pass
    try:
        map_obj.add_layers_control()
        return
    except Exception:
        pass
    try:
        folium.LayerControl().add_to(map_obj)
    except Exception:
        pass

# --- Prepare data ---
ensure_base_state()
df = None
if excel_file:
    df = load_excel_cached(excel_file)
    st.sidebar.success(f"Loaded {len(df)} records")
    ensure_shared_coords(df)
    for idx in range(st.session_state.map_count):
        ensure_map_defaults(idx, df)

geojson_layers = preprocess_geojson_layers(geojson_files) if geojson_files else []

# --- Shared lat/lon selection (applies to all maps) ---
if df is not None:
    st.sidebar.markdown("### Shared coordinates (applies to all maps)")
    st.sidebar.selectbox("Latitude column", options=df.columns, key="shared_lat_col")
    st.sidebar.selectbox("Longitude column", options=df.columns, key="shared_lon_col")

# --- Sidebar controls per map (expanders) ---
filtered_dfs = {}
for idx in range(st.session_state.map_count):
    result_df = render_map_controls(idx, df, geojson_layers)
    if result_df is not None:
        filtered_dfs[idx] = result_df

# --- Map header with add/remove buttons ---
header_cols = st.columns([3, 1, 1, 2])
with header_cols[0]:
    st.subheader("Map View")
with header_cols[1]:
    add_disabled = st.session_state.map_count >= MAX_MAPS
    if st.button("Add map", use_container_width=True, disabled=add_disabled):
        st.session_state.map_count = min(MAX_MAPS, st.session_state.map_count + 1)
        st.rerun()
with header_cols[2]:
    remove_disabled = st.session_state.map_count <= 1
    if st.button("Remove last map", use_container_width=True, disabled=remove_disabled):
        st.session_state.map_count = max(1, st.session_state.map_count - 1)
        st.rerun()
with header_cols[3]:
    st.toggle("Export mode", value=st.session_state.get("export_mode_global", False), key="export_mode_global")

export_mode = st.session_state.get("export_mode_global", False)


def export_map_as_html(map_obj) -> bytes:
    return map_obj.get_root().render().encode("utf-8")


# --- Map grid ---
map_count = st.session_state.map_count
cols_per_row = cols_for_count(map_count)
default_height = calc_map_height(map_count)

map_idx = 0
rows = (map_count + cols_per_row - 1) // cols_per_row

for _ in range(rows):
    row_cols = st.columns(cols_per_row)
    for col in row_cols:
        if map_idx >= map_count:
            break

        with col:
            map_height = st.session_state.get(map_key("map_height", map_idx), default_height)
            m = render_single_map(map_idx, df, geojson_layers)

            # Render map
            if export_mode and HAS_ST_FOLIUM:
                map_state = st_folium(m, height=map_height, width="100%", key=f"map_render_{map_idx}")
                if map_state:
                    if map_state.get("zoom") is not None:
                        st.session_state[map_key("map_zoom", map_idx)] = map_state["zoom"]
                    if map_state.get("center"):
                        st.session_state[map_key("center_lat", map_idx)] = map_state["center"]["lat"]
                        st.session_state[map_key("center_lon", map_idx)] = map_state["center"]["lng"]
            else:
                m.to_streamlit(height=map_height)

            # Map name (NO empty label)
            st.text_input(
                "Map name",
                key=map_key("name", map_idx),
                value=get_map_name(map_idx),
                label_visibility="collapsed",
            )

            # HTML export (Cloud-safe)
            if export_mode:
                html_bytes = export_map_as_html(m)
                st.download_button(
                    f"Download HTML ({get_map_name(map_idx)})",
                    data=html_bytes,
                    file_name=f"{get_map_name(map_idx).replace(' ', '_').lower()}.html",
                    mime="text/html",
                    key=map_key("download_html", map_idx),
                )

        map_idx += 1


# --- Data preview for all maps ---
if filtered_dfs:
    st.subheader("Data Preview (per map)")
    for idx, fdf in filtered_dfs.items():
        st.markdown(f"**Map {idx + 1}**")
        st.dataframe(fdf.head())

st.markdown("---")
st.caption("Built with Streamlit + Leafmap + GeoPandas")
