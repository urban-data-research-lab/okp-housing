import streamlit as st
import geopandas as gpd
import pandas as pd
import pydeck as pdk
import plotly.express as px

# ----------------------
# LOAD / PREP DATA
# ----------------------
@st.cache_data
def load_data():
    resi_parcels_keep = gpd.read_file("data/resi_parcels_keep.geojson")  # or shapefile/parquet
    # make sure it's WGS84
    if resi_parcels_keep.crs is not None and resi_parcels_keep.crs.to_epsg() != 4326:
        resi_parcels_keep = resi_parcels_keep.to_crs(epsg=4326)

    resi_parcels_keep['Owner Type'] = resi_parcels_keep['Owner Type'].fillna("Unknown")
    resi_parcels_keep['Owner Type'] = resi_parcels_keep['Owner Type'].str.title()
    return resi_parcels_keep

resi_parcels_keep = load_data()

# ----------------------
# PAGE CONFIG (white background)
# ----------------------
st.set_page_config(page_title="Owner Type Map", layout="wide")

# Inline CSS override for white background
# st.markdown(
#     """
#     <style>
#         .stApp {
#             background-color: white;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

st.title("Residential Parcels by Owner Type")

# ----------------------
# BUILD A COLOR MAP FOR OWNER TYPE
# ----------------------
# pick some distinct-ish colors
COLOR_PALETTE = [
    [230, 25, 75],    # red
    [60, 180, 75],    # green
    [0, 130, 200],    # blue
    [245, 130, 48],   # orange
    [145, 30, 180],   # purple
    [70, 240, 240],   # cyan
    [240, 50, 230],   # magenta
    [210, 245, 60],   # lime
    [250, 190, 190],  # pink
    [0, 128, 128],    # teal
    [170, 110, 40],   # brown
    [128, 0, 0],      # maroon
]

# 1. collect categories

owner_col = "Owner Type"   # <- your column name

if owner_col in resi_parcels_keep.columns:
    cats = resi_parcels_keep[owner_col].fillna("Unknown").astype(str).unique().tolist()
else:
    cats = ["Unknown"]

# 2. make raw -> display mapping (title case)
label_map = {raw: raw.title() for raw in cats}

# 3. color map keyed on raw values
color_map = {raw: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, raw in enumerate(cats)}



# map each category to a color
color_map = {}
for i, cat in enumerate(cats):
    color_map[cat] = COLOR_PALETTE[i % len(COLOR_PALETTE)]

# add color columns to resi_parcels_keep
resi_parcels_keep[owner_col] = resi_parcels_keep[owner_col].fillna("Unknown").astype(str)
resi_parcels_keep["fill_color"] = resi_parcels_keep[owner_col].map(color_map)

# pydeck wants a flat list, so we split into components
resi_parcels_keep["fill_r"] = resi_parcels_keep["fill_color"].apply(lambda c: c[0])
resi_parcels_keep["fill_g"] = resi_parcels_keep["fill_color"].apply(lambda c: c[1])
resi_parcels_keep["fill_b"] = resi_parcels_keep["fill_color"].apply(lambda c: c[2])



# --------------------
# SIDEBAR CHECKBOXES
# --------------------


with st.sidebar:
    st.header("Owner Types")
    selected_cats = []
    for cat in cats:
        disp = label_map[cat]
        # pre-check everything by default
        checked = st.checkbox(cat, value=True)
        if checked:
            selected_cats.append(cat)

# filter by checkboxes
filtered = resi_parcels_keep[resi_parcels_keep["Owner Type"].isin(selected_cats)].copy()


# ----------------------
# Get center for view
# ----------------------
if len(filtered) > 0:
    # use centroid of all polygons to center the map
    center = filtered.to_crs(epsg=4326).geometry.unary_union.centroid
    mid_lat = center.y
    mid_lon = center.x
else:
    mid_lat, mid_lon = 34.4, -119.7  # fallback


# ----------------------
# Build GeoJsonLayer
# ----------------------
geojson = filtered.__geo_interface__

layer = pdk.Layer(
    "GeoJsonLayer",
    geojson,
    pickable=True,
    stroked=False,
    filled=True,
    extruded=False,
    get_fill_color="[properties.fill_r, properties.fill_g, properties.fill_b, 190]",
    get_line_color=[0, 0, 0],
    line_width_min_pixels=1,
)

view_state = pdk.ViewState(
    latitude=mid_lat,
    longitude=mid_lon,
    zoom=11,
    pitch=0,
)

st.subheader("Map")
# ----------------------
# LEGEND
# ----------------------
legend_html = "<div style='display:flex;flex-wrap:wrap;gap:8px;'>"
for raw in selected_cats:   # show only currently selected
    disp = label_map[raw]
    col = color_map[raw]
    legend_html += (
        f"<div style='display:flex;align-items:center;gap:4px;'>"
        f"<div style='width:14px;height:14px;background:rgba({col[0]}, {col[1]}, {col[2]}, 1);"
        f"border:1px solid #555;'></div>"
        f"<div style='font-size:12px;'>{disp}</div>"
        f"</div>"
    )
legend_html += "</div>"

# st.markdown("**Legend**", unsafe_allow_html=True)
st.markdown(legend_html, unsafe_allow_html=True)




# ✅ Positron basemap here
map_style = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style=map_style,
        tooltip={
            "text": "Parcel Number: {APN}\nOwner Type: {Owner Type}\nActual Owner Type: {Actual Owner Type}\nOwner Name: {Owner Name}\nNotes: {Notes}",
            "style": {
                "backgroundColor": "rgba(50, 50, 50, 0.9)",
                "color": "white",
                "maxWidth": "300px",          # limit tooltip width
                "whiteSpace": "normal",       # allow wrapping
                "wordBreak": "break-word",    # break long words if needed
                "fontSize": "12px",
            },
        },
    )
)

# # ----------------------
# # CHART
# # ----------------------
def wrap_label(text, width=20):
    """Insert a line break if label length exceeds `width`."""
    text = str(text)
    if len(text) > width:
        # break at the closest space before or after `width`
        parts = text.split()
        out = []
        cur = ""
        for word in parts:
            if len(cur + " " + word) <= width:
                cur = (cur + " " + word).strip()
            else:
                out.append(cur)
                cur = word
        out.append(cur)
        return "<br>".join(out)
    return text

st.subheader("Owner Type vs. Actual Owner Type")

col1, col2 = st.columns(2)

# --- LEFT CHART: Owner Type ---
with col1:
    # drop Unknown
    df_owner = (
        resi_parcels_keep[resi_parcels_keep["Owner Type"].notna()]
        .loc[resi_parcels_keep["Owner Type"] != "Unknown"]
        .copy()
    )

    total_owner = len(df_owner)

    counts_owner = (
        df_owner["Owner Type"]
        .astype(str)
        .value_counts()
        .rename_axis("Owner Type")
        .reset_index(name="count")
    )

    counts_owner["pct"] = counts_owner["count"] / total_owner * 100
    counts_owner = counts_owner.sort_values("pct", ascending=True)
    counts_owner["label"] = counts_owner.apply(
        lambda x: f"{x['count']:,} — {x['pct']:.1f}%", axis=1
    )

    st.markdown(f"**Owner Type (n = {total_owner:,})**")

    # after you build counts_owner ...
    max_owner_pct = counts_owner["pct"].max()

    counts_owner["wrapped"] = counts_owner["Owner Type"].apply(wrap_label)

    fig_owner = px.bar(
        counts_owner,
        x="pct",
        y="wrapped",  # your wrapped label from earlier
        orientation="h",
        labels={"pct": "Percent of parcels", "wrapped": "Owner Type"},
        text="label",
        color_discrete_sequence=["#1f77b4"],
    )

    fig_owner.update_layout(
        height=500,
        margin=dict(l=10, r=110, t=30, b=10),   # more right margin
        xaxis=dict(range=[0, max_owner_pct * 1.12]),  # a bit more headroom
    )
    fig_owner.update_traces(
        textposition="outside",
        cliponaxis=False,   # <-- key line
    )

    st.plotly_chart(fig_owner, use_container_width=True)

# --- RIGHT CHART: Actual Owner Type ---
with col2:
    # drop both actual NaNs and string "nan"
    df_actual = resi_parcels_keep[
        resi_parcels_keep["Actual Owner Type"].notna() &
        (resi_parcels_keep["Actual Owner Type"].astype(str).str.lower() != "nan")
    ].copy()

    total_actual = len(df_actual)

    counts_actual = (
        df_actual["Actual Owner Type"]
        .astype(str)
        .value_counts()
        .rename_axis("Actual Owner Type")
        .reset_index(name="count")
    )

    counts_actual["pct"] = counts_actual["count"] / total_actual * 100
    counts_actual = counts_actual.sort_values("pct", ascending=True)
    counts_actual["label"] = counts_actual.apply(
        lambda x: f"{x['count']:,} — {x['pct']:.1f}%", axis=1
    )

    st.markdown(f"**Actual Owner Type (n = {total_actual:,})**")

    max_actual_pct = counts_actual["pct"].max()

    counts_actual["wrapped"] = counts_actual["Actual Owner Type"].apply(wrap_label)
    fig_actual = px.bar(
        counts_actual,
        x="pct",
        y="wrapped",
        orientation="h",
        labels={"pct": "Percent of parcels", "wrapped": "Actual Owner Type"},
        text="label",
        color_discrete_sequence=["#d62728"],
    )

    fig_actual.update_layout(
        height=500,
        margin=dict(l=10, r=110, t=30, b=10),
        xaxis=dict(range=[0, max_actual_pct * 1.12]),
    )
    fig_actual.update_traces(
        textposition="outside",
        cliponaxis=False,   # <-- key line
    )

    st.plotly_chart(fig_actual, use_container_width=True)

# ----------------------
# Show attribute table (without geometry column)
# ----------------------

def filter_dataframe(df: pd.DataFrame):
    df = df.copy()
    for col in df.columns:
        col_data = df[col]
        has_unhashable = col_data.dropna().apply(
            lambda x: isinstance(x, (list, dict, set))
        ).any()

        if has_unhashable:
            # don't try to make a selector for this column
            continue

        unique_vals = col_data.dropna().unique()
    # drop geometry so we don't try to filter on it
    if "geometry" in df.columns:
        df = df.drop(columns=["geometry"])
    
    with st.expander("Filter table by column"):
        for col in df.columns:
            col_data = df[col]
            # categorical / short text
            if col_data.dtype == "object" or col_data.dtype.name == "category":
                unique_vals = col_data.dropna().unique()
                # if there aren't too many unique values, make a multiselect
                if len(unique_vals) > 1 and len(unique_vals) <= 50:
                    selected = st.multiselect(f"{col}", options=sorted(unique_vals), default=sorted(unique_vals))
                    df = df[df[col].isin(selected)]
                else:
                    # fallback: text contains
                    text = st.text_input(f"{col} contains", "")
                    if text:
                        df = df[df[col].astype(str).str.contains(text, case=False, na=False)]
            # numeric
            elif pd.api.types.is_numeric_dtype(col_data):
                min_val = float(col_data.min())
                max_val = float(col_data.max())
                step = (max_val - min_val) / 100 if max_val > min_val else 1.0
                vals = st.slider(f"{col} range", min_val, max_val, (min_val, max_val), step=step)
                df = df[(df[col] >= vals[0]) & (df[col] <= vals[1])]
            # dates
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                start, end = st.date_input(f"{col} range", [col_data.min(), col_data.max()])
                df = df[(df[col] >= pd.to_datetime(start)) & (df[col] <= pd.to_datetime(end))]
    return df

st.subheader("Data")
st.dataframe(filtered.drop(columns="geometry", errors="ignore"))

# filtered_table = filter_dataframe(filtered)
# st.dataframe(filtered_table)