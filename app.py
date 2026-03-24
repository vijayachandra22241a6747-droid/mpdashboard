import re
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium
from shapely.geometry import shape, Point

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Telangana Accident Dashboard", layout="wide")

# =========================
# ZONE MAP (your mapping)
# =========================
ZONE_MAP = {
    "Zone-I – Kaleswaram": [
        "Komaram Bheem Asifabad", "Mancherial", "Peddapalli",
        "Jayashankar Bhupalpally", "Mulugu"
    ],
    "Zone-II – Basara": ["Adilabad", "Nirmal", "Nizamabad", "Jagtial"],
    "Zone-III – Rajanna": ["Karimnagar", "Rajanna Sircilla", "Siddipet", "Medak", "Kamareddy"],
    "Zone-IV – Bhadradri": ["Bhadradri Kothagudem", "Khammam", "Mahabubabad", "Warangal", "Hanumakonda"],
    "Zone-V – Yadadri": ["Suryapet", "Nalgonda", "Yadadri Bhuvanagiri", "Jangaon"],
    "Zone-VI – Charminar": ["Medchal Malkajgiri", "Hyderabad", "Rangareddy", "Sangareddy", "Vikarabad"],
    "Zone-VII – Jogulamba": ["Mahabubnagar", "Narayanpet", "Jogulamba Gadwal", "Wanaparthy", "Nagarkurnool"]
}

def norm(s):
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())

DISTRICT_TO_ZONE = {}
for z, ds in ZONE_MAP.items():
    for d in ds:
        DISTRICT_TO_ZONE[norm(d)] = z

ALIASES = {
    "asifabad komarambheem": "komaram bheem asifabad",
    "komaram bheem asifabad": "komaram bheem asifabad",
    "jagityal": "jagtial",
    "sircilla rajanna": "rajanna sircilla",
    "rajanna sircilla": "rajanna sircilla",
    "kothagudem bhadadri": "bhadradri kothagudem",
    "bhadadri kothagudem": "bhadradri kothagudem",
    "bhongir yadadri": "yadadri bhuvanagiri",
    "yadadri bhongir": "yadadri bhuvanagiri",
    "medchal malkajgiri": "medchal malkajgiri",
    "ranga reddy": "rangareddy",
    "sanga reddy": "sangareddy",
    "mahaboobnagar": "mahabubnagar",
    "warangal rural": "warangal"
}

def canonical(name):
    n = norm(name)
    return ALIASES.get(n, n)

def pick_col(frame, cands):
    for c in cands:
        if c in frame.columns:
            return c
    return None

# =========================
# FILE INPUT
# =========================
st.title("🛣 Telangana Accident Dashboard")
st.caption("Main state map + district detail pages (within Streamlit app)")

colA, colB = st.columns(2)
with colA:
    xlsx_file = st.file_uploader("Upload accident XLSX", type=["xlsx"])
with colB:
    geojson_file = st.file_uploader("Upload Telangana districts GeoJSON", type=["geojson", "json"])

if not xlsx_file or not geojson_file:
    st.info("Upload both files to continue.")
    st.stop()

# =========================
# LOAD DATA
# =========================
df = pd.read_excel(xlsx_file, engine="openpyxl")
gj = json.load(geojson_file)

if "Latitude" not in df.columns or "Longitude" not in df.columns:
    st.error("XLSX must contain Latitude and Longitude columns.")
    st.stop()

death_col = pick_col(df, ["No. of Deaths", "Deaths"])
inj_col = pick_col(df, ["No. of Injured Persons", "Injured"])
ps_col = pick_col(df, ["Police Station"])
off_col = pick_col(df, ["Type of Offence"])
time_col = pick_col(df, ["Offence Time"])

v = df.copy()
v["Latitude"] = pd.to_numeric(v["Latitude"], errors="coerce")
v["Longitude"] = pd.to_numeric(v["Longitude"], errors="coerce")
v = v.dropna(subset=["Latitude", "Longitude"])
v = v[(v["Latitude"] >= 15.5) & (v["Latitude"] <= 20.5) & (v["Longitude"] >= 76.5) & (v["Longitude"] <= 82.5)].copy()

v["Deaths_num"] = pd.to_numeric(v[death_col], errors="coerce").fillna(0) if death_col else 0
v["Injured_num"] = pd.to_numeric(v[inj_col], errors="coerce").fillna(0) if inj_col else 0

if time_col:
    def hr(x):
        try:
            return int(str(x).split(":")[0])
        except:
            return np.nan
    v["hour"] = v[time_col].apply(hr)
else:
    v["hour"] = np.nan

# district key detection
props0 = gj["features"][0]["properties"]
possible_keys = ["district","District","DISTRICT","DIST_NAME","dist_name","NAME","name"]
district_key = next((k for k in possible_keys if k in props0), None)
if district_key is None:
    district_key = list(props0.keys())[0]

# make shapely geometries
district_geoms = []
for ft in gj["features"]:
    d_raw = str(ft["properties"].get(district_key, "")).strip()
    d_can = canonical(d_raw)
    ft["properties"]["district_raw"] = d_raw
    ft["properties"]["district_norm"] = d_can
    district_geoms.append((d_raw, d_can, shape(ft["geometry"])))

def map_district(lat, lon):
    p = Point(float(lon), float(lat))
    for d_raw, d_can, geom in district_geoms:
        if geom.contains(p) or geom.touches(p):
            return d_raw, d_can
    return "Unknown", "unknown"

mapped = v.apply(lambda r: map_district(r["Latitude"], r["Longitude"]), axis=1)
v["District_raw"] = mapped.apply(lambda x: x[0])
v["District_norm"] = mapped.apply(lambda x: x[1])
v["Zone"] = v["District_norm"].apply(lambda x: DISTRICT_TO_ZONE.get(x, "Unknown"))

# aggregate stats for polygons
agg = v.groupby("District_norm", dropna=False).agg(
    accidents=("District_norm", "size"),
    deaths=("Deaths_num", "sum"),
    injured=("Injured_num", "sum")
).reset_index()
agg_map = {r["District_norm"]: r for _, r in agg.iterrows()}

for ft in gj["features"]:
    d = ft["properties"]["district_norm"]
    r = agg_map.get(d)
    ft["properties"]["zone"] = DISTRICT_TO_ZONE.get(d, "Unknown")
    ft["properties"]["accidents"] = int(r["accidents"]) if r is not None else 0
    ft["properties"]["deaths"] = int(r["deaths"]) if r is not None else 0
    ft["properties"]["injured"] = int(r["injured"]) if r is not None else 0

district_list = sorted([ft["properties"]["district_norm"] for ft in gj["features"]])

# =========================
# NAVIGATION STATE
# =========================
if "selected_district" not in st.session_state:
    st.session_state.selected_district = None

# sidebar manual selector (fallback if map click not available)
sel = st.sidebar.selectbox(
    "Select District",
    ["-- Main Telangana View --"] + [d.title() for d in district_list],
    index=0
)
if sel != "-- Main Telangana View --":
    st.session_state.selected_district = norm(sel)

# =========================
# MAIN PAGE
# =========================
if st.session_state.selected_district is None or st.session_state.selected_district not in district_list:
    st.subheader("Telangana - State View")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Accidents", int(len(v)))
    c2.metric("Total Deaths", int(v["Deaths_num"].sum()))
    c3.metric("Total Injured", int(v["Injured_num"].sum()))

    m = folium.Map(location=[17.8, 79.1], zoom_start=7, tiles="CartoDB dark_matter", control_scale=True)

    # heat
    plugins.HeatMap(
        v[["Latitude","Longitude"]].values.tolist(),
        radius=11, blur=16, min_opacity=0.2,
        gradient={0.2:"#2ECC71",0.5:"#F1C40F",0.8:"#E67E22",1.0:"#E74C3C"}
    ).add_to(m)

    max_acc = max([ft["properties"]["accidents"] for ft in gj["features"]] + [1])
    def style_fn(ft):
        a = ft["properties"]["accidents"]
        t = min(1.0, a/max_acc)
        if t > 0.75: c = "#8B0000"
        elif t > 0.50: c = "#C0392B"
        elif t > 0.25: c = "#E67E22"
        elif t > 0.10: c = "#F4D03F"
        else: c = "#1E8449"
        return {"fillColor": c, "color":"#fff", "weight":1.2, "fillOpacity":0.45}

    tooltip = folium.GeoJsonTooltip(
        fields=["district_raw","zone","accidents","deaths","injured"],
        aliases=["District","Zone","Accidents","Deaths","Injured"],
        sticky=False
    )

    folium.GeoJson(
        gj,
        name="Districts",
        style_function=style_fn,
        highlight_function=lambda f: {"weight":2.5,"color":"#00E5FF","fillOpacity":0.65},
        tooltip=tooltip
    ).add_to(m)

    out = st_folium(m, width=1400, height=650)

    # NOTE: reliable selection via sidebar.
    st.info("Select district from left sidebar to open district detail page.")

# =========================
# DISTRICT PAGE
# =========================
else:
    d = st.session_state.selected_district
    dv = v[v["District_norm"] == d].copy()

    top = st.columns([1, 5])
    with top[0]:
        if st.button("⬅ Back"):
            st.session_state.selected_district = None
            st.rerun()
    with top[1]:
        st.subheader(f"{d.title()} - Detailed Analysis")

    if dv.empty:
        st.warning("No accident data for this district.")
        st.stop()

    zone = DISTRICT_TO_ZONE.get(d, "Unknown")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Zone", zone)
    c2.metric("Accidents", int(len(dv)))
    c3.metric("Deaths", int(dv["Deaths_num"].sum()))
    c4.metric("Injured", int(dv["Injured_num"].sum()))

    # district map
    dm = folium.Map(location=[dv["Latitude"].mean(), dv["Longitude"].mean()], zoom_start=9, tiles="CartoDB dark_matter")
    plugins.HeatMap(
        dv[["Latitude","Longitude"]].values.tolist(),
        radius=14, blur=20, min_opacity=0.25,
        gradient={0.2:"#2ECC71",0.5:"#F1C40F",0.8:"#E67E22",1.0:"#E74C3C"}
    ).add_to(dm)
    st_folium(dm, width=1400, height=500)

    # tables + trend
    a, b = st.columns(2)
    with a:
        st.markdown("#### Top Police Stations")
        if ps_col:
            ps_tbl = dv.groupby(ps_col).size().sort_values(ascending=False).head(20).reset_index(name="Accidents")
            st.dataframe(ps_tbl, use_container_width=True)
        else:
            st.info("Police Station column not found.")

    with b:
        st.markdown("#### Top Offence Types")
        if off_col:
            off_tbl = dv.groupby(off_col).size().sort_values(ascending=False).head(20).reset_index(name="Count")
            st.dataframe(off_tbl, use_container_width=True)
        else:
            st.info("Type of Offence column not found.")

    st.markdown("#### Hourly Trend")
    hr_tbl = dv.dropna(subset=["hour"]).groupby("hour").size().reindex(range(24), fill_value=0).reset_index(name="Accidents")
    st.line_chart(hr_tbl.set_index("hour"))
