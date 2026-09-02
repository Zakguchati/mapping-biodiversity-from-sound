import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from pathlib import Path
import numpy as np
import folium
import json

DATA = Path(__file__).parent.parent / 'data'
REPORTS = Path(__file__).parent.parent / 'reports'


st.set_page_config(page_title="Mapping Biodiversity from Sound", layout="wide")

st.title("Mapping Biodiversity from Sound")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sites monitored", "9")
c2.metric("Species detectable", "47", help="of 75 heard in the field")
c3.metric("Model ROC-AUC", "0.81")
c4.metric("Mean undercount", "73%", help="species missed per segment on average")
st.divider()

with st.sidebar:
    st.subheader("About")
    st.markdown(
        "Acoustic biodiversity monitoring across nine Pantanal sites. "
        "Species are detected from soundscape recordings, aggregated into "
        "diversity indices, and mapped alongside a measure of how far the "
        "model-based estimate can be trusted."
    )
    st.subheader("Map legend")
    st.markdown(
        "- **green** - model estimate reasonably reliable\n"
        "- **red** - model under-counts (rich sites)\n"
        "- **purple** - model over-counts (sparse sites)\n"
        "- **circle size** - true diversity"
    )
    st.caption("MSc Individual Project, COMP1885")

tab_map, tab_div, tab_time, tab_space, tab_site, tab_model, tab_limit = st.tabs(
    ["Map", "Diversity", "Time of day", "Spatial pattern", "Site explorer",
     "Model", "The blind spot"]
)

with tab_map:
    st.subheader("Biodiversity map")
    st.markdown(
        "Circle colour shows how reliable the model's diversity estimate is at each site, "
        "fixed at the 0.30 baseline. The slider below changes the detection threshold and "
        "shows how many species are recovered overall."
    )

    @st.cache_data
    def load_map_inputs():
        pp = np.load(DATA / 'app_pred_prob.npy')
        ty = np.load(DATA / 'app_true_Y.npy')
        meta = json.loads((DATA / 'app_meta.json').read_text())
        return pp, ty, meta

    pred_prob, true_Y, meta = load_map_inputs()
    sites_arr = np.array(meta['sites'])
    coords = meta['site_coords']

    # reliability colour is FIXED at the 0.30 baseline so the legend always holds
    base_present = (pred_prob >= 0.30).astype(int)

    def colour(gap):
        if gap > 3:  return '#c0392b'
        if gap < -1: return '#8e44ad'
        return '#27ae60'

    center = [np.mean([c[0] for c in coords.values()]),
              np.mean([c[1] for c in coords.values()])]
    m = folium.Map(location=center, zoom_start=7, tiles='CartoDB positron')

    for s in sorted(set(sites_arr)):
        idx = np.where(sites_arr == s)[0]
        tr = int((true_Y[idx].sum(axis=0) > 0).sum())
        pr = int((base_present[idx].sum(axis=0) > 0).sum())
        gap = tr - pr
        lat, lon = coords[s]
        folium.CircleMarker(
            location=[lat, lon],
            radius=6 + tr * 1.5,
            popup=f"<b>{s}</b><br>true: {tr}<br>detected: {pr}<br>gap: {gap}",
            color=colour(gap), fill=True, fill_opacity=0.7,
        ).add_to(m)

    with st.container(border=True):
        components.html(m._repr_html_(), height=500)

    st.markdown("**Detection threshold**")
    threshold = st.slider("threshold", 0.1, 0.9, 0.30, 0.05, label_visibility='collapsed')
    present = (pred_prob >= threshold).astype(int)
    detected_total = int((present.sum(axis=0) > 0).sum())
    true_total = int((true_Y.sum(axis=0) > 0).sum())
    st.caption(
        f"At threshold {threshold}: {detected_total} of {true_total} species detected "
        f"across all sites. Map colours stay fixed at the 0.30 baseline."
    )

with tab_div:
    st.subheader("Diversity at equal sampling effort")
    st.markdown(
        "Species richness per site, rarefied to 24 segments so sites recorded for "
        "different lengths compare fairly. Higher means more species detected at equal effort."
    )
    rare = pd.read_csv(DATA / 'rarefied_richness.csv')
    rare = rare.rename(columns={
        'site': 'Site',
        'n_segments': 'Segments recorded',
        'raw_richness': 'Raw species count',
        'rarefied_true': 'True richness (equal effort)',
        'rarefied_pred': 'Predicted richness (equal effort)',
    })
    st.dataframe(rare, width='stretch', hide_index=True)
    st.image(str(REPORTS / 'diversity_ci.png'), width='stretch')   

with tab_time:
    st.subheader("Diversity by time of day")
    st.markdown(
        "Within S22, the only site recorded across all periods, richness peaks at night. "
        "Across sites the comparison is confounded, since most recorders ran at a single period."
    )
    st.image(str(REPORTS / 'temporal_heatmap.png'), width='stretch')
   

with tab_space:
    st.subheader("North-south pattern")
    st.markdown(
        "A suggestive rise in diversity from south to north (Spearman rho = 0.75), "
        "resting on schematic positions and driven largely by S22, so indicative only."
    )
    st.image(str(REPORTS / 'spatial_gradient.png'), width='stretch')

with tab_site:
    st.subheader("Explore a single site")
    grid = pd.read_csv(DATA / 'diversity_grid.csv')
    site = st.selectbox("Choose a site", sorted(grid['site'].unique()))

    sub = grid[grid['site'] == site]
    total_segs = int(sub['n_segments'].sum())
    true_rich = int(sub['true_richness'].max())
    pred_rich = int(sub['pred_richness'].max())

    c1, c2, c3 = st.columns(3)
    c1.metric("Segments recorded", total_segs)
    c2.metric("True species richness", true_rich)
    c3.metric("Model detected", pred_rich, delta=f"{pred_rich - true_rich} vs true")

    st.markdown("**Diversity by time of day at this site**")
    st.dataframe(
        sub[['period', 'n_segments', 'true_richness', 'pred_richness',
             'true_shannon', 'pred_shannon']]
        .rename(columns={
            'period': 'Period', 'n_segments': 'Segments',
            'true_richness': 'True species', 'pred_richness': 'Detected species',
            'true_shannon': 'True Shannon', 'pred_shannon': 'Detected Shannon',
        }),
        width='stretch', hide_index=True,
    )
with st.expander("How to read this, and its limitations"):
    st.markdown(
        "- Diversity is estimated from model detections, which miss species with no training audio.\n"
        "- Site positions are schematic; exact recorder coordinates are not published.\n"
        "- The predicted map flattens the true signal: it over-counts sparse sites and under-counts rich ones.\n"
        "- Spatial and temporal comparisons across sites are limited by uneven sampling."
    )

with tab_model:
    st.subheader("How the detection model was chosen")
    st.markdown(
        "The map rests on a species detector. Three approaches were compared on the same "
        "field audio, scored by macro ROC-AUC (0.5 is chance, 1.0 perfect):"
    )
    model_tbl = pd.DataFrame({
        'Approach': [
            'Perch embeddings + logistic regression',
            'EfficientNet CNN on spectrograms',
            'Trained on field audio directly',
        ],
        'ROC-AUC': [0.81, 0.55, 0.56],
        'Verdict': ['Chosen baseline', 'Overfit, failed to transfer', 'Too little field data'],
    })
    st.dataframe(model_tbl, width='stretch', hide_index=True)
    st.markdown(
        "A bioacoustics foundation model's frozen embeddings decisively beat a heavier image "
        "model and direct field training. With scarce, noisy data, strong pretraining carries "
        "the task, which is why the map is built on the Perch baseline."
    )

with tab_limit:
    st.subheader("Why the map undercounts, by design")
    st.markdown(
        "Of the 75 species heard in the field, 28 have no training audio and can never be "
        "detected. So predicted richness cannot reach true richness at any threshold, this is "
        "structural, not a tuning problem. The curve below shows predicted richness staying "
        "below the true total everywhere."
    )
    st.image(str(REPORTS / 'threshold_sweep.png'), width='stretch')
    st.markdown(
        "This is the core reliability finding: any diversity map from this model systematically "
        "undercounts, most where insect and untrained species dominate."
    )