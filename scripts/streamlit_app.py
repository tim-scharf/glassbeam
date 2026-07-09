"""
streamlit_app.py
----------------
Visualize trained embeddings: region/focus extraction and t-SNE clusters.

Usage:
    streamlit run scripts/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from region_extractor import (
    build_region_embeddings_lookup,
    build_focus_embeddings_lookup,
    decode_embedding_column,
    extract_region,
    extract_focus,
)


MODEL_OPTIONS = {
    'Fine-tuned (all_minilm_stratified)': 'output/all_minilm_stratified',
    'Untuned base (all-MiniLM-L6-v2)': 'output/all_minilm_base',
    'MedEmbed-base-v0.1 (untuned baseline)': 'output/medembed_base',
}


@st.cache_resource
def load_model(model_path):
    """Load a SentenceTransformer model from a local path."""
    from sentence_transformers import SentenceTransformer
    try:
        model = SentenceTransformer(model_path)
        return model
    except Exception:
        st.error(f"Model not found at {model_path}.")
        return None


@st.cache_resource
def prepare_data(model_path):
    """Load region and focus embedding lookup tables for the given model."""
    model = load_model(model_path)
    if model is None:
        return None

    region_df = build_region_embeddings_lookup(model)
    region_embeddings = decode_embedding_column(region_df['embedding'])

    focus_df = build_focus_embeddings_lookup(model)
    focus_embeddings = decode_embedding_column(focus_df['embedding'])

    return {
        'model': model,
        'region_df': region_df,
        'region_embeddings': region_embeddings,
        'focus_df': focus_df,
        'focus_embeddings': focus_embeddings,
    }


def main():
    st.set_page_config(page_title="Embedding Explorer", layout="wide")
    st.title("🧬 Triplet Loss Embedding Validator")

    st.sidebar.header("Controls")
    model_choice = st.sidebar.selectbox("Embedding model", list(MODEL_OPTIONS.keys()))
    model_path = MODEL_OPTIONS[model_choice]

    # Load data
    data = prepare_data(model_path)
    if data is None:
        st.stop()

    model = data['model']
    region_df, region_embeddings = data['region_df'], data['region_embeddings']
    focus_df, focus_embeddings = data['focus_df'], data['focus_embeddings']

    foci = focus_df['focus'].tolist()
    regions = focus_df['region'].tolist()
    embeddings = focus_embeddings

    # Create dataframe
    df = pd.DataFrame({
        'focus': foci,
        'region': regions,
    })

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Region & Focus Extractor", "📊 t-SNE 2D", "🎲 t-SNE 3D", "📈 Statistics"])

    with tab1:
        st.header("Region & Focus Extractor")
        st.caption(
            "Embeds each word in the query individually and returns the region/focus "
            "closest to any single word — avoids noise from modality codes, view counts, etc."
        )

        query = st.text_input(
            "Enter study text:",
            value="MR shoulder 3 views",
            key="extractor_query"
        )

        if query:
            region_result = extract_region(query, model, region_df['region'].tolist(), region_embeddings)
            focus_result = extract_focus(query, model, focus_df['focus'].tolist(), focus_df['region'].tolist(), focus_embeddings)

            st.subheader(f"Results for: **{query}**")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Predicted Region**")
                if region_result:
                    st.metric("Region", region_result['region'])
                    st.caption(f"matched word: **{region_result['matched_word']}** · similarity: {region_result['similarity']:.4f}")
                else:
                    st.warning("No confident region match")

            with col2:
                st.write("**Predicted Focus**")
                if focus_result:
                    st.metric("Focus", focus_result['focus'])
                    st.caption(f"matched word: **{focus_result['matched_word']}** · mapped region: {focus_result['region']} · similarity: {focus_result['similarity']:.4f}")
                else:
                    st.warning("No confident focus match")

            if region_result and focus_result and region_result['region'] != focus_result['region']:
                st.info("⚠️ Region and focus predictions disagree on anatomical region — worth reviewing.")

    with tab2:
        st.header("t-SNE Embedding Space")

        with st.spinner("Computing t-SNE (this may take a moment)..."):
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_2d = tsne.fit_transform(embeddings)

        # Create interactive plot
        df_plot = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'focus': foci,
            'region': regions,
        })

        fig = px.scatter(
            df_plot,
            x='x',
            y='y',
            color='region',
            hover_data=['focus', 'region'],
            title='Embedding Space (t-SNE)',
            labels={'x': 't-SNE Dim 1', 'y': 't-SNE Dim 2'},
            height=700,
        )

        fig.update_traces(marker=dict(size=8, opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)

        st.info("✓ Good clustering: each region should form a tight cluster")

    with tab3:
        st.header("3D t-SNE Embedding Space")

        with st.spinner("Computing 3D t-SNE (this may take a moment)..."):
            tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30)
            embeddings_3d = tsne_3d.fit_transform(embeddings)

        # Create interactive 3D plot
        df_plot_3d = pd.DataFrame({
            'x': embeddings_3d[:, 0],
            'y': embeddings_3d[:, 1],
            'z': embeddings_3d[:, 2],
            'focus': foci,
            'region': regions,
        })

        fig_3d = px.scatter_3d(
            df_plot_3d,
            x='x',
            y='y',
            z='z',
            color='region',
            hover_data=['focus', 'region'],
            title='3D Embedding Space (t-SNE)',
            labels={'x': 't-SNE Dim 1', 'y': 't-SNE Dim 2', 'z': 't-SNE Dim 3'},
            height=800,
        )

        fig_3d.update_traces(marker=dict(size=5, opacity=0.7))
        st.plotly_chart(fig_3d, use_container_width=True)

        st.info("🎲 Rotate, zoom, and pan to explore the 3D embedding space")

    with tab4:
        st.header("Validation Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Foci", len(foci))
            st.metric("Total Regions", df['region'].nunique())

        with col2:
            st.metric("Embedding Dim", embeddings.shape[1])

        with col3:
            st.metric("Avg Foci/Region", len(foci) / df['region'].nunique())

        st.subheader("Foci per Region")
        region_counts = df['region'].value_counts().sort_values(ascending=False)
        fig_bar = px.bar(
            x=region_counts.index,
            y=region_counts.values,
            labels={'x': 'Region', 'y': 'Count'},
            height=400,
        )
        st.plotly_chart(fig_bar, use_container_width=True)


if __name__ == '__main__':
    main()
