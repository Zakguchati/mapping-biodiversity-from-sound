# Mapping Biodiversity from Sound

MSc Individual Project (COMP1885). Identifying animal species from field sound
recordings with machine learning, then turning those detections into a spatial
map of biodiversity across recording sites.

**Author:** Zakariya Guechchati
**Dataset:** [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)
(Pantanal wetlands, 234 species across birds, amphibians, reptiles, mammals and
insects)

## Approach

1. Convert soundscape audio into mel-spectrograms.
2. Classify species per clip. Two tiers: a baseline using pretrained audio
   embeddings, and a fine-tuned EfficientNet CNN on the spectrograms. Compare
   the two.
3. Explain predictions with Grad-CAM.
4. Aggregate detections by site, compute a diversity measure per site, and plot
   it as a spatial map. This spatial layer is the main contribution.
5. Deploy an interactive Streamlit app for exploring the map.

## Repository layout

    data/        # datasets, not committed (see .gitignore)
    notebooks/   # exploration and experiments
    src/         # reusable pipeline code
    models/      # trained weights, not committed
    reports/     # write-up drafts, figures
    app/         # Streamlit app

## Setup

Most heavy training runs on free cloud GPUs (Kaggle / Colab), since the local
machine has no suitable GPU. On Kaggle the competition data mounts at
`/kaggle/input/birdclef-2026/` and most libraries below are preinstalled.

Local (exploration, mapping, app):

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

To pull the data locally you need a Kaggle API token (`kaggle.json`) and to have
accepted the competition rules:

    kaggle competitions download -c birdclef-2026 -p data/

## Progress

Task 1 (background research and planning) is underway. Start with
`notebooks/01_data_exploration.ipynb`, which inventories the dataset and settles
whether the soundscapes carry real coordinates or only site codes.

## Data and licence

Data is used under the Kaggle competition rules for academic purposes and is not
redistributed in this repository.
