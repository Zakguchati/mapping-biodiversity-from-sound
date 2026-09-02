# Mapping Biodiversity from Sound

MSc Individual Project (COMP1885). Identifying animal species from field sound
recordings with machine learning, turning those detections into a site-level map
of biodiversity, and measuring how reliable that map is.

**Author:** Zakariya Guechchati
**Dataset:** [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)
(Pantanal wetlands, 234 species across birds, amphibians, reptiles, mammals and
insects).

## Approach

1. Convert soundscape audio into 5-second segments and mel-spectrograms.
2. Generate embeddings for each segment using the pretrained Perch model.
3. Compare three modelling approaches: a Perch-embedding baseline with logistic
   regression, an EfficientNet CNN trained on spectrograms, and a model trained
   directly on the field data. The Perch baseline performed best and was adopted.
4. Evaluate with a site-based holdout and macro ROC-AUC, using a verified scoring
   harness.
5. Aggregate detections by site, compute diversity indices, and assess
   reliability with rarefaction and bootstrapped confidence intervals.
6. Present the results in an interactive Streamlit app.

The main contribution is the reliability layer: measuring where the resulting
biodiversity map can and cannot be trusted.

## Repository layout

    data/        # datasets, not committed (see .gitignore)
    notebooks/   # exploration, modelling and analysis (01 to 12)
    src/         # reusable pipeline code
    models/      # trained artefacts, not committed
    reports/     # figures and outputs
    app/         # Streamlit app

## Setup

Embedding extraction was run on Kaggle, where the competition data is mounted at
`/kaggle/input/competitions/birdclef-2026/`. Local analysis, mapping and the app
run in a virtual environment:

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

To retrieve the data you need a Kaggle API token and to have accepted the
competition rules:

    kaggle competitions download -c birdclef-2026 -p data/

## Running

The notebooks run in order, 01 (data exploration) through 12 (diversity map).
The evaluation harness is notebook 06. The Streamlit app is launched with:

    streamlit run app/streamlit_app.py

## Data and licence

Data is used under the Kaggle competition rules for academic purposes and is not
redistributed in this repository. The code documents how to obtain the dataset
from its original source.