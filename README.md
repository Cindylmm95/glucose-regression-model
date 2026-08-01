# Glucose Profile V2

[![Live app](https://img.shields.io/badge/Live%20App-Streamlit-6941C6)](https://glucose-profile-predictor.streamlit.app)
[![Project page](https://img.shields.io/badge/Project%20Page-GitHub%20Pages-3E176F)](https://cindylmm95.github.io/glucose-regression-model/)

Interactive machine learning project that predicts the next CGM glucose reading, approximately 5 minutes into the future.

The project combines time-series feature engineering, IBM AutoAI, IBM Machine Learning deployments, Python API validation and a Streamlit interface.

## Interactive demo

The public interface allows visitors to:

- Generate stable, rising, falling or rapidly changing glucose simulations.
- Edit a custom sequence of 24 readings.
- Visualize a 2-hour history.
- Request a prediction from the IBM online deployment.
- Compare the current reading with the 5-minute prediction.

## Dataset

The final model uses a public continuous glucose monitoring dataset.

- 64 participants
- 864 original readings per participant
- 53,760 model-ready observations
- 24 historical readings per prediction
- 51 engineered features
- Approximate 5-minute prediction horizon

Training, validation and final test groups were separated by participant to reduce information leakage.

## Model evolution

### V1

- Ridge Regression
- StandardScaler
- Manual time-series feature engineering
- Grouped validation by participant

### V2

- IBM AutoAI experiment
- Eight regression pipelines compared
- Independent participant-level validation
- Batch deployment
- Online deployment
- Python API verification with HTTP 200

## Selected model

`Glucose_Profile_V2_Ridge_Model`

| Metric | Result |
|---|---:|
| Holdout RMSE | 0.785 |
| Cross-validation RMSE | 0.814 |
| R² | 0.998 |
| External MAE | 0.675 mg/dL |
| Within 2.5 mg/dL | 99.39% |
| Grouped 10-fold validation | 99.50% |

The external validation included 8,400 observations from 10 participants excluded from training.

## Architecture

```text
GitHub Pages project site
        |
Embedded Streamlit interface
        |
Python feature engineering
        |
IBM Cloud IAM authentication
        |
IBM Machine Learning online deployment
        |
Prediction and visualization
```

GitHub Pages hosts the static portfolio page. Streamlit Community Cloud runs the Python application and communicates with IBM from the server.

## Security

- The IBM API key is stored in Streamlit secrets.
- The scoring endpoint is not included in the repository.
- The browser does not receive the API key.
- IBM requests are sent from the Streamlit backend.
- Local secret files are excluded through `.gitignore`.

## Technologies

- Python
- pandas
- NumPy
- Plotly
- Streamlit
- IBM AutoAI
- IBM Machine Learning
- IBM Cloud IAM
- REST API
- HTML and CSS
- GitHub Pages

## Repository structure

```text
app.py
glucose_features.py
ibm_client.py
simulations.py
requirements.txt
data/
assets/
docs/
    index.html
    styles.css
    results.md
scripts/
```

## Limitations

This is an academic and experimental project.

It is not designed to:

- Diagnose diabetes
- Replace a clinical measurement
- Calculate insulin doses
- Recommend treatments
- Support medical decisions

Two extreme prediction errors were identified during external validation. Performance also decreased during rapid glucose changes.
