# Technical results

## CGM dataset

- 64 participants
- 864 original readings per participant
- 53,760 model-ready observations
- 24 historical readings per observation
- 51 engineered features
- Approximate 5-minute prediction horizon

## V2 participant split

| Dataset | Participants | Observations |
|---|---:|---:|
| Training | 44 | 36,960 |
| External validation | 10 | 8,400 |
| Reserved final test | 10 | 8,400 |

The split was performed by participant to reduce information leakage.

## IBM AutoAI

Selected model:

`Glucose_Profile_V2_Ridge_Model`

| Metric | Holdout | Cross-validation |
|---|---:|---:|
| RMSE | 0.785 | 0.814 |
| MAE | 0.617 | 0.627 |
| R² | 0.998 | 0.998 |

## External validation

| Metric | Result |
|---|---:|
| Within 2.5 mg/dL | 99.39% |
| Within 5 mg/dL | 99.92% |
| MAE | 0.675 mg/dL |
| Absolute error P95 | 1.559 mg/dL |

The online Python test sent three records to the endpoint and received an HTTP 200 response. All three demonstration predictions were within 2.5 mg/dL.

## Considerations

The high performance is associated with the short prediction horizon and strong autocorrelation in CGM time series. This model is not a clinical tool.
