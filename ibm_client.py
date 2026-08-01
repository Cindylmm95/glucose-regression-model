from __future__ import annotations

import time
from typing import Any

import requests


IAM_URL = "https://iam.cloud.ibm.com/identity/token"


class IBMScoringError(RuntimeError):
    pass


class IBMScoringClient:
    def __init__(
        self,
        api_key: str,
        scoring_url: str,
        timeout_seconds: int = 60,
    ) -> None:
        if not api_key:
            raise IBMScoringError("The IBM API key is not configured.")
        if not scoring_url.startswith("https://"):
            raise IBMScoringError("The IBM scoring endpoint is not valid.")

        self.api_key = api_key
        self.scoring_url = scoring_url
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self._access_token: str | None = None
        self._token_expires_at = 0.0

    def _get_access_token(self) -> str:
        if self._access_token and time.time() < self._token_expires_at:
            return self._access_token

        try:
            response = self.session.post(
                IAM_URL,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                    "apikey": self.api_key,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as error:
            raise IBMScoringError(
                "The request could not be authenticated with IBM Cloud."
            ) from error
        except ValueError as error:
            raise IBMScoringError(
                "IBM Cloud returned an invalid authentication response."
            ) from error

        token = payload.get("access_token")
        expires_in = int(payload.get("expires_in", 3600))

        if not token:
            raise IBMScoringError("IBM Cloud did not return an IAM token.")

        self._access_token = token
        self._token_expires_at = time.time() + max(expires_in - 120, 60)
        return token

    def score(
        self,
        fields: list[str],
        values: list[list[float]],
    ) -> list[float]:
        token = self._get_access_token()
        request_payload = {
            "input_data": [
                {
                    "fields": fields,
                    "values": values,
                }
            ]
        }

        try:
            response = self.session.post(
                self.scoring_url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=request_payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as error:
            raise IBMScoringError(
                "The IBM deployment could not process the prediction."
            ) from error
        except ValueError as error:
            raise IBMScoringError(
                "IBM returned a response that could not be interpreted."
            ) from error

        return self._extract_predictions(payload)

    @staticmethod
    def _extract_predictions(payload: dict[str, Any]) -> list[float]:
        try:
            prediction_block = payload["predictions"][0]
            response_fields = prediction_block.get("fields", [])
            response_values = prediction_block["values"]
        except (KeyError, IndexError, TypeError) as error:
            raise IBMScoringError(
                "The IBM response does not contain predictions."
            ) from error

        predictions: list[float] = []

        for row in response_values:
            if len(row) == 1:
                value = row[0]
            elif "prediction" in response_fields:
                value = row[response_fields.index("prediction")]
            else:
                value = row[0]

            if isinstance(value, list):
                if not value:
                    raise IBMScoringError("IBM returned an empty prediction.")
                value = value[0]

            predictions.append(float(value))

        return predictions
