"""CarsXE API wrapper for market pricing lookups (HTTP based).

The REST quickstart from CarsXE looks like::

    import requests
    params = {"key": api_key, "year": 2023, "make": "Toyota", ...}
    requests.get("https://api.carsxe.com/v1/ymm", params=params)

This module wraps that HTTP flow with some convenience helpers and graceful
fallbacks when the API key is missing.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


YMM_ENDPOINT = "https://api.carsxe.com/v1/ymm"
VIN_ENDPOINT = "https://api.carsxe.com/v1/vin"


class CarsXENotConfigured(RuntimeError):
    """Raised when the CarsXE client is unavailable."""


class CarsXEClient:
    """Thin wrapper around the CarsXE API client.

    The class caches the underlying SDK instance and exposes a couple of helper
    methods tuned for used-car analysis.  All network calls may raise requests
    errors, so callers should wrap invocations accordingly.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("CARSXE_API_KEY")
        self._session: Optional[requests.Session] = None

        if not self.api_key:
            logger.info("CARSXE_API_KEY not set; CarsXEClient disabled")
            return

        self._session = requests.Session()
        logger.info("CarsXE HTTP client initialised")

    @property
    def available(self) -> bool:
        """Whether the underlying CarsXE SDK is ready to use."""

        return self._session is not None

    def _guard(self):
        if not self.available:
            raise CarsXENotConfigured(
                "CarsXE client unavailable. Set CARSXE_API_KEY before using the helper."
            )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def fetch_market_value_by_vin(self, vin: str, **kwargs: Any) -> Dict[str, Any]:
        """Retrieve market valuation by VIN.

        Parameters are forwarded to the `/v1/vin` endpoint. Refer to CarsXE API
        docs for available options (e.g. `state`, `mileage`).
        """

        self._guard()
        assert self._session is not None

        params = {"key": self.api_key, "vin": vin}
        params.update(kwargs)

        resp = self._session.get(VIN_ENDPOINT, params=params, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(f"CarsXE VIN lookup failed ({resp.status_code}): {resp.text}")
        return resp.json()

    def fetch_market_value_by_trim(
        self,
        *,
        make: str,
        model: str,
        year: int,
        trim: Optional[str] = None,
        mileage: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Retrieve pricing insights using make / model / year (+ optional trim).

        For providers requiring VIN this may be less precise, but CarsXE exposes
        the `/v1/ymm` endpoint that accepts these fields as an alternative.
        Additional keyword arguments are forwarded to the HTTP call.
        """

        self._guard()
        assert self._session is not None

        params: Dict[str, Any] = {
            "key": self.api_key,
            "year": year,
            "make": make,
            "model": model,
        }
        if trim:
            params["trim"] = trim
        if mileage is not None:
            params["mileage"] = mileage
        params.update(kwargs)

        resp = self._session.get(YMM_ENDPOINT, params=params, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(f"CarsXE YMM lookup failed ({resp.status_code}): {resp.text}")
        return resp.json()


# Singleton helper for quick imports
carsxe_client = CarsXEClient()
