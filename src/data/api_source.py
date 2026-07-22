import requests
import pandas as pd

from .base import DataSource
from .exceptions import DataSourceError


class APISource(DataSource):
    def __init__(
        self,
        url: str,
        params: dict | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._url = url
        self._params = params
        self._timeout = timeout
        self.connect()

    def connect(self) -> None:
        """Check that the API endpoint is reachable.

        Raises:
            DataSourceError: If the request fails.
        """
        try:
            response = requests.get(self._url, params=self._params, timeout=self._timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise DataSourceError(f"API not reachable: {self._url}") from e

    def fetch(self) -> pd.DataFrame:
        """Retrieve data from the API endpoint.

        Returns:
            The raw JSON response as a DataFrame.

        Raises:
            DataSourceError: If the request fails or the response is not valid JSON.
        """
        try:
            response = requests.get(self._url, params=self._params, timeout=self._timeout)
            response.raise_for_status()
            return pd.DataFrame(response.json())
        except (requests.RequestException, ValueError) as e:
            raise DataSourceError(f"Failed to fetch data from API: {self._url}") from e
