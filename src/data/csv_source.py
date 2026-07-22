from pathlib import Path

import pandas as pd

from .base import DataSource
from .exceptions import DataSourceError


class CSVSource(DataSource):
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self.connect()

    def connect(self) -> None:
        """Check that the CSV file exists and is a file.

        Raises:
            DataSourceError: If the path does not exist or is not a file.
        """
        if not self._path.exists():
            raise DataSourceError(f"File not found: {self._path}")

        if not self._path.is_file():
            raise DataSourceError(f"Not a file: {self._path}")

    def fetch(self) -> pd.DataFrame:
        """Read the CSV file into a DataFrame.

        Returns:
            The raw CSV contents as a DataFrame.

        Raises:
            DataSourceError: If the file cannot be read as CSV.
        """
        try:
            return pd.read_csv(self._path)
        except Exception as e:
            raise DataSourceError(f"Failed to read CSV file: {self._path}") from e
