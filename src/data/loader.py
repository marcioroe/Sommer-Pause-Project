import pandas as pd

from .base import DataSource


class DataLoader:
    def __init__(self, source: DataSource) -> None:
        self._source = source

    def load(self) -> pd.DataFrame:
        """Fetch data from the source and apply generic cleaning.

        Returns:
            The cleaned DataFrame.
        """
        df = self._source.fetch()
        return self._clean(df)

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning steps common to all data sources.

        Args:
            df: The raw DataFrame to clean.

        Returns:
            The cleaned DataFrame with normalized column names.
        """
        return df.rename(columns=lambda col: col.strip().lower().replace(" ", "_"))
