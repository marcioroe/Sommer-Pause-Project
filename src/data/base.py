from abc import ABC, abstractmethod
import pandas as pd


class DataSource(ABC): 
    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Retrieve raw data from the underlying source.

        Returns:
            The raw, unprocessed data as a DataFrame.

        Raises:
            DataSourceError: If the data cannot be retrieved from the source.
        """
        ...