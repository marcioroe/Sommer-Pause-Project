from abc import ABC, abstractmethod
import pandas as pd


class DataSource(ABC):
    @abstractmethod
    def connect(self) -> None:
        """Validate that the source is reachable and ready to use.

        Raises:
            DataSourceError: If the source cannot be validated.
        """
        ...

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Retrieve raw data from the underlying source.

        Returns:
            The raw, unprocessed data as a DataFrame.

        Raises:
            DataSourceError: If the data cannot be retrieved from the source.
        """
        ...