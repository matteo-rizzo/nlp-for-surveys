import abc


class BaseTopicExtractor(abc.ABC):
    @abc.abstractmethod
    def prepare(self, *args, **kwargs) -> None:
        pass

    def train(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def extract(self, document: str, k: int, *args, **kwargs) -> list:
        pass

    @abc.abstractmethod
    def batch_extract(self, documents: list[str], k: int, *args, **kwargs) -> list[list]:
        pass

    def plot_wonders(self, documents: list) -> None:
        pass
