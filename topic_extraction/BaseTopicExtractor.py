import abc


class TopicExtractor(abc.ABC):
    @abc.abstractmethod
    def prepare(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def extract(self, document: str, k: int, *args, **kwargs) -> list:
        pass

    @abc.abstractmethod
    def batch_extract(self, documents: list[str], k: int, *args, **kwargs) -> list[list]:
        pass
