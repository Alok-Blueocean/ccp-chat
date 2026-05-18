from abc import ABC, abstractmethod


class BaseQueryTransform(ABC):

    @abstractmethod
    def transform(self, query: str):
        pass