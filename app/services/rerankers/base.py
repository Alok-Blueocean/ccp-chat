from abc import ABC, abstractmethod


class BaseReranker(ABC):

    @abstractmethod
    def rerank(self, query, nodes):
        pass