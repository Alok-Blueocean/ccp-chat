from abc import ABC, abstractmethod


class BaseFusion:

    @abstractmethod
    def fuse(self, retrieval_results):
        pass