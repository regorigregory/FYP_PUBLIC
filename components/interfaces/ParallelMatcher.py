import abc
class ParallelMatcher(abc.ABC):

    @abc.abstractmethod
    def alignImagesParallel(self):
        pass

    @abc.abstractmethod
    def recompileObject(self, generator):
        pass