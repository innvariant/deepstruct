from abc import ABC


class PruningStrategy(ABC):
    def __init__(self, fn_prune):
        pass

    def run(self):
        pass


class AbsoluteThresholdStrategy(PruningStrategy):
    def run(self):
        pass


class RelativeThresholdStrategy(PruningStrategy):
    def run(self):
        pass


class BucketFillStrategy(PruningStrategy):
    def run(self):
        pass
