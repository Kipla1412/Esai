import os
import logging
import pickle
import warnings

from .base import Serialize

logger = logging.getLogger(__name__)

class Pickle(Serialize):
    def __init__(self, allowpickle=False):
        super().__init__()
        self.allowpickle = allowpickle
        self.version = 4

    def load(self, path):
        return super().load(path) if self.allow(path) else None

    def save(self, data, path):
        if self.allow():
            super().save(data, path)

    def loadstream(self, stream):
        return pickle.load(stream) if self.allow() else None

    def savestream(self, data, stream):
        if self.allow():
            pickle.dump(data, stream, protocol=self.version)

    def loadbytes(self, data):
        return pickle.loads(data) if self.allow() else None

    def savebytes(self, data):
        return pickle.dumps(data, protocol=self.version) if self.allow() else None

    def allow(self, path=None):
        enablepickle = self.allowpickle or os.environ.get("ALLOW_PICKLE", "True") in ("True", "1")
        if not enablepickle:
            raise ValueError(
                (
                    "Loading of pickled index data is disabled. "
                    f"`{path if path else 'stream'}` was not loaded. "
                    "Set the env variable `ALLOW_PICKLE=True` to enable loading pickled index data. "
                    "This should only be done for trusted and/or local data."
                )
            )

        if not self.allowpickle:
            warnings.warn(
                (
                    "Pickled index data formats are deprecated and loading will be disabled by default in the future. "
                    "Set the env variable `ALLOW_PICKLE=False` to disable the loading of pickled index data formats. "
                    "Saving this index will replace pickled index data formats with the latest index formats and remove this warning."
                ),
                FutureWarning,
            )

        return enablepickle
