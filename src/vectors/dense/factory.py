
"""
Factory module 
"""

from ...util import Resolver
from .sbert import STVectors


class VectorsFactory:
    """
    Factory for creating Sentence Transformers vector models.
    """

    @staticmethod
    def create(config, scoring=None, models=None):
        """
        Create a Sentence Transformers vectors model.

        Args:
            config: vector configuration (must include "path")
            scoring: scoring instance (optional)
            models: models cache (optional)

        Returns:
            STVectors instance or None
        """
        config = config or {}
        method = VectorsFactory.method(config)

        if method == "sentence-transformers":
            return STVectors(config, scoring, models) if config.get("path") else None

        return None  # No other vector types supported

    @staticmethod
    def method(config):
        """
        Get or derive the vector method.
        Defaults to 'sentence-transformers' if path is given.

        Args:
            config: vector configuration

        Returns:
            'sentence-transformers' or None
        """
        config = config or {}
        method = config.get("method")
        path = config.get("path")

        if not method and path:
            method = "sentence-transformers"

        return method
