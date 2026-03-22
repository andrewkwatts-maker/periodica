"""
Predictor Registry Module

Provides a singleton registry for managing predictor classes across different domains.
Supports registration via decorator or direct method calls.
"""

from typing import Dict, Type, Callable, Optional, List, Any


class PredictorRegistry:
    """
    Singleton registry for predictor classes.

    Manages predictor classes organized by domain and name, allowing for
    easy registration, retrieval, and instantiation of predictors.

    Example:
        @register_predictor('nuclear', 'semf')
        class SEMFPredictor:
            pass

        registry = PredictorRegistry()
        predictor = registry.get('nuclear', 'semf')
    """

    _instance: Optional['PredictorRegistry'] = None
    _predictors: Dict[str, Dict[str, Type]] = {}
    _factories: Dict[str, Callable] = {}

    def __new__(cls) -> 'PredictorRegistry':
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._predictors = {}
            cls._instance._factories = {}
        return cls._instance

    def register(self, domain: str, name: str, predictor_class: Type) -> None:
        """
        Register a predictor class under a specific domain and name.

        Args:
            domain: The domain category (e.g., 'nuclear', 'atomic', 'molecular')
            name: The specific predictor name within the domain
            predictor_class: The predictor class to register
        """
        if domain not in self._predictors:
            self._predictors[domain] = {}
        self._predictors[domain][name] = predictor_class

    def register_factory(self, key: str, factory: Callable) -> None:
        """
        Register a factory function for creating predictors.

        Args:
            key: Unique identifier for the factory
            factory: Callable that returns a predictor instance
        """
        self._factories[key] = factory

    def get(self, domain: str, name: str = 'default') -> Any:
        """
        Get an instance of a registered predictor.

        Args:
            domain: The domain category to look up
            name: The predictor name within the domain (defaults to 'default')

        Returns:
            An instance of the requested predictor class

        Raises:
            KeyError: If the domain or predictor name is not registered
        """
        if domain not in self._predictors:
            raise KeyError(f"Domain '{domain}' not found in registry")

        if name not in self._predictors[domain]:
            raise KeyError(f"Predictor '{name}' not found in domain '{domain}'")

        predictor_class = self._predictors[domain][name]
        return predictor_class()

    def get_class(self, domain: str, name: str = 'default') -> Type:
        """
        Get a registered predictor class without instantiating it.

        Args:
            domain: The domain category to look up
            name: The predictor name within the domain (defaults to 'default')

        Returns:
            The predictor class

        Raises:
            KeyError: If the domain or predictor name is not registered
        """
        if domain not in self._predictors:
            raise KeyError(f"Domain '{domain}' not found in registry")

        if name not in self._predictors[domain]:
            raise KeyError(f"Predictor '{name}' not found in domain '{domain}'")

        return self._predictors[domain][name]

    def list_predictors(self, domain: str = None) -> Dict[str, List[str]]:
        """
        List all registered predictors, optionally filtered by domain.

        Args:
            domain: If provided, only list predictors for this domain.
                   If None, list predictors for all domains.

        Returns:
            Dictionary mapping domain names to lists of predictor names.
            If domain is specified, returns only that domain's predictors.

        Raises:
            KeyError: If the specified domain is not found
        """
        if domain is not None:
            if domain not in self._predictors:
                raise KeyError(f"Domain '{domain}' not found in registry")
            return {domain: list(self._predictors[domain].keys())}

        return {d: list(names.keys()) for d, names in self._predictors.items()}

    def list_factories(self) -> List[str]:
        """
        List all registered factory keys.

        Returns:
            List of factory keys
        """
        return list(self._factories.keys())

    def has_predictor(self, domain: str, name: str = 'default') -> bool:
        """
        Check if a predictor is registered.

        Args:
            domain: The domain category to check
            name: The predictor name within the domain

        Returns:
            True if the predictor is registered, False otherwise
        """
        return domain in self._predictors and name in self._predictors[domain]

    def clear(self) -> None:
        """Clear all registered predictors and factories."""
        self._predictors.clear()
        self._factories.clear()


def register_predictor(domain: str, name: str = 'default') -> Callable[[Type], Type]:
    """
    Decorator to register a predictor class with the registry.

    Args:
        domain: The domain category for the predictor
        name: The predictor name within the domain (defaults to 'default')

    Returns:
        Decorator function that registers the class and returns it unchanged

    Example:
        @register_predictor('nuclear', 'semf')
        class SEMFPredictor:
            def predict(self, data):
                ...
    """
    def decorator(cls: Type) -> Type:
        registry = PredictorRegistry()
        registry.register(domain, name, cls)
        return cls
    return decorator
