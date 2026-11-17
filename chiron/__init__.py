"""Chiron psychological measurement and behavioural prediction."""

__all__ = ["Chiron", "measurement_messages", "prediction_messages", "population_messages"]


def __getattr__(name):
    if name in __all__:
        from . import inference
        return getattr(inference, name)
    raise AttributeError(name)
