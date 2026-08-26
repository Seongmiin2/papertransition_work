from .fixture import FixtureOcrProvider

__all__ = ["FixtureOcrProvider", "PaddlePdfOcrProvider"]


def __getattr__(name: str) -> object:
    if name == "PaddlePdfOcrProvider":
        from .paddle import PaddlePdfOcrProvider

        return PaddlePdfOcrProvider
    raise AttributeError(name)
