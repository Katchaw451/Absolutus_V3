from .base import Engine, EngineResult
from .statevector import StatevectorEngine
from .mps import MPSEngine

ENGINES = {
    "statevector": StatevectorEngine,
    "mps": MPSEngine,
}
