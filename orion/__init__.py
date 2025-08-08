"""
ORION: A Holistic End-to-End Autonomous Driving Framework
"""

__version__ = "0.1.0"
__author__ = "ORION Team"

from .models import OrionSystem
from .config import OrionConfig
from .training import OrionTrainer

__all__ = [
    "OrionSystem",
    "OrionConfig", 
    "OrionTrainer",
    "__version__",
    "__author__",
]
