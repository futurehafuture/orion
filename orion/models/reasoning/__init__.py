"""推理模块"""

from .llm_interface import LLMInterface, ToyLLM, HuggingFaceLLM
from .vqa_head import VQAHead

__all__ = [
    "LLMInterface",
    "ToyLLM", 
    "HuggingFaceLLM",
    "VQAHead",
]
