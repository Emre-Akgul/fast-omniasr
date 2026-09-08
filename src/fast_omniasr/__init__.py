"""Standalone OmniASR runtime. Conversion dependencies are deliberately separate."""
from .model import OmniASR, Transcription

__all__ = ["OmniASR", "Transcription"]
