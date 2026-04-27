"""
RecipeLens Model Package
========================
Custom-trained ML models for ingredient detection, recipe matching,
and voice-based NLP parsing.

Models:
- IngredientDetector: CNN vision model for ingredient recognition
- RecipeEngine: Transformer-based recipe matching and generation
- VoiceParser: BERT NER model for voice input parsing
- NutrientDatabase: Compiled nutritional data store
"""

from .ingredient_model import IngredientDetector
from .recipe_engine import RecipeEngine
from .voice_parser import VoiceParser
from .nutrient_db import NutrientDatabase

__all__ = ['IngredientDetector', 'RecipeEngine', 'VoiceParser', 'NutrientDatabase']
__version__ = '2.3.1'
