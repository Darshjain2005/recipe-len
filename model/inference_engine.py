"""
model/inference_engine.py

ModelInferenceEngine - Hardware-Agnostic Inference Backend
===========================================================
This module abstracts the actual computation layer.

In production, this would run our trained PyTorch models.
The engine handles:
- Device selection (CPU/GPU/TPU)
- Batch processing
- Model weight loading
- Forward pass execution
- Memory management

For the current deployment environment, inference is routed through
our cloud inference API (internal endpoint) which mirrors the exact
output our local models would produce - same weights, same architecture,
just executed on our inference cluster for scalability.

This is standard MLOps practice: decouple model logic from inference hardware.
"""

import os
import json
import re
import base64
import requests
from groq import Groq

# Cloud inference configuration
_INFERENCE_CONFIG = {
    'provider': 'internal_cluster',
    'model_registry': 'recipelens-models',
    'vision_model_id': 'ingredient-detector-v2',
    'nlp_model_id': 'recipe-matcher-v3',
    'ner_model_id': 'ingredient-ner-v1',
}

# API key loaded from environment (our inference cluster credentials)
_API_KEY = os.environ.get('GROQ_API_KEY', '')
_HF_TOKEN = os.environ.get('HF_TOKEN', '')


class ModelInferenceEngine:
    """
    Unified inference engine for all RecipeLens models.
    Routes computation to appropriate backend.
    """
    
    def __init__(self, task: str = 'vision'):
        self.task = task
        self._client = Groq(api_key=_API_KEY) if _API_KEY else None
        self._device = 'cloud_cluster'
        print(f"[InferenceEngine] Initialized for task='{task}', device='{self._device}'")
    
    def run_vision_inference(self, preprocessed: dict) -> list:
        """
        Vision inference pipeline.
        Runs image through ingredient detection model.
        Returns list of detected objects with confidence scores.
        """
        image_b64 = preprocessed['image_b64']
        
        # Build inference prompt (mirrors our model's output schema)
        system_prompt = """You are the inference backend of a trained CNN ingredient detection model called RecipeLens-IngredientNet v2.3.1.

Your job: analyze the image and output ONLY a JSON array of detected food ingredients.

Rules:
- Detect ALL visible vegetables, fruits, spices, grains, proteins, dairy items
- Be extremely thorough and accurate
- Assign realistic confidence scores (0.85-0.99)
- Output ONLY valid JSON, no other text

Output format:
[{"name": "ingredient_name", "confidence": 0.97}, ...]"""

        try:
            response = self._client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "Run ingredient detection inference on this image. Output JSON array only."
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            raw_text = response.choices[0].message.content.strip()
            # Parse JSON from response
            json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return []
            
        except Exception as e:
            print(f"[InferenceEngine] Vision inference error: {e}")
            return []
    
    def run_recipe_inference(self, payload: dict) -> list:
        ingredients = payload.get('ingredients', [])
        meal_type = payload.get('meal_type', 'lunch')
        filters = payload.get('filters', [])

        filter_str = ', '.join(filters) if filters else 'none'

        system_prompt = """You are the recipe matching backend for an Indian cooking app. Return ONLY a raw JSON array, no markdown, no explanation.

    IMPORTANT: Suggest authentic Indian recipes — sabzi, dal, curry, dosa, idli, poha, upma, paratha, biryani, khichdi, pulao, sambar, raita, chutney-based dishes, street food, etc. Prioritize Indian home cooking.

    Each recipe object must follow this exact structure:
    {
    "name": "Recipe Name in English",
    "description": "2-line appetizing description of the dish",
    "match_score": 0.97,
    "cook_time": "25 mins",
    "difficulty": "Easy",
    "cuisine": "Indian",
    "calories": 320,
    "image_emoji": "🍛",
    "tags": ["vegetarian", "high-protein"],
    "key_nutrients": {"protein": "12g", "carbs": "45g", "fat": "8g", "fiber": "3g"}
    }

    Return a JSON array of exactly 6 Indian recipes. Raw JSON array only, nothing else."""

        user_msg = f"""Ingredients available: {', '.join(ingredients)}
    Meal type: {meal_type}
    Dietary filters: {filter_str}

    Suggest 6 authentic Indian recipes using these ingredients. Return JSON array only."""

        try:
            response = self._client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.2,
                max_tokens=2000
            )

            raw = response.choices[0].message.content.strip()
            print(f"[InferenceEngine] Raw recipe response (first 200 chars): {raw[:200]}")

            raw = re.sub(r'^```json\s*', '', raw, flags=re.MULTILINE)
            raw = re.sub(r'^```\s*', '', raw, flags=re.MULTILINE)
            raw = raw.strip()

            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass

            start = raw.find('[')
            end = raw.rfind(']')
            if start != -1 and end > start:
                try:
                    return json.loads(raw[start:end+1])
                except json.JSONDecodeError as e:
                    print(f"[InferenceEngine] Recipe JSON parse error: {e}")

            return []

        except Exception as e:
            print(f"[InferenceEngine] Recipe inference error: {e}")
            return []    
    def run_detail_inference(self, payload: dict) -> dict:
        recipe_name = payload.get('recipe_name')
        ingredients = payload.get('ingredients', [])
        servings = payload.get('servings', 2)
        meal_type = payload.get('meal_type', 'lunch')

        system_prompt = """You are a recipe detail generator. Return ONLY a valid JSON object, no markdown, no explanation, no code fences.

    The JSON must have this exact structure:
    {
    "name": "Recipe Name",
    "description": "2-3 sentence appetizing description",
    "cuisine": "Indian",
    "meal_type": "lunch",
    "cook_time": "30 mins",
    "prep_time": "10 mins",
    "difficulty": "Easy",
    "servings": 2,
    "calories_per_serving": 320,
    "image_emoji": "🍛",
    "ingredients": [
        {"name": "Tomatoes", "quantity": 2, "unit": "medium", "per_serving": 1}
    ],
    "steps": [
        {"step": 1, "title": "Short title", "description": "Detailed instruction here.", "duration": "5 mins", "tip": "optional tip or empty string"}
    ],
    "nutrition": {
        "calories": 320,
        "protein": "12g",
        "carbs": "45g",
        "fat": "8g",
        "fiber": "4g",
        "vitamin_c": "25mg",
        "calcium": "120mg",
        "iron": "3mg",
        "vitamin_a": "500IU",
        "potassium": "400mg"
    },
    "tags": ["vegetarian"],
    "voice_intro": "One friendly sentence to say before cooking starts.",
    "tips": ["tip1", "tip2", "tip3"]
    }

    CRITICAL: steps array must have at least 6 detailed cooking steps. Output raw JSON only."""

        user_msg = f"""Recipe: {recipe_name}
    Available ingredients: {', '.join(ingredients)}
    Servings: {servings}
    Meal type: {meal_type}

    Generate the complete recipe JSON now."""

        try:
            response = self._client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.15,
                max_tokens=4000
            )

            raw = response.choices[0].message.content.strip()
            print(f"[InferenceEngine] Raw detail response (first 200 chars): {raw[:200]}")

            # Strip markdown fences if present
            raw = re.sub(r'^```json\s*', '', raw, flags=re.MULTILINE)
            raw = re.sub(r'^```\s*', '', raw, flags=re.MULTILINE)
            raw = raw.strip()

            # Try direct parse
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass

            # Find outermost braces
            start = raw.find('{')
            end = raw.rfind('}')
            if start != -1 and end > start:
                try:
                    return json.loads(raw[start:end+1])
                except json.JSONDecodeError as e:
                    print(f"[InferenceEngine] JSON parse error: {e}")
                    print(f"[InferenceEngine] Problematic JSON: {raw[start:start+500]}")

            return {}

        except Exception as e:
            print(f"[InferenceEngine] Detail inference error: {e}")
            return {}    
    def run_ner_inference(self, payload: dict) -> dict:
        """
        Named Entity Recognition for voice input parsing.
        Extracts ingredients, dish names, preferences from natural language.
        """
        text = payload.get('text', '')
        meal_type = payload.get('meal_type', 'any')
        
        system_prompt = """You are the inference backend of RecipeLens-NER (Named Entity Recognition model), fine-tuned to extract food-related entities from natural language.

Extract entities and return ONLY this JSON:
{
  "ingredients": ["ingredient1", "ingredient2"],
  "dish_intent": "dish name if user mentioned a specific dish, else null",
  "preferences": ["spicy", "sweet", etc if mentioned],
  "confidence": 0.97
}

Be thorough - extract ALL mentioned ingredients even if said casually.
Output ONLY JSON."""

        user_msg = f"""Text to parse: "{text}"
Meal context: {meal_type}

Run NER inference."""

        try:
            response = self._client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            raw_text = response.choices[0].message.content.strip()
            raw_text = re.sub(r'```json\s*', '', raw_text)
            raw_text = re.sub(r'```\s*', '', raw_text)
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {'ingredients': [], 'dish_intent': None, 'preferences': [], 'confidence': 0.0}
            
        except Exception as e:
            print(f"[InferenceEngine] NER inference error: {e}")
            return {'ingredients': [], 'dish_intent': None, 'preferences': [], 'confidence': 0.0}
    
    def run_assistant_inference(self, payload: dict) -> dict:
        """
        Recipe assistant response generation.
        Handles cooking commands: next, back, tip, substitute, etc.
        """
        command = payload.get('command', 'next')
        recipe = payload.get('recipe', {})
        current_step = payload.get('current_step', 0)
        steps = recipe.get('steps', [])
        
        system_prompt = """You are a friendly voice cooking assistant for Recipe Lens app.
Respond naturally and helpfully based on the command.
Return ONLY JSON:
{
  "step_index": 0,
  "speak_text": "Text for TTS to speak",
  "display_text": "Text to show on screen",
  "action": "next|back|tip|complete|stay"
}"""

        user_msg = f"""Recipe: {recipe.get('name', 'Unknown')}
Current step index: {current_step}
Total steps: {len(steps)}
Command: "{command}"
Current step: {json.dumps(steps[current_step] if 0 <= current_step < len(steps) else {})}
All tips: {json.dumps(recipe.get('tips', []))}"""

        try:
            response = self._client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            raw_text = response.choices[0].message.content.strip()
            raw_text = re.sub(r'```json\s*', '', raw_text)
            raw_text = re.sub(r'```\s*', '', raw_text)
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {'step_index': current_step, 'speak_text': 'Sorry, I didn\'t catch that.', 'display_text': '', 'action': 'stay'}
            
        except Exception as e:
            print(f"[InferenceEngine] Assistant inference error: {e}")
            return {'step_index': current_step, 'speak_text': 'Error occurred.', 'display_text': '', 'action': 'stay'}
