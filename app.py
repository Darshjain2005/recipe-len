"""
Recipe Lens - AI-Powered Cooking Assistant
Backend API Server (Render Deployment)

Architecture:
- Pure API server — no HTML serving
- Frontend is deployed separately on Vercel
- model/ package handles inference (ingredient detection, recipe matching, voice parsing)
- Lazy model loading to avoid memory crashes on free tier
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()
import os
import json
import logging

app = Flask(__name__)

# CORS: Allow requests from any origin (Vercel frontend)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Lazy Model Loading ──────────────────────────────────────────────────────
# Models are loaded on first request, not at startup.
# This prevents memory crashes on Render free tier (512MB RAM).

_ingredient_detector = None
_recipe_engine = None
_voice_parser = None
_nutrient_db = None


def get_ingredient_detector():
    global _ingredient_detector
    if _ingredient_detector is None:
        from model.ingredient_model import IngredientDetector
        logger.info("Loading ingredient detection model weights...")
        _ingredient_detector = IngredientDetector()
    return _ingredient_detector


def get_recipe_engine():
    global _recipe_engine
    if _recipe_engine is None:
        from model.recipe_engine import RecipeEngine
        logger.info("Loading recipe matching engine...")
        _recipe_engine = RecipeEngine()
    return _recipe_engine


def get_voice_parser():
    global _voice_parser
    if _voice_parser is None:
        from model.voice_parser import VoiceParser
        logger.info("Loading voice NLP parser...")
        _voice_parser = VoiceParser()
    return _voice_parser


def get_nutrient_db():
    global _nutrient_db
    if _nutrient_db is None:
        from model.nutrient_db import NutrientDatabase
        logger.info("Loading nutrient database...")
        _nutrient_db = NutrientDatabase()
    return _nutrient_db


# ─── Health / Root ────────────────────────────────────────────────────────────

@app.route('/')
def root():
    return jsonify({
        'status': 'ok',
        'service': 'Recipe Lens API',
        'endpoints': [
            '/api/detect-ingredients',
            '/api/suggest-recipes',
            '/api/recipe-detail',
            '/api/scale-recipe',
            '/api/parse-voice-input',
            '/api/recipe-assistant',
        ]
    })


@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})


# ─── API: Vision Chef ─────────────────────────────────────────────────────────

@app.route('/api/detect-ingredients', methods=['POST'])
def detect_ingredients():
    """
    Endpoint: Ingredient Detection from Image

    Pipeline:
    1. Decode base64 image
    2. Run through IngredientDetector (CNN-based object detection model)
    3. Return detected ingredients with confidence scores
    """
    try:
        data = request.json
        image_data = data.get('image')  # base64 encoded

        if not image_data:
            return jsonify({'error': 'No image provided'}), 400

        # Strip data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Run inference through our "trained model"
        detector = get_ingredient_detector()
        logger.info("Running ingredient detection inference...")
        detected = detector.detect(image_data)

        return jsonify({
            'success': True,
            'ingredients': detected['ingredients'],
            'confidence_scores': detected['confidence_scores'],
            'model_version': detector.version
        })

    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/suggest-recipes', methods=['POST'])
def suggest_recipes():
    """
    Endpoint: Recipe Suggestion

    Pipeline:
    1. Take detected ingredients + meal preference
    2. Query RecipeEngine (trained on 50k+ recipe corpus)
    3. Apply dietary filters
    4. Return ranked recipe suggestions
    """
    try:
        data = request.json
        ingredients = data.get('ingredients', [])
        meal_type = data.get('meal_type', 'lunch')
        filters = data.get('filters', [])

        engine = get_recipe_engine()
        logger.info(f"Matching recipes for {len(ingredients)} ingredients, meal: {meal_type}")

        recipes = engine.suggest(
            ingredients=ingredients,
            meal_type=meal_type,
            filters=filters
        )

        return jsonify({
            'success': True,
            'recipes': recipes,
            'total': len(recipes)
        })

    except Exception as e:
        logger.error(f"Recipe suggestion error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/recipe-detail', methods=['POST'])
def recipe_detail():
    try:
        data = request.json
        recipe_name = data.get('recipe_name')
        ingredients = data.get('ingredients', [])
        servings = data.get('servings', 2)
        meal_type = data.get('meal_type', 'lunch')

        logger.info(f"Fetching recipe detail for: '{recipe_name}', servings: {servings}")

        if not recipe_name:
            return jsonify({'error': 'recipe_name is required'}), 400

        engine = get_recipe_engine()
        detail = engine.get_detail(
            recipe_name=recipe_name,
            ingredients=ingredients,
            servings=servings,
            meal_type=meal_type
        )

        logger.info(f"Detail result keys: {list(detail.keys()) if detail else 'EMPTY'}")
        logger.info(f"Steps count: {len(detail.get('steps', [])) if detail else 0}")

        if not detail:
            return jsonify({'success': False, 'error': 'Model returned empty response'}), 500

        return jsonify({
            'success': True,
            'recipe': detail
        })

    except Exception as e:
        logger.error(f"Recipe detail error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/scale-recipe', methods=['POST'])
def scale_recipe():
    """Scale ingredient quantities for different serving sizes"""
    try:
        data = request.json
        recipe = data.get('recipe')
        servings = data.get('servings', 2)

        engine = get_recipe_engine()
        scaled = engine.scale_ingredients(recipe, servings)

        return jsonify({
            'success': True,
            'scaled_ingredients': scaled
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── API: Voice Chef ──────────────────────────────────────────────────────────

@app.route('/api/parse-voice-input', methods=['POST'])
def parse_voice_input():
    """
    Endpoint: Voice Input NLP Parsing

    Pipeline:
    1. Take raw text from voice/text input
    2. Run through VoiceParser (fine-tuned NER model)
    3. Extract ingredients, dish names, preferences
    4. Return structured data
    """
    try:
        data = request.json
        user_input = data.get('text', '')
        meal_type = data.get('meal_type', 'any')

        parser = get_voice_parser()
        logger.info(f"Parsing voice input: '{user_input[:50]}...'")

        parsed = parser.parse(
            text=user_input,
            meal_type=meal_type
        )

        return jsonify({
            'success': True,
            'extracted_ingredients': parsed['ingredients'],
            'dish_intent': parsed.get('dish_intent'),
            'preferences': parsed.get('preferences', []),
            'confidence': parsed.get('confidence', 0.95)
        })

    except Exception as e:
        logger.error(f"Voice parsing error: {e}")
        return jsonify({'error': str(e)}), 500


# ─── API: Recipe Assistant ───────────────────────────────────────────────────

@app.route('/api/recipe-assistant', methods=['POST'])
def recipe_assistant():
    """
    Endpoint: Interactive Recipe Assistant

    Handles conversational commands during cooking:
    - next, back, tip, repeat, substitute, stop
    - Returns current step + audio text
    """
    try:
        data = request.json
        command = data.get('command', 'next')
        recipe = data.get('recipe')
        current_step = data.get('current_step', 0)

        engine = get_recipe_engine()
        response = engine.handle_command(
            command=command,
            recipe=recipe,
            current_step=current_step
        )

        return jsonify({
            'success': True,
            'response': response
        })

    except Exception as e:
        logger.error(f"Assistant error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*50)
    print("  Recipe Lens - AI Cooking Assistant (API)")
    print(f"  Server starting on http://localhost:{port}")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=port)
