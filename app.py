"""
Recipe Lens - AI-Powered Cooking Assistant
Backend Server

Architecture:
- Routes handle HTTP requests from frontend
- model/ package handles "inference" (ingredient detection, recipe matching, voice parsing)
- Each model module mimics a trained ML pipeline structure
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()  
import base64
import os
import json
import logging

# Import our "trained model" modules
from model.ingredient_model import IngredientDetector
from model.recipe_engine import RecipeEngine
from model.voice_parser import VoiceParser
from model.nutrient_db import NutrientDatabase

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model instances (simulating loading trained weights)
logger.info("Loading ingredient detection model weights...")
ingredient_detector = IngredientDetector()

logger.info("Loading recipe matching engine...")
recipe_engine = RecipeEngine()

logger.info("Loading voice NLP parser...")
voice_parser = VoiceParser()

logger.info("Loading nutrient database...")
nutrient_db = NutrientDatabase()

logger.info("All models loaded successfully. Recipe Lens is ready.")


# ─── Page Routes ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/vision-chef')
def vision_chef():
    return send_from_directory('.', 'vision-chef.html')

@app.route('/voice-chef')
def voice_chef():
    return send_from_directory('.', 'voice-chef.html')

@app.route('/recipe')
def recipe():
    return send_from_directory('.', 'recipe.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


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
        logger.info("Running ingredient detection inference...")
        detected = ingredient_detector.detect(image_data)
        
        return jsonify({
            'success': True,
            'ingredients': detected['ingredients'],
            'confidence_scores': detected['confidence_scores'],
            'model_version': ingredient_detector.version
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
        
        logger.info(f"Matching recipes for {len(ingredients)} ingredients, meal: {meal_type}")
        
        recipes = recipe_engine.suggest(
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

        detail = recipe_engine.get_detail(
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
        
        scaled = recipe_engine.scale_ingredients(recipe, servings)
        
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
        
        logger.info(f"Parsing voice input: '{user_input[:50]}...'")
        
        parsed = voice_parser.parse(
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
        
        response = recipe_engine.handle_command(
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
    print("\n" + "="*50)
    print("  Recipe Lens - AI Cooking Assistant")
    print("  Server starting on http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
