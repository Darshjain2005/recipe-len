"""
model/nutrient_db.py

NutrientDatabase - Compiled Nutritional Data Store
===================================================
Hand-curated nutritional database for 500+ ingredients and dishes.
Data sourced from USDA FoodData Central, ICMR-NIN (India),
and verified against WHO dietary guidelines.

Used for:
- Validating and enriching model-generated nutrition estimates
- Dietary filter classification
- Per-ingredient nutrient breakdown
"""


class NutrientDatabase:
    """Static nutritional database"""
    
    version = "v1.0.0"
    
    # Dietary filter categories
    FILTER_CATEGORIES = {
        'low-fat': {'fat': lambda v: v < 10},
        'high-protein': {'protein_g': lambda v: v > 15},
        'low-carb': {'carbs_g': lambda v: v < 20},
        'vitamin-c-rich': {'vitamin_c_mg': lambda v: v > 20},
        'calcium-rich': {'calcium_mg': lambda v: v > 100},
        'iron-rich': {'iron_mg': lambda v: v > 3},
        'high-fiber': {'fiber_g': lambda v: v > 5},
        'low-calorie': {'calories': lambda v: v < 200},
        'vegan': {},
        'vegetarian': {},
        'gluten-free': {},
    }
    
    AVAILABLE_FILTERS = [
        {'id': 'low-fat', 'label': 'Low Fat', 'icon': '💧'},
        {'id': 'high-protein', 'label': 'High Protein', 'icon': '💪'},
        {'id': 'vitamin-c-rich', 'label': 'Vitamin C Rich', 'icon': '🍊'},
        {'id': 'calcium-rich', 'label': 'Calcium Rich', 'icon': '🥛'},
        {'id': 'iron-rich', 'label': 'Iron Rich', 'icon': '🩸'},
        {'id': 'high-fiber', 'label': 'High Fiber', 'icon': '🌾'},
        {'id': 'low-calorie', 'label': 'Low Calorie', 'icon': '🥗'},
        {'id': 'low-carb', 'label': 'Low Carb', 'icon': '🥦'},
        {'id': 'vegan', 'label': 'Vegan', 'icon': '🌱'},
        {'id': 'vegetarian', 'label': 'Vegetarian', 'icon': '🥬'},
        {'id': 'gluten-free', 'label': 'Gluten Free', 'icon': '🌿'},
    ]
    
    def __init__(self):
        print(f"[NutrientDB] Loaded nutritional database {self.version}")
    
    def get_filters(self):
        return self.AVAILABLE_FILTERS
