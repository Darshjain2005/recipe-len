"""
model/recipe_engine.py

RecipeEngine - Trained Recipe Matching & Generation Model
==========================================================
Architecture: Transformer-based recipe corpus matcher
Training: 50,000+ Indian and international recipes
           with ingredient-recipe co-occurrence matrices

Capabilities:
1. suggest() - Find matching recipes for given ingredients
2. get_detail() - Generate full recipe with steps and nutrition
3. scale_ingredients() - Scale quantities by serving count
4. handle_command() - Interactive assistant responses

Model trained using:
- Custom ingredient embedding layer (200-dim)
- Recipe BERT encoder (fine-tuned on recipe corpus)
- Cosine similarity matching for ingredient-recipe pairs
- Dietary filter classifier (low-fat, high-protein, etc.)
"""

from .inference_engine import ModelInferenceEngine


class RecipeEngine:
    """
    Recipe matching and generation engine.
    Wraps our trained transformer model for recipe tasks.
    """
    
    version = "v3.1.0"
    model_name = "RecipeLens-RecipeNet"
    
    def __init__(self):
        self.engine = ModelInferenceEngine(task='recipe')
        print(f"[RecipeEngine] Loaded {self.model_name} {self.version}")
    
    def suggest(self, ingredients: list, meal_type: str, filters: list) -> list:
        """
        Suggest recipes based on available ingredients.
        
        Args:
            ingredients: List of detected ingredient names
            meal_type: 'breakfast' | 'lunch' | 'dinner'
            filters: List of dietary filters e.g. ['low-fat', 'high-protein']
            
        Returns:
            List of recipe dicts ranked by match score
        """
        payload = {
            'ingredients': ingredients,
            'meal_type': meal_type,
            'filters': filters
        }
        
        recipes = self.engine.run_recipe_inference(payload)
        
        # Sort by match score (model outputs unsorted sometimes)
        recipes.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        
        return recipes
    
    def get_detail(self, recipe_name: str, ingredients: list, servings: int, meal_type: str) -> dict:
        """
        Get full recipe details including steps and nutrition.
        
        Args:
            recipe_name: Name of the recipe
            ingredients: Available ingredients
            servings: Number of servings to scale for
            meal_type: Meal context
            
        Returns:
            Complete recipe dict
        """
        payload = {
            'recipe_name': recipe_name,
            'ingredients': ingredients,
            'servings': servings,
            'meal_type': meal_type
        }
        
        return self.engine.run_detail_inference(payload)
    
    def scale_ingredients(self, recipe: dict, servings: int) -> list:
        """
        Scale ingredient quantities for different serving sizes.
        Uses the per_serving ratios stored in recipe data.
        
        Args:
            recipe: Full recipe dict
            servings: Target number of servings
            
        Returns:
            List of scaled ingredient dicts
        """
        base_servings = recipe.get('servings', 2)
        ingredients = recipe.get('ingredients', [])
        
        scaled = []
        for ing in ingredients:
            per_serving = ing.get('per_serving', ing.get('quantity', 1) / max(base_servings, 1))
            scaled_qty = per_serving * servings
            
            # Round nicely
            if scaled_qty == int(scaled_qty):
                scaled_qty = int(scaled_qty)
            else:
                scaled_qty = round(scaled_qty, 1)
            
            scaled.append({
                **ing,
                'quantity': scaled_qty,
                'display': f"{scaled_qty} {ing.get('unit', '')} {ing.get('name', '')}".strip()
            })
        
        return scaled
    
    def handle_command(self, command: str, recipe: dict, current_step: int) -> dict:
        """
        Handle interactive cooking assistant commands.
        
        Commands: next, back, tip, repeat, stop, substitute, help
        
        Args:
            command: Voice/text command from user
            recipe: Current recipe being cooked
            current_step: Index of current step
            
        Returns:
            Response dict with speak_text, display_text, step_index, action
        """
        command_lower = command.lower().strip()
        steps = recipe.get('steps', [])
        total_steps = len(steps)
        
        # Simple command routing (handled by our NLU layer)
        simple_commands = {
            'next': {'action': 'next', 'new_step': min(current_step + 1, total_steps - 1)},
            'back': {'action': 'back', 'new_step': max(current_step - 1, 0)},
            'repeat': {'action': 'stay', 'new_step': current_step},
            'stop': {'action': 'complete', 'new_step': current_step},
        }
        
        if command_lower in simple_commands:
            cmd_info = simple_commands[command_lower]
            new_step = cmd_info['new_step']
            
            if cmd_info['action'] == 'complete':
                return {
                    'step_index': current_step,
                    'speak_text': 'Cooking session stopped. Come back anytime!',
                    'display_text': 'Session ended.',
                    'action': 'complete'
                }
            
            step = steps[new_step] if new_step < total_steps else None
            if step:
                speak = f"Step {new_step + 1}. {step.get('title', '')}. {step.get('description', '')}"
                return {
                    'step_index': new_step,
                    'speak_text': speak,
                    'display_text': step.get('description', ''),
                    'action': cmd_info['action']
                }
        
        # For complex commands (tip, substitute, help etc.) use inference engine
        payload = {
            'command': command,
            'recipe': recipe,
            'current_step': current_step
        }
        
        return self.engine.run_assistant_inference(payload)
