import sys
import os

# Add OmniCore to sys.path
sys.path.insert(0, r"e:\OmniCore")

from core.app_state import app_state

def test_inference():
    taiji = app_state.get_taiji_engine()
    if not taiji:
        print("No taiji engine. We need to load it first.")
        
        # Load taiji
        from taiji.loader import load_taiji_model
        
        # Where is the model? Let's find it.
        # It's probably in user_data or taiji_checkpoints
        # Or let's just see if app_state is initialized. 
        # Actually app_state is empty if we just import it.
        
    print("taiji:", taiji)

test_inference()
