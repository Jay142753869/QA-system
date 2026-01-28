import sys
import os
import torch
from config import Config
from core.regcn_wrapper import REGCNWrapper

def test_integration():
    print("Testing REGCN Integration...")
    
    # Force mock to False to load real model
    Config.USE_MOCK_MODELS = False
    
    try:
        # Initialize Wrapper
        print("Initializing Wrapper...")
        wrapper = REGCNWrapper(Config.__dict__)
        
        # Test Case from CSV: 招商银行, 监事会提名委员会委员, 蔡洪平, 20250101
        head = "招商银行"
        relation = "监事会提名委员会委员"
        time_str = "20250102" # Using a date from CSV
        
        print(f"Predicting: ({head}, {relation}, ?, {time_str})")
        results = wrapper.predict(head, relation, time_str)
        
        print("Top Predictions:")
        for r in results:
            print(f" - {r['name']}: {r['score']:.4f}")
            
        # Check if '蔡洪平' is in results
        found = any(r['name'] == '蔡洪平' for r in results)
        if found:
            print("SUCCESS: Expected tail '蔡洪平' found in predictions.")
        else:
            print("WARNING: Expected tail '蔡洪平' NOT found in top predictions. Model might need tuning or history is insufficient.")
            
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_integration()
