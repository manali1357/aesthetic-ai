#!/usr/bin/env python3
"""
Test script to verify skin analysis and recommendations functionality
"""

import json
from skin_analyzer import SkinAnalyzer
from gpt_recommendations import GPTRecommendations
from PIL import Image
import numpy as np

def test_analysis():
    """Test the skin analysis functionality"""
    print("Testing Skin Analysis...")
    
    # Create a test image (simple colored rectangle)
    test_image = Image.new('RGB', (400, 400), color='pink')
    
    # Initialize analyzer
    analyzer = SkinAnalyzer()
    
    # Test analysis
    try:
        result = analyzer.analyze_skin(test_image)
        print("Analysis result type:", type(result))
        print("Analysis success:", result.get("success", False))
        
        if result.get("success") and result.get("analysis"):
            analysis = result["analysis"]
            print("Analysis type:", type(analysis))
            print("Analysis keys:", list(analysis.keys()))
            
            # Test JSON serialization
            try:
                json_str = json.dumps(analysis)
                print("JSON serialization successful")
                print("JSON length:", len(json_str))
            except Exception as e:
                print("JSON serialization failed:", e)
                
        else:
            print("Analysis failed:", result.get("error", "Unknown error"))
            
    except Exception as e:
        print("Analysis exception:", e)
        import traceback
        traceback.print_exc()

def test_recommendations():
    """Test the recommendations functionality"""
    print("\nTesting Recommendations...")
    
    # Create a test analysis
    test_analysis = {
        "overall_score": 0.65,
        "overall_description": "Test analysis",
        "breakouts": {
            "detected": True,
            "severity": 0.6,
            "confidence": 0.8,
            "description": "Test breakouts",
            "specific_observations": "Test observations"
        },
        "pigmentation": {
            "detected": False,
            "severity": 0.2,
            "confidence": 0.7,
            "description": "Test pigmentation",
            "specific_observations": "Test observations"
        },
        "redness": {
            "detected": True,
            "severity": 0.4,
            "confidence": 0.75,
            "description": "Test redness",
            "specific_observations": "Test observations"
        },
        "aging": {
            "detected": False,
            "severity": 0.3,
            "confidence": 0.8,
            "description": "Test aging",
            "specific_observations": "Test observations"
        },
        "dehydration": {
            "detected": True,
            "severity": 0.5,
            "confidence": 0.7,
            "description": "Test dehydration",
            "specific_observations": "Test observations"
        },
        "under_eye": {
            "detected": False,
            "severity": 0.25,
            "confidence": 0.6,
            "description": "Test under_eye",
            "specific_observations": "Test observations"
        }
    }
    
    # Initialize recommendations
    recommendations = GPTRecommendations()
    
    try:
        result = recommendations.get_recommendations(test_analysis)
        print("Recommendations result type:", type(result))
        print("Recommendations success:", result.get("success", False))
        
        if result.get("success"):
            print("Recommendations generated successfully")
            if result.get("recommendations"):
                print("Recommendations type:", type(result["recommendations"]))
                print("Recommendations keys:", list(result["recommendations"].keys()))
        else:
            print("Recommendations failed:", result.get("error", "Unknown error"))
            
    except Exception as e:
        print("Recommendations exception:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_analysis()
    test_recommendations()
    print("\nTest completed!") 