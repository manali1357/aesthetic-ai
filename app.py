import os
import base64
import io
import json
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
from dotenv import load_dotenv
from skin_analyzer import SkinAnalyzer
from body_skin_analyzer import BodySkinAnalyzer
from gpt_recommendations import GPTRecommendations
from treatment_plan_generator import TreatmentPlanGenerator
from logger_utils import logger

# Load environment variables
load_dotenv()

# FastAPI app setup
app = FastAPI(title="Skin Analyzer API", version="1.0.0")

# CORS middleware - configure for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the SkinAnalyzer and GPTRecommendations
skin_analyzer = SkinAnalyzer()
gpt_recommendations = GPTRecommendations()
body_skin_analyzer = BodySkinAnalyzer()
treatment_plan_generator = TreatmentPlanGenerator()

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Skin Analyzer API is running"}

@app.post("/api/analyze-skin")
async def analyze_skin(request: Request):
    try:
        # Extract base64 image from request
        body = await request.json()
        image_base64 = body.get('image', None)
        
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decode the image from base64
        try:
            img_data = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(img_data))
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
        except Exception as e:
            logger.error(f"Image decoding error: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Perform skin analysis
        analysis_result = skin_analyzer.analyze_skin(img)
        
        if analysis_result is None:
            raise HTTPException(status_code=500, detail="Skin analysis failed")
        
        return JSONResponse(content=analysis_result)

    except HTTPException as e:
        logger.error(f"HTTP error in analyze_skin: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in analyze_skin: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during skin analysis")

@app.post("/api/recommendations")
async def generate_recommendations(request: Request):
    try:
        body = await request.json()
        analysis_results = body.get('analysis', None)
        
        if not analysis_results:
            raise HTTPException(status_code=400, detail="No analysis results provided")
        
        # Generate recommendations based on analysis results
        recommendations = gpt_recommendations.get_recommendations(analysis_results)
        
        return JSONResponse(content=recommendations)

    except HTTPException as e:
        logger.error(f"HTTP error in generate_recommendations: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in generate_recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while generating recommendations")

@app.post("/api/ingredients")
async def get_ingredients(request: Request):
    """Get ingredient recommendations based on skin analysis"""
    try:
        body = await request.json()
        analysis_results = body.get('analysis', {})
        
        # Handle nested analysis structure (from /api/analyze-skin response)
        if isinstance(analysis_results, dict) and 'analysis' in analysis_results:
            analysis_results = analysis_results['analysis']
        # Extract detected concerns from analysis
        concerns = []
        for concern_name, concern_data in analysis_results.items():
            if concern_name == "overall_score":
                continue
            if isinstance(concern_data, dict) and concern_data.get("detected", False):
                concerns.append(concern_name)
        
        # If no concerns detected, use common ones
        if not concerns:
            concerns = ["acne", "dryness", "aging", "hyperpigmentation", "redness", "dullness"]
        
        ingredients_data = gpt_recommendations.get_ingredient_recommendations(concerns)
        
        return JSONResponse(content=ingredients_data)
        
    except Exception as e:
        logger.error(f"Error in get_ingredients: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching ingredients")

@app.post("/api/ingredients/by-concern")
async def get_ingredients_by_concern(request: Request):
    """Get ingredient recommendations based on specific skin concerns using LLM"""
    try:
        body = await request.json()
        concerns = body.get('concerns', [])
        
        if not concerns:
            raise HTTPException(status_code=400, detail="No concerns provided")
        
        # Use LLM to generate ingredient recommendations based on specific concerns
        ingredients_data = gpt_recommendations.get_ingredient_recommendations(concerns)
        
        return JSONResponse(content=ingredients_data)
        
    except Exception as e:
        logger.error(f"Error in get_ingredients_by_concern: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating ingredient recommendations")

@app.post("/api/ingredients/from-analysis")
async def get_ingredients_from_analysis(request: Request):
    """Get ingredient recommendations directly from skin analysis results"""
    try:
        body = await request.json()
        analysis_results = body.get('analysis', None)
        
        if not analysis_results:
            raise HTTPException(status_code=400, detail="No analysis results provided")
        
        # Extract detected concerns from analysis results
        concerns = []
        for concern_name, concern_data in analysis_results.items():
            if concern_name != "overall_score" and isinstance(concern_data, dict):
                if concern_data.get("detected", False):
                    concerns.append(concern_name)
        
        if not concerns:
            concerns = ["general skin health", "maintenance"]
        
        # Use LLM to generate ingredient recommendations
        ingredients_data = gpt_recommendations.get_ingredient_recommendations(concerns)
        
        # Add analysis context to the response
        ingredients_data["analysis_context"] = {
            "detected_concerns": concerns,
            "overall_score": analysis_results.get("overall_score", 0.5)
        }
        
        return JSONResponse(content=ingredients_data)
        
    except Exception as e:
        logger.error(f"Error in get_ingredients_from_analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating ingredient recommendations from analysis")

@app.get("/api/ingredient/{ingredient_name}")
async def get_ingredient_detail(ingredient_name: str):
    """Get detailed information about a specific ingredient using LLM"""
    try:
        # Use LLM to get detailed information about a specific ingredient
        prompt = f"""
        Provide detailed scientific information about the skincare ingredient: {ingredient_name}
        
        Please provide information in JSON format with this structure:
        {{
            "name": "{ingredient_name}",
            "scientific_name": "Scientific/Latin name if applicable",
            "category": "Ingredient category",
            "description": "Comprehensive description",
            "mechanism_of_action": "Detailed scientific mechanism",
            "clinical_studies": [
                {{
                    "study_focus": "What was studied",
                    "results": "Key findings",
                    "significance": "Clinical importance"
                }}
            ],
            "benefits": ["list", "of", "benefits"],
            "side_effects": ["potential", "side", "effects"],
            "contraindications": ["situations", "to", "avoid"],
            "optimal_concentration": "Effective percentage range",
            "stability": "Storage and stability information",
            "compatible_ingredients": ["list", "of", "compatible", "ingredients"],
            "incompatible_ingredients": ["list", "of", "incompatible", "ingredients"],
            "product_formulations": ["types", "of", "products", "it's", "used", "in"],
            "research_sources": ["key", "research", "sources"],
            "dermatologist_consensus": "Expert opinion summary"
        }}
        
        Focus on evidence-based information from reliable scientific sources.
        """
        
        if not gpt_recommendations.client:
            return JSONResponse(content={
                "name": ingredient_name,
                "error": "LLM service unavailable",
                "message": "Detailed ingredient information requires LLM access"
            })
        
        response = gpt_recommendations.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are a skincare scientist and dermatology expert. 
                    Provide detailed, evidence-based information about skincare ingredients.
                    Reference clinical studies, scientific literature, and dermatological consensus."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=1200,
            temperature=0.3
        )
        
        ingredient_detail = json.loads(response.choices[0].message.content)
        return JSONResponse(content=ingredient_detail)
        
    except Exception as e:
        logger.error(f"Error in get_ingredient_detail: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching details for ingredient: {ingredient_name}")

@app.post("/api/test-fallback")
async def test_fallback():
    """Test endpoint for fallback mode"""
    try:
        # Simulate analysis results for testing
        test_results = {
            "skin_concerns": {
                "breakouts": {"score": 0.3, "severity": "low"},
                "pigmentation": {"score": 0.6, "severity": "medium"},
                "redness": {"score": 0.4, "severity": "low"},
                "aging": {"score": 0.2, "severity": "low"},
                "dehydration": {"score": 0.7, "severity": "high"}
            },
            "overall_score": 0.44,
            "message": "Fallback analysis completed successfully"
        }
        return JSONResponse(content=test_results)
    except Exception as e:
        logger.error(f"Error in test_fallback: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in fallback test")

@app.post("/api/analyze-body-part")
async def analyze_body_part(request: Request):
    """Analyze skin conditions for body parts (hands, legs, abdomen, etc.)"""
    try:
        # Extract base64 image and body part from request
        body = await request.json()
        image_base64 = body.get('image', None)
        body_part = body.get('body_part', 'hands')  # Default to hands
        
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decode the image from base64
        try:
            img_data = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(img_data))
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
        except Exception as e:
            logger.error(f"Image decoding error: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Perform body part skin analysis
        analysis_result = body_skin_analyzer.analyze_body_part(img, body_part)
        
        if not analysis_result.get('success', False):
            raise HTTPException(
                status_code=500, 
                detail=analysis_result.get('error', 'Body part analysis failed')
            )
        
        return JSONResponse(content=analysis_result)

    except HTTPException as e:
        logger.error(f"HTTP error in analyze_body_part: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in analyze_body_part: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="An unexpected error occurred during body part analysis"
        )

@app.get("/api/supported-body-parts")
async def get_supported_body_parts():
    """Get list of supported body parts for analysis"""
    return JSONResponse(content={
        "supported_parts": list(BodySkinAnalyzer.BODY_PART_CONCERNS.keys()),
        "details": BodySkinAnalyzer.BODY_PART_CONCERNS
    })

@app.post("/api/body-recommendations")
async def generate_body_recommendations(request: Request):
    """Generate recommendations for body part skin concerns"""
    try:
        body = await request.json()
        analysis_results = body.get('analysis', None)
        body_part = body.get('body_part', 'hands')
        
        if not analysis_results:
            raise HTTPException(status_code=400, detail="No analysis results provided")
        
        # Generate body-specific recommendations
        recommendations = gpt_recommendations.get_body_part_recommendations(
            analysis_results, 
            body_part
        )
        
        return JSONResponse(content=recommendations)

    except HTTPException as e:
        logger.error(f"HTTP error in generate_body_recommendations: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in generate_body_recommendations: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="An unexpected error occurred while generating body recommendations"
        )

@app.post("/api/compare-body-parts")
async def compare_body_parts(request: Request):
    """Compare skin analysis across multiple body parts"""
    try:
        body = await request.json()
        analyses = body.get('analyses', {})  # Dictionary of {body_part: analysis}
        
        if not analyses or len(analyses) < 2:
            raise HTTPException(
                status_code=400, 
                detail="Need at least 2 body part analyses to compare"
            )
        
        # Create comparison summary
        comparison = {
            "body_parts_analyzed": list(analyses.keys()),
            "overall_comparison": {},
            "recommendations": []
        }
        
        for part, analysis in analyses.items():
            if 'analysis' in analysis:
                score = analysis['analysis'].get('overall_score', 0.5)
                comparison["overall_comparison"][part] = {
                    "score": score,
                    "status": "Good" if score > 0.7 else "Needs Attention" if score > 0.4 else "Concerning"
                }
        
        # Sort by score to identify problem areas
        sorted_parts = sorted(
            comparison["overall_comparison"].items(), 
            key=lambda x: x[1]["score"]
        )
        
        if sorted_parts:
            worst_part = sorted_parts[0][0]
            comparison["recommendations"].append(
                f"Focus treatment on {worst_part} which shows the most concerns"
            )
        
        return JSONResponse(content=comparison)
        
    except Exception as e:
        logger.error(f"Error in compare_body_parts: {str(e)}")
        raise HTTPException(status_code=500, detail="Error comparing body parts")

@app.post("/api/analyze-body-part/upload")
async def analyze_body_part_upload(
    body_part: str = Form("hands"),
    file: UploadFile = File(...)
):
    """Analyze skin conditions for body parts with direct file upload"""
    try:
        # Normalize body part name
        body_part = body_part.lower()
        
        # Validate file type
        if not file.content_type in ["image/jpeg", "image/jpg", "image/png", "image/webp"]:
            raise HTTPException(status_code=400, detail="Only JPEG, PNG, and WebP images are supported")
        
        # Read uploaded file
        image_data = await file.read()
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        analysis_result = body_skin_analyzer.analyze_body_part(img, body_part)
        
        # Debug logging
        logger.info(f"Analysis result type: {type(analysis_result)}")
        logger.info(f"Analysis result keys: {analysis_result.keys() if isinstance(analysis_result, dict) else 'Not a dict'}")
        
        if not analysis_result.get('success', False):
            error_msg = analysis_result.get('error', 'Analysis failed')
            logger.error(f"Analysis failed: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Ensure the result is JSON serializable
        try:
            import json
            json.dumps(analysis_result)  # Test serialization
            logger.info("Analysis result is JSON serializable")
        except (TypeError, ValueError) as e:
            logger.error(f"Analysis result not JSON serializable: {e}")
            # Convert numpy types to native Python types
            def make_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                elif hasattr(obj, 'item'):  # numpy types
                    return obj.item()
                else:
                    return obj
            
            analysis_result = make_serializable(analysis_result)
            logger.info("Converted analysis result to JSON serializable format")
        
        return JSONResponse(content=analysis_result)
        
    except HTTPException as e:
        logger.error(f"HTTP error in analyze_body_part_upload: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in analyze_body_part_upload: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during body part analysis")

@app.post("/api/analyze-face/upload")
async def analyze_skin_upload(file: UploadFile = File(...)):
    """Analyze facial skin with direct file upload"""
    try:
        # Validate file type
        if not file.content_type in ["image/jpeg", "image/jpg", "image/png", "image/webp"]:
            raise HTTPException(status_code=400, detail="Only JPEG, PNG, and WebP images are supported")
        
        # Read uploaded file
        image_data = await file.read()
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Perform skin analysis
        analysis_result = skin_analyzer.analyze_skin(img)
        
        if analysis_result is None:
            raise HTTPException(status_code=500, detail="Skin analysis failed")
        
        return JSONResponse(content=analysis_result)

    except HTTPException as e:
        logger.error(f"HTTP error in analyze_skin_upload: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in analyze_skin_upload: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during skin analysis")
@app.post("/api/treatment-plans")
async def generate_treatment_plans(request: Request):
    """
    Generate treatment plan options based on skin analysis
    
    Request body:
    {
        "analysis": {...},  # Skin analysis results
        "body_part": "face",  # Optional: face, hands, legs, etc.
        "preferences": {  # Optional user preferences
            "budget": "low/moderate/high",
            "time_commitment": "low/moderate/high",
            "treatment_type": "topical/professional/both"
        }
    }
    """
    try:
        body = await request.json()
        analysis = body.get('analysis', None)
        body_part = body.get('body_part', 'face')
        preferences = body.get('preferences', None)
        
        if not analysis:
            raise HTTPException(status_code=400, detail="No analysis results provided")
        
        # Generate treatment plans
        treatment_plans = treatment_plan_generator.generate_treatment_plans(
            analysis=analysis,
            body_part=body_part,
            user_preferences=preferences
        )
        
        return JSONResponse(content=treatment_plans)
        
    except HTTPException as e:
        logger.error(f"HTTP error in generate_treatment_plans: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in generate_treatment_plans: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="An unexpected error occurred while generating treatment plans"
        )

@app.post("/api/treatment-plans/face")
async def generate_facial_treatment_plans(request: Request):
    """Generate treatment plans specifically for facial skin analysis"""
    try:
        body = await request.json()
        analysis = body.get('analysis', None)
        preferences = body.get('preferences', None)
        
        if not analysis:
            raise HTTPException(status_code=400, detail="No analysis results provided")
        
        treatment_plans = treatment_plan_generator.generate_treatment_plans(
            analysis=analysis,
            body_part="face",
            user_preferences=preferences
        )
        
        return JSONResponse(content=treatment_plans)
        
    except Exception as e:
        logger.error(f"Error in generate_facial_treatment_plans: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating facial treatment plans")

@app.post("/api/treatment-plans/body")
async def generate_body_treatment_plans(request: Request):
    """Generate treatment plans for body parts"""
    try:
        body = await request.json()
        analysis = body.get('analysis', None)
        body_part = body.get('body_part', 'hands')
        preferences = body.get('preferences', None)
        
        if not analysis:
            raise HTTPException(status_code=400, detail="No analysis results provided")
        
        treatment_plans = treatment_plan_generator.generate_treatment_plans(
            analysis=analysis,
            body_part=body_part,
            user_preferences=preferences
        )
        
        return JSONResponse(content=treatment_plans)
        
    except Exception as e:
        logger.error(f"Error in generate_body_treatment_plans: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating body treatment plans")

@app.get("/api/treatment-plan/{plan_id}")
async def get_treatment_plan_details(plan_id: int, request: Request):
    """
    Get detailed information about a specific treatment plan
    
    Query params:
    - treatment_plans: JSON string of the full treatment plans response
    """
    try:
        # Get treatment plans from query or request body
        treatment_plans_json = request.query_params.get('treatment_plans')
        
        if not treatment_plans_json:
            # Try to get from body for POST requests
            body = await request.json() if request.method == "POST" else {}
            treatment_plans = body.get('treatment_plans', None)
        else:
            treatment_plans = json.loads(treatment_plans_json)
        
        if not treatment_plans:
            raise HTTPException(
                status_code=400, 
                detail="No treatment plans provided"
            )
        
        # Get specific plan details
        plan_details = treatment_plan_generator.get_treatment_details(
            plan_id, 
            treatment_plans
        )
        
        if not plan_details.get('success'):
            raise HTTPException(status_code=404, detail=plan_details.get('error'))
        
        return JSONResponse(content=plan_details)
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in get_treatment_plan_details: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching treatment plan details")

@app.post("/api/treatment-plan/{plan_id}/details")
async def post_treatment_plan_details(plan_id: int, request: Request):
    """POST version of get treatment plan details"""
    try:
        body = await request.json()
        treatment_plans = body.get('treatment_plans', None)
        
        if not treatment_plans:
            raise HTTPException(status_code=400, detail="No treatment plans provided")
        
        plan_details = treatment_plan_generator.get_treatment_details(
            plan_id, 
            treatment_plans
        )
        
        if not plan_details.get('success'):
            raise HTTPException(status_code=404, detail=plan_details.get('error'))
        
        return JSONResponse(content=plan_details)
        
    except Exception as e:
        logger.error(f"Error in post_treatment_plan_details: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching treatment plan details")

@app.post("/api/compare-treatments")
async def compare_treatments(request: Request):
    """
    Compare multiple treatment plans side by side
    
    Request body:
    {
        "treatment_plans": {...},  # Full treatment plans response
        "compare_plan_ids": [1, 2, 3]  # Plans to compare
    }
    """
    try:
        body = await request.json()
        treatment_plans = body.get('treatment_plans', None)
        compare_ids = body.get('compare_plan_ids', [])
        
        if not treatment_plans or not compare_ids:
            raise HTTPException(
                status_code=400, 
                detail="Treatment plans and plan IDs required"
            )
        
        # Extract plans to compare
        plans_to_compare = []
        for plan in treatment_plans.get('treatment_plans', []):
            if plan.get('plan_id') in compare_ids:
                plans_to_compare.append(plan)
        
        if len(plans_to_compare) < 2:
            raise HTTPException(
                status_code=400, 
                detail="At least 2 plans required for comparison"
            )
        
        # Create comparison
        comparison = {
            "plans": plans_to_compare,
            "comparison_matrix": treatment_plans.get('comparison_matrix', {}),
            "winner_by_category": {
                "effectiveness": max(plans_to_compare, key=lambda x: x.get('effectiveness_rate', 0)),
                "cost": min(plans_to_compare, key=lambda x: self._extract_min_cost(x.get('cost_range', '$0'))),
                "duration": min(plans_to_compare, key=lambda x: self._extract_max_weeks(x.get('duration_weeks', '0')))
            }
        }
        
        return JSONResponse(content=comparison)
        
    except Exception as e:
        logger.error(f"Error in compare_treatments: {str(e)}")
        raise HTTPException(status_code=500, detail="Error comparing treatments")

# Helper functions for comparison endpoint
def _extract_min_cost(cost_str: str) -> int:
    """Extract minimum cost from cost range string"""
    try:
        # Extract numbers from string like "$200-$400"
        import re
        numbers = re.findall(r'\d+', cost_str)
        return int(numbers[0]) if numbers else 0
    except:
        return 0

def _extract_max_weeks(duration_str: str) -> int:
    """Extract maximum weeks from duration string"""
    try:
        # Extract numbers from string like "10-16"
        import re
        numbers = re.findall(r'\d+', duration_str)
        return int(numbers[-1]) if numbers else 0
    except:
        return 0

@app.post("/api/treatment-progress")
async def track_treatment_progress(request: Request):
    """
    Track progress on a treatment plan
    
    Request body:
    {
        "plan_id": 1,
        "current_week": 4,
        "user_notes": "Seeing some improvements",
        "side_effects": [],
        "effectiveness_rating": 7
    }
    """
    try:
        body = await request.json()
        plan_id = body.get('plan_id')
        current_week = body.get('current_week')
        
        if not plan_id or current_week is None:
            raise HTTPException(
                status_code=400, 
                detail="Plan ID and current week required"
            )
        
        # In a real app, you'd save this to a database
        # For now, return acknowledgment
        progress_data = {
            "success": True,
            "plan_id": plan_id,
            "current_week": current_week,
            "status": "on_track",
            "next_steps": f"Continue with week {current_week + 1} of your treatment plan",
            "recorded_at": str(body.get('recorded_at', 'now'))
        }
        
        return JSONResponse(content=progress_data)
        
    except Exception as e:
        logger.error(f"Error in track_treatment_progress: {str(e)}")
        raise HTTPException(status_code=500, detail="Error tracking treatment progress")
# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "Skin Analyzer API", "version": "1.0.0", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)