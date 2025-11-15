import os
import json
import httpx
from typing import Dict, Any, List, Optional
from logger_utils import logger
import openai

class TreatmentPlanGenerator:
    """Generate evidence-based treatment plans for skin concerns"""
    
    def __init__(self):
        """Initialize OpenAI client"""
        try:
            api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')
            if api_key and api_key not in ['your_openai_api_key_here', 'your_openrouter_api_key_here']:
                http_client = httpx.Client(trust_env=False)
                self.client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", http_client=http_client)
                logger.info("Treatment Plan Generator initialized successfully")
            else:
                logger.warning("OPENROUTER_API_KEY or OPENAI_API_KEY not set. Treatment plans will not be available.")
                self.client = None
        except Exception as e:
            logger.error(f"Could not initialize OpenAI client: {e}")
            self.client = None
    
    def generate_treatment_plans(
        self, 
        analysis: Dict[str, Any],
        body_part: str = "face",
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate multiple treatment plan options based on analysis
        
        Args:
            analysis: Skin analysis results
            body_part: Body part analyzed (face, hands, legs, etc.)
            user_preferences: User preferences (budget, time commitment, treatment type)
            
        Returns:
            Dict containing multiple treatment plan options
        """
        try:
            # Extract concerns from analysis
            concerns = self._extract_concerns(analysis)
            
            if not concerns:
                return self._get_maintenance_plans(body_part)
            
            # Get primary concern (highest severity)
            primary_concern = max(concerns, key=lambda x: x['severity'])
            
            if not self.client:
                logger.error("OpenAI client not available - cannot generate treatment plans")
                return {
                    "success": False,
                    "error": "AI service unavailable for treatment plan generation"
                }
            
            # Generate treatment plans using LLM
            prompt = self._create_treatment_plan_prompt(
                primary_concern, 
                concerns, 
                body_part,
                user_preferences
            )
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert dermatologist and aesthetic physician specializing in evidence-based treatment planning.
                        Generate multiple treatment plan options with SPECIFIC treatment names and combinations.
                        
                        IMPORTANT: Use actual aesthetic treatment names like:
                        - Chemical Peels (Glycolic, Salicylic, TCA)
                        - Laser Treatments (IPL, Fractional, PDL)
                        - Microneedling with RF
                        - PRP (Vampire Facial)
                        - Dermal Fillers
                        - Botox
                        - LED Light Therapy
                        - Microdermabrasion
                        - Hydrafacial
                        - Morpheus8
                        - Clear + Brilliant
                        
                        Combine treatments logically based on the skin concerns."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=2000,
                temperature=0.4
            )
            
            treatment_plans = json.loads(response.choices[0].message.content)
            
            # Validate and enhance the response
            treatment_plans = self._validate_and_enhance_plans(treatment_plans)
            
            return {
                "success": True,
                "primary_concern": primary_concern['name'],
                "all_concerns": [c['name'] for c in concerns],
                "body_part": body_part,
                **treatment_plans
            }
            
        except Exception as e:
            logger.error(f"Error generating treatment plans: {e}")
            return {
                "success": False,
                "error": f"Failed to generate treatment plans: {str(e)}"
            }
    
    def _extract_concerns(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and prioritize concerns from analysis"""
        concerns = []
        
        # Define concern mapping to proper names
        concern_names = {
            'breakouts': 'Acne & Breakouts',
            'pigmentation': 'Hyperpigmentation',
            'redness': 'Redness & Inflammation', 
            'aging': 'Aging & Wrinkles',
            'dehydration': 'Dehydration & Dryness',
            'under_eye': 'Under Eye Concerns'
        }
        
        for concern_name, concern_data in analysis.items():
            if concern_name in ["overall_score", "overall_description", "estimated_age", "body_part"]:
                continue
            
            if isinstance(concern_data, dict) and concern_data.get("detected", False):
                display_name = concern_names.get(concern_name, concern_name.replace('_', ' ').title())
                concerns.append({
                    "name": display_name,
                    "technical_name": concern_name,
                    "severity": concern_data.get("severity", 0.5),
                    "confidence": concern_data.get("confidence", 0.5),
                    "description": concern_data.get("description", ""),
                    "observations": concern_data.get("specific_observations", "")
                })
        
        # Sort by severity
        concerns.sort(key=lambda x: x['severity'], reverse=True)
        return concerns
    
    def _create_treatment_plan_prompt(
        self, 
        primary_concern: Dict[str, Any],
        all_concerns: List[Dict[str, Any]],
        body_part: str,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create detailed prompt for treatment plan generation"""
        
        preferences_text = ""
        if user_preferences:
            budget = user_preferences.get('budget', 'moderate')
            time_commitment = user_preferences.get('time_commitment', 'moderate')
            treatment_type = user_preferences.get('treatment_type', 'both')
            preferences_text = f"\nUser Preferences:\n- Budget: {budget}\n- Time Commitment: {time_commitment}\n- Treatment Type: {treatment_type}"
        
        # Build detailed concern analysis
        concern_details = "\n".join([
            f"- {c['name']}: Severity {c['severity']:.2f}/1.0 - {c['observations']}"
            for c in all_concerns
        ])
        
        return f"""
Generate evidence-based aesthetic treatment plans for facial skin concerns.

DETECTED SKIN CONCERNS:
Primary Concern: {primary_concern['name']} (Severity: {primary_concern['severity']:.2f}/1.0)
All Concerns:
{concern_details}
{preferences_text}

REQUIREMENTS FOR TREATMENT PLANS:

1. Plan 1 (Professional + Advanced):
   - Combination of in-office aesthetic treatments
   - Include specific treatment names and technologies
   - Higher effectiveness, higher cost
   - Examples: "IPL Photofacial + Microneedling with PRP", "Fractional Laser + Clear + Brilliant"

2. Plan 2 (Professional + Topical):
   - Mix of in-office treatments and medical-grade topicals
   - Moderate effectiveness and cost
   - Examples: "Chemical Peel Series + Prescription Retinoids", "LED Therapy + Growth Factor Serums"

3. Plan 3 (Topical + At-Home):
   - Medical-grade topical treatments and devices
   - Lower cost, good for maintenance
   - Examples: "Tranexamic Acid + Retinol Protocol", "Vitamin C + Niacinamide Regimen"

SPECIFIC TREATMENT OPTIONS BY CONCERN:

For Under Eye Concerns (primary):
- PRP Under Eye Treatment
- Fractional Laser for under eyes
- Hyaluronic Acid Filler
- Radiofrequency Microneedling
- Vitamin C + Caffeine Topicals

For Hyperpigmentation:
- IPL Photofacial
- Chemical Peels (Glycolic, Salicylic)
- Tranexamic Acid Mesotherapy
- Laser Toning
- Vitamin C + Alpha Arbutin

For Redness & Inflammation:
- PDL Laser (Pulsed Dye Laser)
- BBL (BroadBand Light)
- LED Red Light Therapy
- Centella Asiatica + Azelaic Acid
- Growth Factor Serums

Please provide 3 treatment plan options in this EXACT JSON format:

{{
    "treatment_plans": [
        {{
            "plan_id": 1,
            "name": "Specific Treatment Combination Name",
            "type": "Professional + Advanced",
            "effectiveness_rate": 85,
            "duration_weeks": "8-12",
            "cost_range": "$1200-$2500",
            "cost_currency": "USD",
            "description": "Detailed description of what this treatment combination addresses",
            "professional_treatments": [
                {{
                    "name": "Specific Treatment Name",
                    "type": "Laser/Peel/Injection/etc",
                    "frequency": "Every 4 weeks",
                    "sessions": "3-4 sessions",
                    "targets": ["concern1", "concern2"]
                }}
            ],
            "topical_treatments": [
                {{
                    "product_type": "Serum/Cream",
                    "key_ingredients": ["Ingredient1", "Ingredient2"],
                    "purpose": "What it addresses",
                    "frequency": "Daily/Twice daily"
                }}
            ],
            "timeline": [
                {{
                    "week": "1-4",
                    "phase": "Initial Treatment",
                    "actions": ["Action 1", "Action 2"],
                    "expected_results": "Initial improvements"
                }}
            ],
            "research_evidence": [
                {{
                    "study": "Clinical study name",
                    "findings": "85% improvement in targeted concerns",
                    "source": "Journal of Cosmetic Dermatology"
                }}
            ],
            "contraindications": ["Active acne", "Pregnancy", "Recent sun exposure"],
            "expected_results": {{
                "week_4": "Reduced pigmentation, improved texture",
                "week_8": "Significant improvement in primary concerns",
                "week_12": "Optimal results, begin maintenance",
                "maintenance": "Quarterly touch-ups recommended"
            }},
            "safety_score": "moderate",
            "requires_consultation": true,
            "downtime": "1-3 days",
            "pain_level": "mild"
        }}
    ],
    "comparison_matrix": {{
        "effectiveness": {{"plan_1": 88, "plan_2": 75, "plan_3": 65}},
        "cost": {{"plan_1": "high", "plan_2": "moderate", "plan_3": "low"}},
        "time_commitment": {{"plan_1": "moderate", "plan_2": "moderate", "plan_3": "low"}},
        "invasiveness": {{"plan_1": "moderate", "plan_2": "low", "plan_3": "none"}}
    }},
    "general_recommendations": {{
        "do": ["Use SPF 50+ daily", "Avoid sun exposure", "Follow post-care instructions"],
        "dont": ["Don't pick at skin", "Avoid harsh products", "No sunbeds"],
        "lifestyle": ["Adequate sleep", "Balanced diet", "Stress management"]
    }}
}}

Focus on creating SPECIFIC, REAL aesthetic treatment names and combinations that logically address the detected concerns.
"""
    
    def _validate_and_enhance_plans(self, plans: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enhance treatment plans with additional data"""
        
        # Ensure required fields exist
        if "treatment_plans" not in plans:
            plans["treatment_plans"] = []
        
        # Add metadata to each plan
        for i, plan in enumerate(plans.get("treatment_plans", [])):
            if "plan_id" not in plan:
                plan["plan_id"] = i + 1
            
            # Ensure treatment names are specific
            if "name" in plan and any(generic in plan["name"].lower() for generic in ["professional treatment", "topical treatment", "basic treatment"]):
                # Regenerate with more specific name based on type
                plan_type = plan.get("type", "").lower()
                if "professional" in plan_type or "advanced" in plan_type:
                    plan["name"] = "Customized Aesthetic Treatment Plan"
                elif "topical" in plan_type:
                    plan["name"] = "Medical-Grade Topical Regimen"
                else:
                    plan["name"] = "Comprehensive Skincare Protocol"
            
            # Add safety score based on invasiveness
            plan_type = plan.get("type", "").lower()
            if "professional" in plan_type or "advanced" in plan_type:
                plan["safety_score"] = "moderate"
                plan["requires_consultation"] = True
                plan.setdefault("downtime", "1-3 days")
                plan.setdefault("pain_level", "mild")
            else:
                plan["safety_score"] = "high"
                plan["requires_consultation"] = False
                plan.setdefault("downtime", "none")
                plan.setdefault("pain_level", "none")
            
            # Validate effectiveness rate
            if "effectiveness_rate" in plan:
                effectiveness = plan["effectiveness_rate"]
                if effectiveness > 95:
                    plan["effectiveness_rate"] = 95
                elif effectiveness < 50:
                    plan["effectiveness_rate"] = 50
        
        return plans
    
    def _get_maintenance_plans(self, body_part: str) -> Dict[str, Any]:
        """Generate maintenance plans for healthy skin"""
        return {
            "success": True,
            "primary_concern": "maintenance",
            "all_concerns": [],
            "body_part": body_part,
            "treatment_plans": [
                {
                    "plan_id": 1,
                    "name": "Preventive Aesthetic Maintenance",
                    "type": "Preventive + Topical",
                    "effectiveness_rate": 90,
                    "duration_weeks": "ongoing",
                    "cost_range": "$100-$300/month",
                    "cost_currency": "USD",
                    "description": "Maintain optimal skin health and prevent future concerns with medical-grade prevention",
                    "professional_treatments": [
                        {
                            "name": "Quarterly Hydrafacial",
                            "type": "Medical Facial",
                            "frequency": "Every 3 months",
                            "sessions": "4 per year",
                            "targets": ["prevention", "maintenance"]
                        }
                    ],
                    "topical_treatments": [
                        {
                            "product_type": "Medical-Grade Skincare",
                            "key_ingredients": ["Growth Factors", "Antioxidants", "Peptides"],
                            "purpose": "Maintain skin health and prevent aging",
                            "frequency": "Daily"
                        }
                    ],
                    "timeline": [
                        {
                            "week": "Daily",
                            "phase": "Maintenance",
                            "actions": ["Morning: Cleanse, antioxidant serum, moisturizer, SPF 50+", "Evening: Double cleanse, treatment serum, moisturizer"],
                            "expected_results": "Maintained skin health and prevention"
                        }
                    ],
                    "safety_score": "high",
                    "requires_consultation": False,
                    "downtime": "none",
                    "pain_level": "none"
                }
            ],
            "general_recommendations": {
                "do": [
                    "Use medical-grade skincare daily",
                    "Apply SPF 50+ every morning",
                    "Get quarterly professional maintenance",
                    "Stay hydrated and maintain healthy diet"
                ],
                "dont": [
                    "Don't skip sunscreen",
                    "Avoid excessive sun exposure",
                    "Don't use harsh DIY treatments"
                ],
                "lifestyle": [
                    "7-9 hours quality sleep nightly",
                    "Balanced diet rich in antioxidants",
                    "Regular exercise and stress management"
                ]
            }
        }
    
    def get_treatment_details(self, plan_id: int, treatment_plans: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed information about a specific treatment plan"""
        try:
            plans = treatment_plans.get("treatment_plans", [])
            
            for plan in plans:
                if plan.get("plan_id") == plan_id:
                    return {
                        "success": True,
                        "plan": plan
                    }
            
            return {
                "success": False,
                "error": f"Plan with ID {plan_id} not found"
            }
            
        except Exception as e:
            logger.error(f"Error getting treatment details: {e}")
            return {
                "success": False,
                "error": str(e)
            }