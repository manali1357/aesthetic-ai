import os
import openai
import httpx
import json
from typing import Dict, Any, List
from logger_utils import logger

class GPTRecommendations:
    def __init__(self):
        """Initialize OpenAI client"""
        try:
            api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')
            if api_key and api_key not in ['your_openai_api_key_here', 'your_openrouter_api_key_here']:
                http_client = httpx.Client(trust_env=False)
                self.client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1",http_client=http_client)
                logger.info("Info: OpenRouter client initialized successfully")
            else:
                logger.error("Error: OPENROUTER_API_KEY or OPENAI_API_KEY not set. GPT recommendations will not function.")
                self.client = None
        except Exception as e:
            logger.error(f"Error initializing OpenRouter client: {e}")
            self.client = None

    def get_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get personalized skincare recommendations based on skin analysis
        
        Args:
            analysis: Dictionary containing skin analysis results
            
        Returns:
            dict: Personalized recommendations and advice
        """
        try:
            if not isinstance(analysis, dict):
                logger.warning(f"Warning: analysis is not a dictionary: {type(analysis)}")
                return {
                    "success": False,
                    "error": "Invalid analysis data format"
                }
            
            concerns = [concern_name for concern_name, concern_data in analysis.items() if concern_name != "overall_score" and isinstance(concern_data, dict) and concern_data.get("detected", False)]

            if not concerns:
                return {
                    "success": True,
                    "recommendations": {
                        "full_recommendations": "**Your Skin Analysis Results**\n\nGreat news! No major skin concerns were detected in your analysis. Your skin appears to be healthy and well-maintained.\n\n**General Maintenance Recommendations:**\n\n1. **Daily Routine:**\n   - Morning: Gentle cleanser, moisturizer, sunscreen SPF 30+\n   - Evening: Gentle cleanser, moisturizer\n\n2. **Prevention Tips:**\n   - Continue using broad-spectrum sunscreen daily\n   - Stay hydrated by drinking plenty of water\n   - Get adequate sleep (7-9 hours)\n   - Maintain a balanced diet rich in antioxidants\n\n3. **Lifestyle Recommendations:**\n   - Manage stress through meditation or exercise\n   - Avoid touching your face throughout the day\n   - Use clean pillowcases and towels\n\n**Remember:** Even healthy skin benefits from consistent care and protection!",
                        "recommendations": [
                            "Use a gentle cleanser twice daily",
                            "Apply broad-spectrum sunscreen with SPF 30+",
                            "Stay hydrated by drinking plenty of water"
                        ],
                        "lifestyle_tips": [
                            "Get 7-9 hours of sleep",
                            "Manage stress through meditation or exercise",
                            "Eat a balanced diet rich in antioxidants"
                        ],
                        "product_suggestions": [
                            "Gentle cleanser suitable for your skin type",
                            "Moisturizer with hyaluronic acid",
                            "Sunscreen with SPF 30 or higher"
                        ]
                    },
                    "analysis_summary": f"Overall Skin Health Score: {analysis.get('overall_score', 1.0):.2f}/1.0\n\nNo major skin concerns detected. Skin appears healthy."
                }
            
            if not self.client:
                logger.error("GPT client is not available. Recommendations cannot be generated.")
                return {
                    "success": False,
                    "error": "GPT client is unavailable."
                }

            ingredient_recs = self.get_ingredient_recommendations(concerns)
            
            analysis_summary = self._prepare_analysis_summary(analysis)
            prompt = self._create_comprehensive_recommendations_prompt(analysis_summary, ingredient_recs)
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert dermatologist and skincare specialist. 
                        Provide personalized, evidence-based skincare recommendations based on skin analysis results.
                        Be specific, practical, and prioritize the most important concerns first.
                        Include both immediate actions and long-term strategies.
                        Consider the ingredient recommendations provided and integrate them into a comprehensive routine."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1200,
                temperature=0.7
            )
            
            recommendations_text = response.choices[0].message.content
            structured_data = self._convert_to_structured_format(recommendations_text)
            
            return {
                "success": True,
                "recommendations": {
                    "full_recommendations": recommendations_text,
                    **structured_data
                },
                "analysis_summary": analysis_summary,
                "ingredient_recommendations": ingredient_recs
            }
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return {
                "success": False,
                "error": f"An error occurred while generating recommendations: {str(e)}"
            }

    def _create_comprehensive_recommendations_prompt(self, analysis_summary: str, ingredient_recs: Dict[str, Any]) -> str:
        """Create a comprehensive recommendations prompt"""
        return f"""
Based on the following skin analysis and ingredient recommendations, provide comprehensive skincare advice:

{analysis_summary}

Ingredient Recommendations Available:
{json.dumps(ingredient_recs, indent=2)}

Please provide detailed skincare recommendations that integrate these ingredients into a complete routine. Include:

1. **Primary Concerns** (List the top 2-3 most important issues)
2. **Immediate Actions** (What to do right now)
3. **Daily Routine** (Morning and evening routines incorporating recommended ingredients)
4. **Weekly Treatments** (Any weekly treatments or masks)
5. **Lifestyle Recommendations** (Diet, sleep, stress management)
6. **Product Integration** (How to incorporate the recommended ingredients)
7. **Timeline** (Expected improvements and long-term maintenance)
8. **Precautions** (Any warnings or things to avoid)

Focus on practical, actionable advice that prioritizes skin health and safety.
"""

    def _prepare_analysis_summary(self, analysis: Dict[str, Any]) -> str:
        """Prepare a summary of the skin analysis for recommendations"""
        if not isinstance(analysis, dict):
            return "Error: Invalid analysis data format"
            
        concerns = []
        overall_score = analysis.get("overall_score", 0.5)
        
        for concern_name, concern_data in analysis.items():
            if concern_name == "overall_score":
                continue
                
            if isinstance(concern_data, dict) and concern_data.get("detected", False):
                severity = concern_data.get("severity", 0)
                confidence = concern_data.get("confidence", 0)
                concerns.append({
                    "name": concern_name,
                    "severity": severity,
                    "confidence": confidence,
                    "description": concern_data.get("description", "")
                })
        
        # Sort by severity
        concerns.sort(key=lambda x: x["severity"], reverse=True)
        
        summary = f"Overall Skin Health Score: {overall_score:.2f}/1.0\n\n"
        summary += "Detected Skin Concerns:\n"
        
        for concern in concerns:
            summary += f"- {concern['name'].title()}: Severity {concern['severity']:.2f}, Confidence {concern['confidence']:.2f}\n"
            if concern['description']:
                summary += f"  Description: {concern['description']}\n"
        
        if not concerns:
            summary += "No major skin concerns detected. Skin appears healthy.\n"
        
        return summary

    def _convert_to_structured_format(self, recommendations_text: str) -> Dict[str, Any]:
        """Convert text recommendations into structured format"""
        try:
            lines = recommendations_text.split('\n')
            
            recommendations = []
            lifestyle_tips = []
            product_suggestions = []
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if 'routine' in line.lower() or 'daily' in line.lower():
                    current_section = 'recommendations'
                elif 'lifestyle' in line.lower() or 'sleep' in line.lower() or 'diet' in line.lower():
                    current_section = 'lifestyle'
                elif 'product' in line.lower() or 'ingredient' in line.lower():
                    current_section = 'products'
                elif line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    content = line[1:].strip()
                    if current_section == 'recommendations' and len(recommendations) < 3:
                        recommendations.append(content)
                    elif current_section == 'lifestyle' and len(lifestyle_tips) < 3:
                        lifestyle_tips.append(content)
                    elif current_section == 'products' and len(product_suggestions) < 3:
                        product_suggestions.append(content)
                elif line.startswith('1.') or line.startswith('2.') or line.startswith('3.'):
                    content = line[3:].strip()
                    if current_section == 'recommendations' and len(recommendations) < 3:
                        recommendations.append(content)
                    elif current_section == 'lifestyle' and len(lifestyle_tips) < 3:
                        lifestyle_tips.append(content)
                    elif current_section == 'products' and len(product_suggestions) < 3:
                        product_suggestions.append(content)
            
            # Ensure we have at least some recommendations
            if len(recommendations) < 3:
                recommendations.extend([
                    "Use a gentle cleanser twice daily",
                    "Apply broad-spectrum sunscreen with SPF 30+",
                    "Stay hydrated by drinking plenty of water"
                ][:3-len(recommendations)])
            
            if len(lifestyle_tips) < 3:
                lifestyle_tips.extend([
                    "Get 7-9 hours of sleep",
                    "Manage stress through meditation or exercise",
                    "Eat a balanced diet rich in antioxidants"
                ][:3-len(lifestyle_tips)])
            
            if len(product_suggestions) < 3:
                product_suggestions.extend([
                    "Gentle cleanser suitable for your skin type",
                    "Moisturizer with hyaluronic acid",
                    "Sunscreen with SPF 30 or higher"
                ][:3-len(product_suggestions)])
            
            return {
                "recommendations": recommendations[:3],
                "lifestyle_tips": lifestyle_tips[:3],
                "product_suggestions": product_suggestions[:3]
            }
            
        except Exception as e:
            logger.error(f"Error converting to structured format: {e}")
            return {
                "recommendations": [
                    "Use a gentle cleanser twice daily",
                    "Apply broad-spectrum sunscreen with SPF 30+",
                    "Stay hydrated by drinking plenty of water"
                ],
                "lifestyle_tips": [
                    "Get 7-9 hours of sleep",
                    "Manage stress through meditation or exercise",
                    "Eat a balanced diet rich in antioxidants"
                ],
                "product_suggestions": [
                    "Gentle cleanser suitable for your skin type",
                    "Moisturizer with hyaluronic acid",
                    "Sunscreen with SPF 30 or higher"
                ]
            }
    
    def get_ingredient_recommendations(self, concerns: List[str]) -> Dict[str, Any]:
        """
        Get ingredient recommendations from LLM based on detected skin concerns
        LLM will fetch and synthesize information from online knowledge
        """
        try:
            if not self.client:
                return {
                    "error": "GPT client is unavailable."
                }
            
            prompt = f"""Based on the following skin concerns: {', '.join(concerns)}

Please provide ingredient recommendations in JSON format with this structure:
{{
    "primary_concerns": ["list", "of", "concerns"],
    "ingredient_recommendations": [
        {{
            "name": "Ingredient Name",
            "category": "Category",
            "benefits": ["benefit1", "benefit2"],
            "usage": "How to use"
        }}
    ],
    "routine_integration": {{
        "morning": ["ingredient1"],
        "evening": ["ingredient2"]
    }}
}}

Return the response as a valid JSON object."""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "Provide evidence-based ingredient recommendations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=1500,
                temperature=0.3
            )
            
            ingredients_data = json.loads(response.choices[0].message.content)
            return ingredients_data
            
        except Exception as e:
            logger.error(f"Error getting ingredient recommendations: {e}")
            return {"error": "Failed to get ingredient recommendations"}
