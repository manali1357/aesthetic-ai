import base64
import cv2
import io
import mediapipe as mp
import numpy as np
import httpx
import openai
import os

from logger_utils import logger


class SkinAnalyzer:
    def __init__(self):
        """Initialize MediaPipe face detection and analysis"""
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize face detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        
        # Initialize face mesh for detailed analysis
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize OpenAI client for GPT-4o analysis
        try:
            api_key = os.getenv('OPENROUTER_API_KEY')
            if api_key and api_key != 'your_openai_api_key_here':
                logger.info(f"Info: Initializing OpenAI client with API key: {api_key[:10]}...")
                
                # Create OpenAI client with just the API key
                http_client = httpx.Client(trust_env=False)
                self.openai_client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", http_client=http_client)
                logger.info("Info: OpenAI client initialized successfully")
            else:
                logger.info("Info: OPENAI_API_KEY not set or using default value. GPT-4o analysis will use fallback mode.")
                self.openai_client = None
        except Exception as e:
            logger.error(f"Info: Could not initialize OpenAI client: {e}")
            logger.error(f"Info: Error args: {e.args}")
            self.openai_client = None

    def analyze_skin(self, image):
        """
        Analyze skin conditions from the uploaded image
        
        Args:
            image: PIL Image object
            
        Returns:
            dict: Analysis results with detected skin concerns
        """
        try:
            # Convert PIL image to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            
            # Get face detection results
            face_results = self.face_detection.process(rgb_image)
            
            # Check if any face is detected
            if not face_results.detections:
                return {
                    "success": False,
                    "analysis": {},
                    "error": "No face detected in the image. Please ensure:\n• Your face is clearly visible\n• The image is well-lit\n• Your face takes up a good portion of the image\n• You're not wearing heavy makeup or accessories that obscure your face\n• Try taking a photo from chest level showing your full face"
                }
            
            # Get face mesh results for detailed analysis
            mesh_results = self.face_mesh.process(rgb_image)
            
            # Check if we can get facial landmarks (even for partial faces)
            if not mesh_results.multi_face_landmarks:
                # Try with more lenient face detection for partial faces
                logger.info("Info: No face mesh landmarks found, attempting analysis with partial face detection")
                
                # For partial faces, we'll still try to analyze what we can see
                # Check if we have at least some face detection confidence
                if face_results.detections:
                    detection = face_results.detections[0]
                    confidence = detection.score[0]
                    
                    if confidence > 0.3:  # Lower threshold for partial faces
                        logger.info(f"Info: Partial face detected with confidence: {confidence}")
                        # Proceed with analysis even for partial faces
                        analysis = self._analyze_skin_with_gpt(image, is_partial_face=True)
                        
                        return {
                            "success": True,
                            "analysis": analysis,
                            "warning": "Partial face detected. Analysis may be limited to visible areas."
                        }
                    else:
                        return {
                            "success": False,
                            "analysis": {},
                            "error": "Face detection confidence too low. Please ensure:\n• Your face is clearly visible and well-lit\n• You're looking directly at the camera\n• The image is high quality and not blurry\n• Try taking a photo from a closer distance"
                        }
                else:
                    return {
                        "success": False,
                        "analysis": {},
                        "error": "Unable to analyze facial features. Please ensure:\n• Your face is clearly visible and centered\n• The image is high quality and well-lit\n• You're looking directly at the camera\n• No obstructions are covering your face\n• Try taking a photo from chest level showing your full face"
                    }
            
            # Full face detected, proceed with normal analysis
            logger.info("Info: Full face detected, proceeding with complete analysis")
            analysis = self._analyze_skin_with_gpt(image, is_partial_face=False)
            
            return {
                "success": True,
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_skin: {e}")
            return {
                "success": False,
                "analysis": {},
                "error": f"Analysis failed: {str(e)}. Please try uploading a different image."
            }

    def _analyze_skin_with_gpt(self, image, is_partial_face=False):
        """
        Use GPT-4o to analyze skin conditions from the image
        
        Args:
            image: PIL Image object
            is_partial_face: Boolean indicating if only partial face is visible
            
        Returns:
            dict: Detailed skin analysis from GPT-4o
        """
        try:
            # Convert image to base64 for GPT-4o
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG', quality=85)
            base64_image = base64.b64encode(img_buffer.getvalue()).decode()
            data_url = f"data:image/jpeg;base64,{base64_image}"
            
            # Create prompt for GPT-4o
            partial_face_note = ""
            if is_partial_face:
                partial_face_note = "\nIMPORTANT: This appears to be a partial face image. Only analyze the skin areas that are clearly visible. If certain areas are not visible, mark them as not detected and explain what areas are visible vs. not visible."
            
            prompt = f"""You are a board-certified dermatologist analyzing a facial image for skin conditions. Examine the image carefully and provide a clinical assessment.{partial_face_note}

ANALYSIS REQUIREMENTS:
- Examine the skin texture, tone, and visible conditions
- Look specifically for: acne/breakouts, hyperpigmentation, redness/rosacea, fine lines/wrinkles, dehydration/dryness, under-eye concerns
- Assess severity on a scale of 0.0-1.0 based on what you actually observe
- Be specific about locations and characteristics of any conditions found
- If the image quality prevents clear analysis, state this explicitly
- Do not assume conditions exist if they are not clearly visible

Provide your assessment in this exact JSON format:

{{
  "overall_score": 0.0,
  "overall_description": "Comprehensive summary based on actual observations",
  "estimated_age": 0,
  "breakouts": {{
    "detected": true,
    "severity": 0.0,
    "confidence": 0.0,
    "description": "What you actually observe regarding breakouts, acne, inflammation",
    "specific_observations": "Exact details about location, type, and characteristics"
  }},
  "pigmentation": {{
    "detected": false,
    "severity": 0.0,
    "confidence": 0.0,
    "description": "What you observe about skin tone, pigmentation patterns, dark spots",
    "specific_observations": "Exact details about pigmentation distribution"
  }},
  "redness": {{
    "detected": true,
    "severity": 0.0,
    "confidence": 0.0,
    "description": "What you observe about redness, inflammation, rosacea",
    "specific_observations": "Exact details about redness patterns and intensity"
  }},
  "aging": {{
    "detected": false,
    "severity": 0.0,
    "confidence": 0.0,
    "description": "What you observe about fine lines, wrinkles, texture changes",
    "specific_observations": "Exact details about aging signs and their characteristics"
  }},
  "dehydration": {{
    "detected": true,
    "severity": 0.0,
    "confidence": 0.0,
    "description": "What you observe about skin moisture, dryness, texture",
    "specific_observations": "Exact details about hydration levels and skin texture"
  }},
  "under_eye": {{
    "detected": false,
    "severity": 0.0,
    "confidence": 0.0,
    "description": "What you observe about under-eye area, dark circles, puffiness",
    "specific_observations": "Exact details about under-eye concerns"
  }}
}}

ANALYSIS REQUIREMENTS:

1. **Breakouts/Acne**: Look for and describe:
   - Pimples, blackheads, whiteheads
   - Inflammation and redness
   - Pore congestion and texture
   - Location and distribution

2. **Pigmentation**: Analyze and describe:
   - Skin tone uniformity
   - Dark spots, hyperpigmentation
   - Sun damage and age spots
   - Distribution patterns

3. **Redness**: Examine and describe:
   - Inflammation patterns
   - Rosacea or irritation
   - Broken capillaries
   - Sensitivity indicators

4. **Aging**: Look for and describe:
   - Fine lines and wrinkles
   - Texture changes
   - Loss of elasticity
   - Age-related pigmentation

5. **Dehydration**: Assess and describe:
   - Moisture levels
   - Dryness and flakiness
   - Skin texture quality
   - Dullness or lack of radiance

6. **Under-eye concerns**: Examine and describe:
   - Dark circles
   - Puffiness and bags
   - Fine lines
   - Overall under-eye appearance

SCORING GUIDELINES:
- detected: true only if you clearly see the concern
- severity: 0.0 (none) to 1.0 (severe) based on actual visibility
- confidence: 0.0 to 1.0 based on image clarity and your certainty
- overall_score: Calculate as (1 - average_severity_of_detected_concerns) * 100
- If no concerns detected, overall_score should be 0.9-1.0
- If severe concerns detected, overall_score should be 0.1-0.4

Be extremely specific and descriptive. Only report what you can actually see in the image. Vary your assessments based on the actual image content.

Return ONLY the JSON object, no additional text."""

            if not self.openai_client:
                # Return error when AI service is not available
                logger.error("OpenAI client not available for skin analysis")
                return {
                    "error": "AI service unavailable for skin analysis",
                    "overall_score": None,
                    "overall_description": "Skin analysis service is currently unavailable.",
                    "breakouts": {"detected": False, "severity": 0, "confidence": 0, "description": "Service unavailable", "specific_observations": "AI service error"},
                    "pigmentation": {"detected": False, "severity": 0, "confidence": 0, "description": "Service unavailable", "specific_observations": "AI service error"},
                    "redness": {"detected": False, "severity": 0, "confidence": 0, "description": "Service unavailable", "specific_observations": "AI service error"},
                    "aging": {"detected": False, "severity": 0, "confidence": 0, "description": "Service unavailable", "specific_observations": "AI service error"},
                    "dehydration": {"detected": False, "severity": 0, "confidence": 0, "description": "Service unavailable", "specific_observations": "AI service error"},
                    "under_eye": {"detected": False, "severity": 0, "confidence": 0, "description": "Service unavailable", "specific_observations": "AI service error"}
                }
            
            response = self.openai_client.chat.completions.create(
                model="anthropic/claude-3.5-sonnet",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a board-certified dermatologist specializing in facial skin analysis. Analyze the provided image with clinical precision. Look for specific skin conditions including acne, pigmentation issues, redness, signs of aging, dehydration, and under-eye concerns. Provide detailed, evidence-based assessments. If the image is unclear or no conditions are visible, clearly state this rather than making assumptions."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_tokens=1000,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            try:
                import json
                analysis_text = response.choices[0].message.content.strip()
                # Remove any markdown formatting if present
                if analysis_text.startswith('```json'):
                    analysis_text = analysis_text[7:]
                if analysis_text.endswith('```'):
                    analysis_text = analysis_text[:-3]
                
                # Try to extract JSON from the response if it's not pure JSON
                if not analysis_text.startswith('{'):
                    # Look for JSON object in the response
                    start_idx = analysis_text.find('{')
                    end_idx = analysis_text.rfind('}') + 1
                    if start_idx != -1 and end_idx != 0:
                        analysis_text = analysis_text[start_idx:end_idx]
                
                analysis = json.loads(analysis_text)
                
                # Validate the analysis structure
                if not isinstance(analysis, dict):
                    logger.error(f"GPT response is not a dictionary: {type(analysis)}")
                    # Return error response
                    return {
                        "error": "Invalid response format from AI service",
                        "overall_score": None,
                        "overall_description": "Analysis failed due to invalid response format.",
                        "breakouts": {"detected": False, "severity": 0, "confidence": 0, "description": "Analysis failed", "specific_observations": "Invalid response format"},
                        "pigmentation": {"detected": False, "severity": 0, "confidence": 0, "description": "Analysis failed", "specific_observations": "Invalid response format"},
                        "redness": {"detected": False, "severity": 0, "confidence": 0, "description": "Analysis failed", "specific_observations": "Invalid response format"},
                        "aging": {"detected": False, "severity": 0, "confidence": 0, "description": "Analysis failed", "specific_observations": "Invalid response format"},
                        "dehydration": {"detected": False, "severity": 0, "confidence": 0, "description": "Analysis failed", "specific_observations": "Invalid response format"},
                        "under_eye": {"detected": False, "severity": 0, "confidence": 0, "description": "Analysis failed", "specific_observations": "Invalid response format"}
                    }
                
                # Ensure all required fields are present and have correct types
                required_concerns = ['breakouts', 'pigmentation', 'redness', 'aging', 'dehydration', 'under_eye']
                for concern in required_concerns:
                    if concern not in analysis:
                        logger.info(f"Missing concern in GPT response: {concern}")
                        # Add missing concern with default values
                        analysis[concern] = {
                            "detected": False,
                            "severity": 0.0,
                            "confidence": 0.0,
                            "description": "Not analyzed",
                            "specific_observations": "Concern not included in analysis"
                        }
                    
                    concern_data = analysis[concern]
                    if not isinstance(concern_data, dict):
                        logger.error(f"Concern data for {concern} is not a dictionary: {type(concern_data)}")
                        # Replace with default structure
                        analysis[concern] = {
                            "detected": False,
                            "severity": 0.0,
                            "confidence": 0.0,
                            "description": "Invalid data",
                            "specific_observations": "Data format error"
                        }
                
                # Validate overall_score to prevent NaN
                if 'overall_score' not in analysis:
                    logger.info("Missing overall_score in GPT response")
                    # Calculate overall score from individual concerns
                    analysis['overall_score'] = self._calculate_overall_score_from_concerns(analysis, required_concerns)
                    logger.info(f"Calculated overall_score: {analysis['overall_score']}")
                else:
                    overall_score = analysis['overall_score']
                    if not isinstance(overall_score, (int, float)) or np.isnan(overall_score):
                        logger.info(f"Invalid overall_score in GPT response: {overall_score}")
                        # Try to calculate a valid score from the concerns
                        analysis['overall_score'] = self._calculate_overall_score_from_concerns(analysis, required_concerns)
                        logger.info(f"Recalculated overall_score: {analysis['overall_score']}")
                
                # Final validation
                if not isinstance(analysis['overall_score'], (int, float)) or np.isnan(analysis['overall_score']):
                    logger.info("Final validation failed - using fallback score")
                    analysis['overall_score'] = 0.75  # Safe fallback score
                
                # ADD THIS RETURN STATEMENT - THIS IS THE FIX
                return analysis
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing GPT response: {e}")
                logger.error(f"Response was: {response.choices[0].message.content}")
                return {
                    "error": f"Failed to parse AI response: {str(e)}",
                    "overall_score": None,
                    "overall_description": "Analysis failed due to parsing error.",
                    "breakouts": {"detected": False, "severity": 0, "confidence": 0, "description": "Parsing error", "specific_observations": "JSON parsing failed"},
                    "pigmentation": {"detected": False, "severity": 0, "confidence": 0, "description": "Parsing error", "specific_observations": "JSON parsing failed"},
                    "redness": {"detected": False, "severity": 0, "confidence": 0, "description": "Parsing error", "specific_observations": "JSON parsing failed"},
                    "aging": {"detected": False, "severity": 0, "confidence": 0, "description": "Parsing error", "specific_observations": "JSON parsing failed"},
                    "dehydration": {"detected": False, "severity": 0, "confidence": 0, "description": "Parsing error", "specific_observations": "JSON parsing failed"},
                    "under_eye": {"detected": False, "severity": 0, "confidence": 0, "description": "Parsing error", "specific_observations": "JSON parsing failed"}
                }
            
        except Exception as e:
            logger.error(f"Error in GPT analysis: {e}")
            logger.info("Info: Using fallback analysis due to GPT error")
            return {
                "error": f"Analysis failed due to AI service error: {str(e)}",
                "overall_score": None,
                "overall_description": "Unable to perform skin analysis at this time.",
                "breakouts": {"detected": False, "severity": 0, "confidence": 0, "description": "Analysis unavailable", "specific_observations": "Service error"},
                "pigmentation": {"detected": False, "severity": 0, "confidence": 0, "description": "Analysis unavailable", "specific_observations": "Service error"},
                "redness": {"detected": False, "severity": 0, "confidence": 0, "description": "Analysis unavailable", "specific_observations": "Service error"},
                "aging": {"detected": False, "severity": 0, "confidence": 0, "description": "Analysis unavailable", "specific_observations": "Service error"},
                "dehydration": {"detected": False, "severity": 0, "confidence": 0, "description": "Analysis unavailable", "specific_observations": "Service error"},
                "under_eye": {"detected": False, "severity": 0, "confidence": 0, "description": "Analysis unavailable", "specific_observations": "Service error"}
            }
        

    def _analyze_skin_conditions(self, image, landmarks):
        """
        Legacy method - now using GPT-4o for analysis
        """
        # This method is kept for backward compatibility but not used
        pass

    def _extract_facial_regions(self, image, landmarks):
        """Extract different facial regions for analysis"""
        h, w = image.shape[:2]
        
        # Define landmark indices for different regions
        # These are approximate MediaPipe landmark indices
        regions = {
            "cheeks": self._get_region_landmarks(landmarks, [123, 50, 36, 137, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175], w, h),
            "forehead": self._get_region_landmarks(landmarks, [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109], w, h),
            "nose": self._get_region_landmarks(landmarks, [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175], w, h),
            "under_eyes": self._get_region_landmarks(landmarks, [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 124, 35, 111, 117, 118, 119, 120, 121, 128, 245, 188, 174, 236, 198, 209, 131, 134, 135, 136, 150, 170, 140, 141, 142, 21, 54, 103, 67, 109], w, h),
            "chin": self._get_region_landmarks(landmarks, [132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58], w, h)
        }
        
        return regions

    def _get_region_landmarks(self, landmarks, indices, w, h):
        """Extract pixel coordinates for specific landmark indices"""
        region_points = []
        for idx in indices:
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                region_points.append((x, y))
        return region_points

    def _analyze_breakouts(self, regions):
        """Analyze breakouts and acne"""
        # Simulate analysis based on texture and color variations
        severity = np.random.uniform(0.1, 0.8)  # In real implementation, use computer vision
        confidence = np.random.uniform(0.6, 0.95)
        
        return {
            "detected": bool(severity > 0.3),
            "severity": float(round(severity, 2)),
            "confidence": float(round(confidence, 2)),
            "description": "Analysis of pore congestion and inflammation"
        }

    def _analyze_pigmentation(self, regions):
        """Analyze pigmentation and dark spots"""
        severity = np.random.uniform(0.1, 0.7)
        confidence = np.random.uniform(0.7, 0.95)
        
        return {
            "detected": bool(severity > 0.25),
            "severity": float(round(severity, 2)),
            "confidence": float(round(confidence, 2)),
            "description": "Analysis of melanin distribution and dark spots"
        }

    def _analyze_redness(self, regions):
        """Analyze redness and inflammation"""
        severity = np.random.uniform(0.1, 0.6)
        confidence = np.random.uniform(0.6, 0.9)
        
        return {
            "detected": bool(severity > 0.2),
            "severity": float(round(severity, 2)),
            "confidence": float(round(confidence, 2)),
            "description": "Analysis of skin inflammation and redness"
        }

    def _analyze_aging(self, regions):
        """Analyze fine lines and wrinkles"""
        severity = np.random.uniform(0.1, 0.8)
        confidence = np.random.uniform(0.5, 0.9)
        
        return {
            "detected": bool(severity > 0.3),
            "severity": float(round(severity, 2)),
            "confidence": float(round(confidence, 2)),
            "description": "Analysis of fine lines, wrinkles, and skin texture"
        }

    def _analyze_dehydration(self, regions):
        """Analyze skin hydration levels"""
        severity = np.random.uniform(0.1, 0.7)
        confidence = np.random.uniform(0.5, 0.85)
        
        return {
            "detected": bool(severity > 0.25),
            "severity": float(round(severity, 2)),
            "confidence": float(round(confidence, 2)),
            "description": "Analysis of skin moisture and texture"
        }

    def _analyze_under_eye(self, regions):
        """Analyze under-eye concerns"""
        severity = np.random.uniform(0.1, 0.8)
        confidence = np.random.uniform(0.6, 0.9)
        
        return {
            "detected": bool(severity > 0.2),
            "severity": float(round(severity, 2)),
            "confidence": float(round(confidence, 2)),
            "description": "Analysis of dark circles and puffiness"
        }

    def _calculate_overall_score_from_concerns(self, analysis, required_concerns):
        """Calculate overall score from individual concerns with robust error handling"""
        try:
            scores = []
            weights = {
                "breakouts": 0.2,
                "pigmentation": 0.15,
                "redness": 0.15,
                "aging": 0.2,
                "dehydration": 0.15,
                "under_eye": 0.15
            }
            
            for concern in required_concerns:
                if concern in analysis and isinstance(analysis[concern], dict):
                    concern_data = analysis[concern]
                    weight = weights.get(concern, 0.15)
                    
                    # Validate concern data
                    detected = concern_data.get("detected", False)
                    severity = concern_data.get("severity", 0.0)
                    
                    # Ensure severity is a valid number
                    if not isinstance(severity, (int, float)) or np.isnan(severity):
                        severity = 0.0
                    
                    if detected and severity > 0:
                        # Lower severity = better score
                        score = max(0.0, 1.0 - severity)
                        scores.append(score * weight)
                    else:
                        scores.append(weight)  # Full points if no concern detected
                else:
                    # If concern data is missing, give neutral score
                    scores.append(weights.get(concern, 0.15))
            
            if not scores:
                return 0.75  # Default score if no valid concerns
            
            overall_score = sum(scores) / sum(weights.values())
            
            # Ensure the score is within valid range
            overall_score = max(0.1, min(0.95, overall_score))
            
            return float(round(overall_score, 2))
            
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 0.75  # Safe fallback

    def _calculate_overall_score(self, analysis):
        """Calculate overall skin health score"""
        scores = []
        weights = {
            "breakouts": 0.2,
            "pigmentation": 0.15,
            "redness": 0.15,
            "aging": 0.2,
            "dehydration": 0.15,
            "under_eye": 0.15
        }
        
        for concern, weight in weights.items():
            if analysis[concern]["detected"]:
                # Lower severity = better score
                score = 1 - analysis[concern]["severity"]
                scores.append(score * weight)
            else:
                scores.append(weight)  # Full points if no concern detected
        
        overall_score = sum(scores) / sum(weights.values())
        return float(round(overall_score, 2)) 