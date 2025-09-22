import os
import numpy as np
import cv2
import base64
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Optional, List, Dict, Any
import uvicorn
from io import BytesIO
from PIL import Image
import json

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import face_recognition with multiple fallback methods
FACE_RECOGNITION_AVAILABLE = False
face_recognition = None

def test_face_recognition():
    """Test face_recognition functionality"""
    global FACE_RECOGNITION_AVAILABLE, face_recognition
    
    try:
        # Method 1: Direct import
        import face_recognition as fr
        
        # Test if the main functions are available
        if hasattr(fr, 'face_locations') and hasattr(fr, 'face_encodings'):
            # Test with a simple image
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            test_locations = fr.face_locations(test_image)
            
            face_recognition = fr
            FACE_RECOGNITION_AVAILABLE = True
            logger.info("✅ face_recognition library loaded successfully (Method 1)")
            return True
            
    except Exception as e:
        logger.warning(f"Method 1 failed: {e}")
    
    try:
        # Method 2: Alternative import
        from face_recognition import face_locations, face_encodings
        
        # Create a mock face_recognition module
        class FaceRecognitionModule:
            def __init__(self):
                self.face_locations = face_locations
                self.face_encodings = face_encodings
        
        # Test functionality
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_locations = face_locations(test_image)
        
        face_recognition = FaceRecognitionModule()
        FACE_RECOGNITION_AVAILABLE = True
        logger.info("✅ face_recognition library loaded successfully (Method 2)")
        return True
        
    except Exception as e:
        logger.warning(f"Method 2 failed: {e}")
    
    try:
        # Method 3: Reload and reimport
        import importlib
        import sys
        
        if 'face_recognition' in sys.modules:
            importlib.reload(sys.modules['face_recognition'])
        
        import face_recognition as fr
        
        if hasattr(fr, 'face_locations'):
            face_recognition = fr
            FACE_RECOGNITION_AVAILABLE = True
            logger.info("✅ face_recognition library loaded successfully (Method 3)")
            return True
            
    except Exception as e:
        logger.warning(f"Method 3 failed: {e}")
    
    logger.error("❌ All face_recognition import methods failed")
    return False

# Initialize face_recognition
test_face_recognition()

app = FastAPI(
    title="Face Recognition Service",
    description="AI service for face recognition and verification",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class FaceEnrollRequest(BaseModel):
    user_id: str
    face_image: str

class FaceVerifyRequest(BaseModel):
    face_image: str
    stored_embedding: List[float]

class BatchVerifyRequest(BaseModel):
    face_image: str
    stored_embeddings: List[Dict[str, Any]]

class FaceDetectRequest(BaseModel):
    face_image: str

class FaceQualityRequest(BaseModel):
    face_image: str

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 image string to numpy array"""
    try:
        logger.info(f"Decoding image - input type: {type(base64_string)}, length: {len(base64_string) if base64_string else 0}")
        
        if not base64_string:
            raise ValueError("Empty image data")
        
        if not isinstance(base64_string, str):
            raise ValueError("Image data must be a string")
        
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Remove any whitespace
        base64_string = base64_string.strip()
        
        if not base64_string:
            raise ValueError("No image data after processing")
        
        # Decode base64 to bytes
        image_data = base64.b64decode(base64_string)
        
        if len(image_data) == 0:
            raise ValueError("Decoded image data is empty")
        
        # Convert to PIL Image
        pil_image = Image.open(BytesIO(image_data))
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image = np.array(pil_image)
        
        logger.info(f"Successfully decoded image - shape: {image.shape}")
        return image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        raise ValueError(f"Invalid image data: {str(e)}")

def extract_face_embedding(image: np.ndarray):
    """Extract face embedding from image"""
    try:
        if not FACE_RECOGNITION_AVAILABLE or not face_recognition:
            return None, "Face recognition library not available"
        
        logger.info("Extracting face embedding...")
        
        # Find face locations
        face_locations = face_recognition.face_locations(image, model="hog")
        
        if len(face_locations) == 0:
            return None, "No face detected in the image"
        
        if len(face_locations) > 1:
            return None, "Multiple faces detected. Please ensure only one face is visible"
        
        # Check face size
        top, right, bottom, left = face_locations[0]
        face_width = right - left
        face_height = bottom - top
        
        if face_width < 50 or face_height < 50:
            return None, "Face is too small. Please move closer to the camera"
        
        # Generate face encodings
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if len(face_encodings) == 0:
            return None, "Could not generate face encoding"
        
        logger.info("Successfully extracted face embedding")
        return face_encodings[0].tolist(), "Face processed successfully"
        
    except Exception as e:
        logger.error(f"Error extracting face embedding: {e}")
        return None, f"Error processing face: {str(e)}"

def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate similarity between two face embeddings"""
    try:
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(emb1 - emb2)
        
        # Convert distance to similarity score (0-1)
        max_distance = 0.6
        similarity = max(0, (max_distance - distance) / max_distance)
        
        return float(similarity)
        
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0

def analyze_face_quality(image: np.ndarray) -> Dict:
    """Analyze face image quality"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Sharpness score using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 500)
        
        # Brightness score
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128
        
        # Overall quality score
        quality_score = (sharpness_score + brightness_score) / 2
        
        return {
            "quality_score": float(quality_score),
            "sharpness": float(sharpness_score),
            "brightness": float(brightness_score),
            "mean_brightness": float(mean_brightness),
            "laplacian_variance": float(laplacian_var),
            "recommendation": "Good quality" if quality_score > 0.7 else "Improve lighting and focus"
        }
        
    except Exception as e:
        logger.error(f"Face quality analysis error: {e}")
        return {
            "quality_score": 0.0,
            "sharpness": 0.0,
            "brightness": 0.0,
            "recommendation": "Quality analysis failed"
        }

def perform_liveness_detection(image: np.ndarray) -> Dict:
    """Simple liveness detection based on image quality"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Check image sharpness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 100:
            return {"is_live": False, "message": "Image is too blurry"}
        
        # Check brightness
        mean_brightness = np.mean(gray)
        if mean_brightness < 50:
            return {"is_live": False, "message": "Image is too dark"}
        
        if mean_brightness > 200:
            return {"is_live": False, "message": "Image is overexposed"}
        
        # Check if image is too small
        height, width = gray.shape
        if height < 100 or width < 100:
            return {"is_live": False, "message": "Image resolution is too low"}
        
        return {"is_live": True, "message": "Liveness check passed"}
        
    except Exception as e:
        logger.error(f"Liveness detection error: {e}")
        return {"is_live": False, "message": "Liveness detection failed"}

@app.get("/")
async def root():
    return {
        "message": "Face Recognition Service is running",
        "version": "1.0.0",
        "status": "healthy" if FACE_RECOGNITION_AVAILABLE else "limited",
        "face_recognition_available": FACE_RECOGNITION_AVAILABLE,
        "endpoints": [
            "/health",
            "/detect-face",
            "/face-quality", 
            "/enroll-face",
            "/verify-face",
            "/batch-verify"
        ]
    }

@app.get("/health")
async def health_check():
    try:
        if not FACE_RECOGNITION_AVAILABLE:
            return {
                "status": "limited",
                "service": "Face Recognition API",
                "message": "Face recognition library not fully available",
                "face_recognition_ready": False,
                "note": "Service running in limited mode"
            }
        
        # Test basic functionality
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        face_locations = face_recognition.face_locations(test_image)
        
        return {
            "status": "healthy",
            "service": "Face Recognition API",
            "version": "1.0.0",
            "face_recognition_ready": True,
            "timestamp": str(np.datetime64('now'))
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "Face Recognition API",
            "error": str(e),
            "face_recognition_ready": False
        }

@app.post("/detect-face")
async def detect_face(request: FaceDetectRequest):
    try:
        if not FACE_RECOGNITION_AVAILABLE:
            return {
                "success": False,
                "faces_detected": 0,
                "face_locations": [],
                "message": "Face recognition library not available"
            }
        
        logger.info("Detecting faces in image")
        
        # Decode image
        image = decode_base64_image(request.face_image)
        
        # Find face locations
        face_locations = face_recognition.face_locations(image, model="hog")
        
        result = {
            "success": True,
            "faces_detected": len(face_locations),
            "face_locations": face_locations,
            "message": ""
        }
        
        if len(face_locations) == 0:
            result["message"] = "No face detected"
        elif len(face_locations) == 1:
            result["message"] = "Single face detected"
        else:
            result["message"] = f"{len(face_locations)} faces detected"
        
        return result
        
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return {
            "success": False,
            "faces_detected": 0,
            "face_locations": [],
            "message": f"Face detection failed: {str(e)}"
        }

@app.post("/face-quality")
async def analyze_face_quality_endpoint(request: FaceQualityRequest):
    try:
        logger.info(f"Face quality request - has image: {bool(request.face_image)}")
        
        if not request.face_image:
            return {
                "success": False,
                "quality": {
                    "quality_score": 0.0,
                    "sharpness": 0.0,
                    "brightness": 0.0,
                    "recommendation": "No image provided"
                },
                "liveness": {
                    "is_live": False,
                    "message": "No image provided"
                },
                "message": "Face image is required"
            }
        
        logger.info("Analyzing face image quality")
        
        # Decode image
        image = decode_base64_image(request.face_image)
        
        # Analyze quality
        quality_info = analyze_face_quality(image)
        
        # Perform liveness detection
        liveness_info = perform_liveness_detection(image)
        
        return {
            "success": True,
            "quality": quality_info,
            "liveness": liveness_info,
            "message": "Quality analysis completed"
        }
        
    except Exception as e:
        logger.error(f"Face quality analysis error: {e}")
        return {
            "success": False,
            "quality": {
                "quality_score": 0.0,
                "sharpness": 0.0,
                "brightness": 0.0,
                "recommendation": "Failed to analyze image quality"
            },
            "liveness": {
                "is_live": False,
                "message": "Quality analysis failed"
            },
            "message": f"Quality analysis failed: {str(e)}"
        }

@app.post("/enroll-face")
async def enroll_face(request: FaceEnrollRequest):
    try:
        logger.info(f"Face enrollment request - user: {request.user_id}, has image: {bool(request.face_image)}")
        
        if not request.face_image:
            return {
                "success": False,
                "message": "Face image is required"
            }
        
        if not FACE_RECOGNITION_AVAILABLE:
            return {
                "success": False,
                "message": "Face recognition service not available"
            }
        
        logger.info(f"Enrolling face for user: {request.user_id}")
        
        # Decode image
        image = decode_base64_image(request.face_image)
        
        # Perform quality checks
        quality_info = analyze_face_quality(image)
        if quality_info["quality_score"] < 0.3:  # Lower threshold for enrollment
            return {
                "success": False,
                "message": f"Image quality too low. {quality_info['recommendation']}"
            }
        
        # Perform liveness detection
        liveness_info = perform_liveness_detection(image)
        if not liveness_info["is_live"]:
            return {
                "success": False,
                "message": liveness_info["message"]
            }
        
        # Extract face embedding
        embedding, message = extract_face_embedding(image)
        
        if embedding is None:
            return {"success": False, "message": message}
        
        logger.info(f"Successfully enrolled face for user: {request.user_id}")
        
        return {
            "success": True,
            "message": "Face enrolled successfully",
            "embedding": embedding,
            "quality": quality_info
        }
        
    except Exception as e:
        logger.error(f"Face enrollment error: {e}")
        return {
            "success": False,
            "message": f"Face enrollment failed: {str(e)}"
        }

@app.post("/verify-face")
async def verify_face(request: FaceVerifyRequest):
    try:
        logger.info("Performing face verification")
        
        if not FACE_RECOGNITION_AVAILABLE:
            return {
                "success": False,
                "message": "Face recognition service not available",
                "confidence": 0.0
            }
        
        # Decode image
        image = decode_base64_image(request.face_image)
        
        # Perform liveness detection
        liveness_info = perform_liveness_detection(image)
        if not liveness_info["is_live"]:
            return {
                "success": False,
                "message": liveness_info["message"],
                "confidence": 0.0
            }
        
        # Extract embedding from current image
        current_embedding, message = extract_face_embedding(image)
        
        if current_embedding is None:
            return {"success": False, "message": message, "confidence": 0.0}
        
        # Calculate similarity
        confidence = calculate_similarity(current_embedding, request.stored_embedding)
        
        success = confidence >= 0.7
        result_message = "Face verified successfully" if success else "Face verification failed"
        
        logger.info(f"Face verification completed. Success: {success}, Confidence: {confidence:.3f}")
        
        return {
            "success": success,
            "message": result_message,
            "confidence": confidence
        }
        
    except Exception as e:
        logger.error(f"Face verification error: {e}")
        return {
            "success": False,
            "message": "Face verification failed",
            "confidence": 0.0
        }

@app.post("/batch-verify")
async def batch_verify_face(request: BatchVerifyRequest):
    try:
        logger.info("Performing batch face verification")
        
        if not FACE_RECOGNITION_AVAILABLE:
            return {"success": False, "message": "Face recognition service not available"}
        
        # Decode image
        image = decode_base64_image(request.face_image)
        
        # Perform liveness detection
        liveness_info = perform_liveness_detection(image)
        if not liveness_info["is_live"]:
            return {"success": False, "message": liveness_info["message"]}
        
        # Extract embedding from current image
        current_embedding, message = extract_face_embedding(image)
        
        if current_embedding is None:
            return {"success": False, "message": message}
        
        # Find best match
        best_match = None
        best_confidence = 0.0
        
        for stored_data in request.stored_embeddings:
            confidence = calculate_similarity(current_embedding, stored_data['embedding'])
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = {
                    "user_id": stored_data['user_id'],
                    "confidence": confidence
                }
        
        success = best_confidence >= 0.7
        
        return {
            "success": success,
            "best_match": best_match,
            "total_checked": len(request.stored_embeddings),
            "message": "Best match found" if success else "No matching face found"
        }
        
    except Exception as e:
        logger.error(f"Batch verification error: {e}")
        return {"success": False, "message": "Batch verification failed"}

@app.post("/upload-face")
async def upload_face_file(file: UploadFile = File(...)):
    """Upload face image file for processing"""
    try:
        if not FACE_RECOGNITION_AVAILABLE:
            return {
                "success": False,
                "message": "Face recognition service not available"
            }
        
        # Read file content
        content = await file.read()
        
        # Convert to PIL Image
        pil_image = Image.open(BytesIO(content))
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image = np.array(pil_image)
        
        # Detect faces
        face_locations = face_recognition.face_locations(image)
        
        # Analyze quality
        quality_info = analyze_face_quality(image)
        
        # Perform liveness detection
        liveness_info = perform_liveness_detection(image)
        
        return {
            "success": True,
            "faces_detected": len(face_locations),
            "quality": quality_info,
            "liveness": liveness_info,
            "message": "File processed successfully"
        }
        
    except Exception as e:
        logger.error(f"File upload error: {e}")
        return {
            "success": False,
            "message": "File processing failed"
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting Face Recognition Service on {host}:{port}")
    logger.info(f"Face Recognition Available: {FACE_RECOGNITION_AVAILABLE}")
    
    if FACE_RECOGNITION_AVAILABLE:
        logger.info("✅ Face recognition fully functional")
    else:
        logger.warning("⚠️ Face recognition running in limited mode")
    
    logger.info("Available endpoints:")
    logger.info("  GET  /health - Health check")
    logger.info("  POST /detect-face - Detect faces in image")
    logger.info("  POST /face-quality - Analyze face image quality")
    logger.info("  POST /enroll-face - Enroll face")
    logger.info("  POST /verify-face - Verify face")
    logger.info("  POST /batch-verify - Batch face verification")
    logger.info("  POST /upload-face - Upload face file")
    
    uvicorn.run(
        "app:app", 
        host=host, 
        port=port,
        reload=True,
        log_level="info"
    )
