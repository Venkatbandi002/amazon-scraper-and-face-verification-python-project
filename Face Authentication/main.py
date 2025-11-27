from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer
from fastapi.middleware.cors import CORSMiddleware
import io

app = FastAPI(
    title="Face Verification API (keras-facenet)",
    description="Verify whether two face images belong to the same person using keras-facenet embeddings.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = MTCNN()
embedder = FaceNet()
l2_normalizer = Normalizer(norm="l2")


def read_image_from_upload(file: UploadFile) -> np.ndarray:

    image_bytes = file.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    # Decode image bytes to BGR (OpenCV default)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not read image file")

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def get_face_boxes_and_embeddings(img_rgb: np.ndarray):

    # MTCNN expects RGB image as numpy array
    detections = detector.detect_faces(img_rgb)

    if detections is None or len(detections) == 0:
        return [], None, []
    
    boxes = []
    probs = []
    face_crops = []

    for det in detections:
        x, y, w, h = det["box"]
        confidence = det.get("confidence", 0.0)

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = max(0, x + w)
        y2 = max(0, y + h)

        # Crop face
        face = img_rgb[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # Resize to 160x160 as expected by FaceNet
        face = cv2.resize(face, (160, 160))

        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        probs.append(float(confidence))
        face_crops.append(face)

    if len(face_crops) == 0:
        return [], None, []

    # Compute embeddings
    face_crops = np.asarray(face_crops)
    embeddings = embedder.embeddings(face_crops)

    return boxes, embeddings, probs

def get_main_face_embedding(img_rgb: np.ndarray):

    boxes, embeddings, probs = get_face_boxes_and_embeddings(img_rgb)

    if embeddings is None or len(boxes) == 0:
        return None, [], []

    probs_np = np.array(probs)
    best_idx = int(probs_np.argmax())

    best_emb = embeddings[best_idx]  # shape: (D,)
    best_emb_norm = l2_normalizer.transform(best_emb.reshape(1, -1))[0]

    return best_emb_norm, boxes, probs


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return float(np.dot(emb1, emb2))

@app.post("/verify-faces")
async def verify_faces(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    """
    Accept two images, detect faces, extract embeddings,
    compute similarity, and return whether they are the same person.
    """
    # Basic file type check
    valid_types = {"image/jpeg", "image/png", "image/jpg"}
    if image1.content_type not in valid_types:
        raise HTTPException(status_code=400, detail="image1 must be JPEG or PNG")
    if image2.content_type not in valid_types:
        raise HTTPException(status_code=400, detail="image2 must be JPEG or PNG")

    # Read both images as RGB numpy arrays
    img1 = read_image_from_upload(image1)
    img2 = read_image_from_upload(image2)

    # Get main face embedding + all boxes
    emb1, boxes1, probs1 = get_main_face_embedding(img1)
    emb2, boxes2, probs2 = get_main_face_embedding(img2)

    if emb1 is None:
        raise HTTPException(status_code=400, detail="No face detected in image1")
    if emb2 is None:
        raise HTTPException(status_code=400, detail="No face detected in image2")

    # Similarity and decision
    similarity = compute_cosine_similarity(emb1, emb2)

    # Threshold for deciding "same person" (tune as needed)
    THRESHOLD = 0.7

    if similarity >= THRESHOLD:
        result = "Same person"
    else:
        result = "Different person"

    # Build response
    response_data = {
        "verification_result": result,
        "similarity_score": similarity,
        "image1": {
            "num_faces": len(boxes1),
            "bounding_boxes": boxes1,
            "detection_probs": probs1,
        },
        "image2": {
            "num_faces": len(boxes2),
            "bounding_boxes": boxes2,
            "detection_probs": probs2,
        },
        "threshold_used": THRESHOLD,
        "model": "FaceNet (keras-facenet) + MTCNN",
    }

    return JSONResponse(content=response_data)

@app.get("/")
def root():
    return {
        "message": "Face Verification API (keras-facenet) is running.",
        "usage": "Go to /docs to test the /verify-faces endpoint.",
    }
