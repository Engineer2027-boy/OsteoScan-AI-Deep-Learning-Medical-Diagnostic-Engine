from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
from model import FracturePipeline

app = FastAPI(title="OsteoScan AI Engine")

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = FracturePipeline()

@app.get("/")
def health_check():
    return {"status": "online", "device": str(pipeline.device)}

@app.post("/api/predict")
async def predict_xray(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded must be an image.")
    
    try:
        start_time = time.time()
        contents = await file.read()
        
        result = pipeline.predict(contents)
        
        end_time = time.time()
        latency = int((end_time - start_time) * 1000)
        result["inference_time_ms"] = latency
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))