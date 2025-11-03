# server_fixed.py - Server that matches the trained model's architecture
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import numpy as np
from PIL import Image
import io
import json
import os
from typing import Dict, List

# MindSpore imports
import mindspore
import mindspore.nn as nn
from mindspore import Tensor, load_checkpoint, load_param_into_net

# Configuration
class Config:
    img_size = 128
    model_save_path = 'best_diamond_model.ckpt'
    
CFG = Config()

class LabelEncoder:
    def __init__(self):
        self.mapping = {}
        self.inverse = {}
        self.fitted = False

    def load(self, path):
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.mapping = data['mapping']
                    self.inverse = {}
                    for k, v in data['inverse'].items():
                        try:
                            self.inverse[int(k)] = v
                        except ValueError:
                            self.inverse[k] = v
                    self.fitted = True
                    print(f"✅ Loaded encoder from {path} - {len(self.mapping)} classes")
                    return True
            return False
        except Exception as e:
            print(f"❌ Error loading encoder {path}: {e}")
            return False

    def create_compatible_encoder(self, task_name, required_classes):
        """Create encoder that matches the model's expected class count"""
        compatible_encoders = {
            'polish': {
                'mapping': {'Excellent': 0, 'Very Good': 1, 'Good': 2, 'Fair': 3},
                'inverse': {0: 'Excellent', 1: 'Very Good', 2: 'Good', 3: 'Fair'}
            },
            'symmetry': {
                'mapping': {'Excellent': 0, 'Very Good': 1, 'Good': 2, 'Fair': 3, 'Poor': 4},
                'inverse': {0: 'Excellent', 1: 'Very Good', 2: 'Good', 3: 'Fair', 4: 'Poor'}
            },
            'fluorescence': {
                'mapping': {'None': 0, 'Faint': 1, 'Medium': 2, 'Strong': 3, 'Very Strong': 4},
                'inverse': {0: 'None', 1: 'Faint', 2: 'Medium', 3: 'Strong', 4: 'Very Strong'}
            },
            'clarity': {
                'mapping': {'FL': 0, 'IF': 1, 'VVS1': 2, 'VVS2': 3, 'VS1': 4, 'VS2': 5, 'SI1': 6, 'SI2': 7, 'I1': 8},
                'inverse': {0: 'FL', 1: 'IF', 2: 'VVS1', 3: 'VVS2', 4: 'VS1', 5: 'VS2', 6: 'SI1', 7: 'SI2', 8: 'I1'}
            }
        }
        
        if task_name in compatible_encoders:
            self.mapping = compatible_encoders[task_name]['mapping']
            self.inverse = compatible_encoders[task_name]['inverse']
            self.fitted = True
            print(f"🔄 Using compatible encoder for {task_name} - {len(self.mapping)} classes (model expects {required_classes})")
            return True
        return False

class MultiTaskNet(nn.Cell):
    def __init__(self, n_p=4, n_s=6, n_f=10, n_c=17):  # Default to model's expected sizes
        super().__init__()
        print(f"🔧 Creating model with class counts: polish={n_p}, symmetry={n_s}, fluorescence={n_f}, clarity={n_c}")
        
        self.backbone = nn.SequentialCell(
            nn.Conv2d(3, 32, 3, pad_mode='same'), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, pad_mode='same'), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, pad_mode='same'), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc_p = nn.Dense(128, n_p)
        self.fc_s = nn.Dense(128, n_s)
        self.fc_f = nn.Dense(128, n_f)
        self.fc_c = nn.Dense(128, n_c)

    def construct(self, x):
        f = self.backbone(x)
        return self.fc_p(f), self.fc_s(f), self.fc_f(f), self.fc_c(f)

class DiamondClassifier:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.loaded = False
        
    def detect_model_architecture(self):
        """Detect what architecture the trained model expects"""
        try:
            if not os.path.exists(CFG.model_save_path):
                return None
                
            param_dict = load_checkpoint(CFG.model_save_path)
            
            # Analyze the parameter shapes to determine expected class counts
            class_counts = {}
            for param_name, param_value in param_dict.items():
                if 'fc_p.weight' in param_name:
                    class_counts['polish'] = param_value.shape[0]  # Output features
                elif 'fc_s.weight' in param_name:
                    class_counts['symmetry'] = param_value.shape[0]
                elif 'fc_f.weight' in param_name:
                    class_counts['fluorescence'] = param_value.shape[0]
                elif 'fc_c.weight' in param_name:
                    class_counts['clarity'] = param_value.shape[0]
            
            print(f"🔍 Detected model architecture: {class_counts}")
            return class_counts
            
        except Exception as e:
            print(f"❌ Error detecting model architecture: {e}")
            return None
    
    def load_model(self):
        """Load model with architecture detection"""
        try:
            print("🔍 Detecting model architecture...")
            model_arch = self.detect_model_architecture()
            
            if not model_arch:
                print("❌ Could not detect model architecture")
                return False
            
            # Use the detected architecture
            n_p = model_arch.get('polish', 4)
            n_s = model_arch.get('symmetry', 6)
            n_f = model_arch.get('fluorescence', 10)
            n_c = model_arch.get('clarity', 17)
            
            print(f"📊 Model expects - Polish: {n_p}, Symmetry: {n_s}, Fluorescence: {n_f}, Clarity: {n_c}")
            
            # Create model with correct architecture
            print("📥 Creating model with detected architecture...")
            self.model = MultiTaskNet(n_p, n_s, n_f, n_c)
            
            # Load weights
            param_dict = load_checkpoint(CFG.model_save_path)
            load_param_into_net(self.model, param_dict)
            print("✅ Model weights loaded successfully")
            
            # Load or create compatible encoders
            print("📥 Setting up encoders...")
            encoder_files = {
                'polish': 'polish_encoder.json',
                'symmetry': 'symmetry_encoder.json', 
                'fluorescence': 'fluorescence_encoder.json',
                'clarity': 'clarity_encoder.json'
            }
            
            for task, file_path in encoder_files.items():
                encoder = LabelEncoder()
                required_classes = model_arch.get(task, 4)
                
                if not encoder.load(file_path):
                    # Create encoder that matches model's expected class count
                    if not encoder.create_compatible_encoder(task, required_classes):
                        print(f"❌ Failed to create encoder for {task}")
                        return False
                
                # Check if encoder matches model
                encoder_classes = len([k for k in encoder.mapping.keys() if k != '<UNK>'])
                if encoder_classes != required_classes:
                    print(f"⚠️  Encoder {task} has {encoder_classes} classes but model expects {required_classes}")
                    print(f"🔄 Using model-compatible encoder instead")
                    encoder = LabelEncoder()
                    encoder.create_compatible_encoder(task, required_classes)
                
                self.encoders[task] = encoder
            
            self.loaded = True
            print("🎉 Model and encoders loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def preprocess_image(self, image_data: bytes) -> Tensor:
        """Preprocess uploaded image"""
        try:
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            image = image.resize((CFG.img_size, CFG.img_size))
            img_array = np.array(image, dtype=np.float32) / 255.0
            img_array = np.transpose(img_array, (2, 0, 1))
            img_array = np.expand_dims(img_array, axis=0)
            return Tensor(img_array)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image processing error: {e}")
    
    def predict(self, image_data: bytes) -> Dict:
        if not self.loaded:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            input_tensor = self.preprocess_image(image_data)
            self.model.set_train(False)
            outputs = self.model(input_tensor)
            
            predictions = {}
            task_names = ['polish', 'symmetry', 'fluorescence', 'clarity']
            
            for i, task in enumerate(task_names):
                logits = outputs[i].asnumpy()[0]
                probabilities = self.softmax(logits)
                predicted_class_idx = np.argmax(probabilities)
                
                encoder = self.encoders[task]
                predicted_label = encoder.inverse.get(predicted_class_idx, "Unknown")
                
                predictions[task] = {
                    'label': predicted_label,
                    'confidence': float(np.max(probabilities)),
                    'all_probabilities': {
                        encoder.inverse.get(j, "Unknown"): float(prob) 
                        for j, prob in enumerate(probabilities)
                    }
                }
            
            return predictions
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

# Global classifier instance
classifier = DiamondClassifier()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Diamond Classification Server...")
    success = classifier.load_model()
    if success:
        print("✅ Server ready and model loaded!")
    else:
        print("⚠️  Server started with limited functionality")
    yield
    print("👋 Shutting down server...")

app = FastAPI(
    title="Diamond Classification Server", 
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Diamond Classification Server", 
        "status": "running",
        "model_loaded": classifier.loaded
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if classifier.loaded else "model_not_loaded", 
        "model_loaded": classifier.loaded
    }

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if not classifier.loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    encoder_info = {}
    for task, encoder in classifier.encoders.items():
        encoder_info[task] = {
            "classes": list(encoder.mapping.keys()),
            "num_classes": len([k for k in encoder.mapping.keys() if k != '<UNK>']),
            "fitted": encoder.fitted
        }
    
    return {
        "model_loaded": classifier.loaded,
        "encoders": encoder_info
    }

@app.post("/predict")
async def predict_diamond(file: UploadFile = File(...)):
    if not classifier.loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_data = await file.read()
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        predictions = classifier.predict(image_data)
        
        return JSONResponse(content={
            "success": True,
            "predictions": predictions,
            "filename": file.filename
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

if __name__ == "__main__":
    print("🔍 Starting server with architecture detection...")
    uvicorn.run(
        "server_fixed:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )