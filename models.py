from ultralytics import YOLO
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch

def load_models(segment_model_dir, obb_model_dir, processor_dir, donut_model_dir):
    # Move model to GPU
    segment_model = YOLO(segment_model_dir)
    obb_model = YOLO(obb_model_dir)
    processor = DonutProcessor.from_pretrained(processor_dir)
    model = VisionEncoderDecoderModel.from_pretrained(donut_model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    return segment_model, obb_model, processor, model
