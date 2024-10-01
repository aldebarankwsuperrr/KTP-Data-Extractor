import cv2
import torch
import numpy as np

def ktp_wrapped(image, mask_top_left, mask_top_right, mask_bottom_left, mask_bottom_right, width=700, height=500):
    # Define transformation points
    converted_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    point_matrix = np.float32([mask_top_left, mask_top_right, mask_bottom_left, mask_bottom_right])
    perspective_transform = cv2.getPerspectiveTransform(point_matrix, converted_points)

    # Apply perspective transform
    wrapped_img = cv2.warpPerspective(image, perspective_transform, (width, height))
    
    return wrapped_img

def preprocess_wrapped_image(processor, wrapped_img):
    pixel_values = processor(wrapped_img, return_tensors="pt").pixel_values.squeeze()
    pixel_values = torch.tensor(pixel_values).unsqueeze(0)
    task_prompt = "<s>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    return pixel_values, decoder_input_ids

def generate_predictions(model, processor, pixel_values, decoder_input_ids, device="cpu"):
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    
    prediction = processor.batch_decode(outputs.sequences)[0]
    prediction = processor.token2json(prediction)
    
    return prediction
