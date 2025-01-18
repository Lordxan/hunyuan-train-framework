import os
from glob import iglob
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

def describe_images(folder_path, prompt, image_glob_pattern, processor, model):
    for image_path in iglob(image_glob_pattern, recursive=True):
        try:
            print(f"Processing image: {image_path}")
            image = Image.open(image_path)
        
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = inputs.to(device)

            outputs = model.generate(**inputs, max_new_tokens=500)
            
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
            
            description = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            txt_file_path = os.path.splitext(image_path)[0] + '.txt'
            
            with open(txt_file_path, 'w') as txt_file:
                txt_file.write(description)
        except Exception as e:
            print(f"An error occurred while processing {image_path}: {e}")
