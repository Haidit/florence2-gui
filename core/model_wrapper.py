import torch
from PIL import Image, ImageDraw
from config import VISUAL_TASKS

class Florence2ModelWrapper:
    def __init__(self, model_data):
        self.model = model_data['model']
        self.processor = model_data['processor']
        self.device = model_data['device']
        self.torch_dtype = model_data['torch_dtype']
        self.model.eval()
        
    def process_image(self, image_pil, task_tag, prompt, generation_params):
        try:
            if not prompt.startswith(task_tag):
                prompt = f"{task_tag}{prompt}"
            
            inputs = self.processor(
                text=prompt,
                images=image_pil,
                return_tensors="pt",
                padding=True
            ).to(self.device, self.torch_dtype)
            
            if self.torch_dtype == torch.float16:
                inputs = {k: v.half() if v.is_floating_point() else v for k, v in inputs.items()}
            elif self.torch_dtype == torch.bfloat16:
                inputs = {k: v.bfloat16() if v.is_floating_point() else v for k, v in inputs.items()}
            else:
                inputs = {k: v.float() if v.is_floating_point() else v for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    use_cache=False,
                    output_scores=True,
                    min_new_tokens=generation_params.get('min_new_tokens', 512),
                    max_new_tokens=generation_params.get('max_new_tokens', 512),
                    num_beams=generation_params.get('num_beams', 1),
                    temperature=generation_params.get('temperature', 1.0),
                    top_k=generation_params.get('top_k', 50),
                    top_p=generation_params.get('top_p', 0.9),
                    early_stopping=generation_params.get('early_stopping', False),
                    do_sample=generation_params.get('do_sample', False),
                    no_repeat_ngram_size=generation_params.get('no_repeat_ngram_size', 0),
                    length_penalty=generation_params.get('length_penalty', 1.0),
                    repetition_penalty=generation_params.get('repetition_penalty', 1.0),
                )
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            is_visual_task = task_tag in VISUAL_TASKS

            if is_visual_task:
                processed_results = self.processor.post_process_generation(
                    generated_text,
                    task=task_tag,
                    image_size=image_pil.size
                )
                
                # mask_img = self._create_mask_image(processed_results, image_pil.size)
                
                return {
                    'text': generated_text,
                    'task_tag': task_tag,
                    'is_visual_task': True,
                    'processed_results': processed_results,
                    # 'mask_image': mask_img,
                    'success': True
                }
            return {
                'text': generated_text.replace(task_tag, "").strip(),
                'task_tag': task_tag,
                'is_visual_task': False,
                'success': True
            }
            
        except Exception as e:
            raise Exception(f"Processing error: {str(e)}")
        
    def _create_mask_image(self, processed_results, image_size):
        mask = Image.new('L', image_size, 0)
        draw = ImageDraw.Draw(mask)
        
        if not processed_results or not next(iter(processed_results.values())):
            return mask
        
        result_data = next(iter(processed_results.values()))
        
        if 'bboxes' in result_data:
            for bbox in result_data['bboxes']:
                x1, y1, x2, y2 = map(int, bbox)
                draw.rectangle([x1, y1, x2, y2], fill=255)
        
        if 'polygons' in result_data:
            for polygon in result_data['polygons']:
                try:
                    flat_polygon = []
                    for point in polygon:
                        if isinstance(point, (list, tuple)):
                            flat_polygon.extend(map(float, point))
                        else:
                            flat_polygon.append(float(point))
                    
                    if len(flat_polygon) >= 6:
                        draw.polygon(flat_polygon, fill=255)
                    else:
                        print(f"Ignoring invalid polygon with {len(flat_polygon)//2} points")
                except Exception as e:
                    print(f"Error drawing polygon: {e}")
                    continue
        
        return mask
    