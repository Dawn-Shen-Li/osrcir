import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import time
import random
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from accelerate import infer_auto_device_map

def generate_CoT_with_hidden_states(sys_prompt, user_prompt, image_path, max_new_tokens=512, temperature=0.0):
    image = Image.open(image_path).convert("RGB")
    prompt = f"{sys_prompt}\n{user_prompt}"

    # Process input
    inputs = processor(prompt, images=image, return_tensors="pt").to(device, torch.float16)

    # Optional: register hook for hidden states
    hidden_states = []

    def hook(module, input, output):
        hidden_states.append(output)

    hook_handle = model.model.layers[-1].register_forward_hook(hook)

    # Generate output (CoT)
    with torch.no_grad():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, temperature=temperature)

    # Decode output
    answer = processor.batch_decode(output, skip_special_tokens=True)[0]

    # Detach hook
    hook_handle.remove()

    print("CoT Output:\n", answer)
    return answer, hidden_states[-1] if hidden_states else None


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(60))
def hf_completion_vision_CoT(sys_prompt, user_prompt, image_path, model, processor, max_new_tokens=512, temperature=0.0):
    global_attempt, local_attempt = 0, 0
    global_max_attempts, local_max_attempts = 2, 3

    while global_attempt < global_max_attempts:
        try:
            try:
                return attempt_hf_completion_CoT(sys_prompt, user_prompt, image_path, model, processor, max_new_tokens, temperature)
            except Exception as e:
                local_attempt += 1
                if local_attempt < local_max_attempts:
                    wait_time = random.randint(1, 60)
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"[Local Fail] {str(e)} — switching to fallback model (not implemented here).")
                    return attempt_hf_completion_CoT(sys_prompt, user_prompt, image_path, model, processor, max_new_tokens, temperature)
        except Exception as e:
            global_attempt += 1
            if global_attempt == global_max_attempts:
                print(f"[Global Fail] {str(e)} — returning empty output.")
                return "", None


def attempt_hf_completion_CoT(sys_prompt, user_prompt, image_path, model, processor, max_new_tokens=512, temperature=0.0):
    image = Image.open(image_path).convert("RGB")
    prompt = f"{sys_prompt}\n{user_prompt}"

    # Preprocess
    inputs = processor(prompt, images=image, return_tensors="pt").to(model.device, torch.float16)

    # Register hidden state hook
    hidden_states = []
    def hook(module, input, output):
        hidden_states.append(output)
    
    handle = model.model.layers[-1].register_forward_hook(hook)

    # Generate output
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=temperature > 0,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

    # Decode output
    decoded = processor.batch_decode(output, skip_special_tokens=True)[0]

    handle.remove()  # cleanup
    print("Model Output:", decoded)

    return decoded, hidden_states[-1] if hidden_states else None

if __name__ == "__main__":
    
    # Choose your model: "llava-hf/llava-1.5-7b-hf" or "Qwen/Qwen-VL-Chat"
    
    model_id = "llava-hf/llava-1.5-7b-hf"  # or "Qwen/Qwen-VL-Chat"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",                 # automatic device placement
        torch_dtype="auto"                # uses fp16/bf16 if available).to(device).eval()
    )
    
    #device_map = infer_auto_device_map(model, max_memory={i: "24GiB" for i in range(torch.cuda.device_count())})
    #print(device_map)    
    processor = AutoProcessor.from_pretrained(model_id)

    sys_prompt = "You are an image description expert."
    user_prompt = "Describe the image and provide a chain of thought."
    image_path = "path_to_your_image.jpg"  # Replace with your image path
    
    cot_output, hidden = hf_completion_vision_CoT(sys_prompt, user_prompt, image_path, model, processor)
    print("Final CoT Output:", cot_output)
    if hidden is not None:
        print("Last hidden state shape:", hidden.shape)