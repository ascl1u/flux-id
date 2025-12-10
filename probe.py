import torch
from diffusers import FluxPipeline
from transformers import BitsAndBytesConfig
from huggingface_hub import login

login()

print("Loading FluxPipeline...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer_quantization_config=quant_config,
    torch_dtype=torch.bfloat16
)

pipe.enable_sequential_cpu_offload()

class ProbeProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        print("PROBE SUCCESSFUL")
        
        # Image Stream Shape
        # at 512 x 512, we expect 1024 tokens (512 * 512 / 16 / 16)
        print(f"Stream A (Image) Input Shape: {hidden_states.shape}")
        
        # Head Dimension
        key = attn.to_k(hidden_states)
        print(f"Computed Key Shape: {key.shape}") 
        
        # RoPE Check
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            print(f"RoPE (Cos) Shape: {cos.shape}")
            print(f"RoPE (Sin) Shape: {sin.shape}")
        
        raise ValueError("Stop Generation - Probe Data Acquired")

print("Injecting probe...")
pipe.transformer.transformer_blocks[0].attn.processor = ProbeProcessor()

print("Running Pipeline (512 x 512)...")

pipe(
    prompt="test", 
    height=512, 
    width=512, 
    num_inference_steps=1,
    guidance_scale=3.5 
)