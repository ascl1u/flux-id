import torch
import torch.nn.functional as F
import gc
from diffusers import FluxPipeline
from transformers import BitsAndBytesConfig
from diffusers.models.embeddings import apply_rotary_emb
from huggingface_hub import login

REF_CACHE = { "keys": None, "values": None, "mode": "OFF" }
gc.collect()
torch.cuda.empty_cache()
try: login(new_session=False)
except: pass

class FluxZeroRopeProcessor:
    def __init__(self):
        self.num_heads = 24
        self.head_dim = 128

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        # image stream (1, 1024, 3072)
        batch_size, img_seq_len, _ = hidden_states.shape
        img_q = attn.to_q(hidden_states)
        img_k = attn.to_k(hidden_states)
        img_v = attn.to_v(hidden_states)

        # text stream (1, 512, 3072)
        txt_seq_len = encoder_hidden_states.shape[1]
        txt_q = attn.add_q_proj(encoder_hidden_states)
        txt_k = attn.add_k_proj(encoder_hidden_states)
        txt_v = attn.add_v_proj(encoder_hidden_states)

        # concatenate joint cross-attention (1, 1536, 3072)
        query = torch.cat([img_q, txt_q], dim=1)
        key = torch.cat([img_k, txt_k], dim=1)
        value = torch.cat([img_v, txt_v], dim=1)

        # reshape for multi-head attention (1, 24, 1536, 128)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        if REF_CACHE["mode"] == "WRITE":
            # cache only visual tokens (1, 24, 1024, 128)
            visual_key = key[:, :, :img_seq_len, :]
            visual_val = value[:, :, :img_seq_len, :]
            
            print(f"  [WRITE] Caching RAW Keys (Zero-RoPE): {visual_key.shape}")
            REF_CACHE["keys"] = visual_key.detach().clone()
            REF_CACHE["values"] = visual_val.detach().clone()

        if image_rotary_emb is not None:
            if isinstance(image_rotary_emb, (list, tuple)) and len(image_rotary_emb) == 1:
                image_rotary_emb = image_rotary_emb[0]
            
            # apply RoPE to embeddings
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if REF_CACHE["mode"] == "READ":
            ref_k = REF_CACHE["keys"]
            ref_v = REF_CACHE["values"]
            
            if ref_k is not None:
                if ref_k.shape[0] != batch_size:
                    ref_k = ref_k.repeat(batch_size, 1, 1, 1)
                    ref_v = ref_v.repeat(batch_size, 1, 1, 1)

                # concatenate cached key value (1, 24, 2560, 128)
                key = torch.cat([key, ref_k], dim=2)
                value = torch.cat([value, ref_v], dim=2)
                
                if batch_size == 1: 
                    print(f"  [READ] Injection Active. Mixing Spatial + Global. Total: {key.shape[2]}")

        # Q * K^T (1, 24, 1536, 128)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )

        # (1, 1536, 3072)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        # split joint sequence into image (1024) and text (512)
        img_out, txt_out = hidden_states.split([img_seq_len, txt_seq_len], dim=1)
        img_out = attn.to_out[0](img_out)
        txt_out = attn.to_add_out(txt_out)
        
        return img_out, txt_out

# --- 3. EXECUTION ---
print(">>> Loading Pipeline...")
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", transformer_quantization_config=quant_config, torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()

print(">>> Installing Zero-RoPE Processor...")
pipe.transformer.transformer_blocks[0].attn.processor = FluxZeroRopeProcessor()

# PASS 1: CAT (Reference)
print("\n>>> PASS 1 (REFERENCE)")
REF_CACHE["mode"] = "WRITE"
pipe("A fluffy white cat", height=512, width=512, num_inference_steps=4, guidance_scale=3.5)

# PASS 2: DOG (Target)
if REF_CACHE["keys"] is not None:
    print("\n>>> PASS 2 (TARGET)")
    REF_CACHE["mode"] = "READ"
    result = pipe("A golden retriever", height=512, width=512, num_inference_steps=4, guidance_scale=3.5).images[0]
    result.save("zero_rope.png")
    print("\n>>> DONE. Check 'zero_rope.png'")