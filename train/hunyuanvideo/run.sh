docker run -ti --rm --gpus all \
  -v ~/comfyui/models/vae/hunyuan_video_vae_bf16.safetensors:/hunyuan_video_vae_bf16.safetensors:ro \
  -v ~/comfyui/models/LLM/llava-llama-3-8b-text-encoder-tokenizer:/llava-llama-3-8b-text-encoder-tokenizer:ro \
  -v ~/comfyui/models/clip/clip-vit-large-patch14:/clip-vit-large-patch14:ro \
  -v ~/comfyui/models/diffusion_models/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors:/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors:ro \
  -v .:/config \
  $(docker build . -q -t hunyuan-train)
