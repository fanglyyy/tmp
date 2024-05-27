export HF_ENDPOINT=https://hf-mirror.com

# dataset
huggingface-cli download --resume-download --repo-type dataset truthful_qa --local-dir /root/autodl-tmp/MLLM/Invention/NL-ITI/datasets

# model
# huggingface-cli download --resume-download daryl149/llama-2-7b-chat-hf --local-dir /root/autodl-tmp/LLM/llama/llama-2-7b-chat-hf --local-dir-use-symlinks False --resume