from huggingface_hub import snapshot_download
import os

print("⏳ Downloading CLIP model (approx 1.7 GB)... please wait.")

# Download strictly the pytorch weights
path = snapshot_download(
    repo_id="openai/clip-vit-large-patch14-336",
    local_dir="clip-encoder",
    local_dir_use_symlinks=False,
    ignore_patterns=["*.safetensors", "*.h5", "*.msgpack"] # Force .bin file
)

print(f"✅ Download complete! Files saved to: {path}")

# Verify size
bin_file = os.path.join("clip-encoder", "pytorch_model.bin")
if os.path.exists(bin_file):
    size_gb = os.path.getsize(bin_file) / (1024 * 1024 * 1024)
    print(f"📦 File Size: {size_gb:.2f} GB (Should be ~1.71 GB)")
else:
    print("❌ Error: pytorch_model.bin not found!")