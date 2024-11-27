import os
os.environ["HF_HOME"] = "/mnt/ssd2/lmm/huggingface/"
import safetensors
import torch

from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM, LlavaMambaConfig
from llava.train.train import ModelArguments

model_path = "/mnt/ssd2/lmm/huggingface/hub/LongMamba-Stage2-1.3b"
train_args_file = os.path.join(model_path, "training_args.bin")
safe_weights_file = os.path.join(model_path, "model.safetensors")

train_args = torch.load(train_args_file)
state_dict = safetensors.torch.load_file(safe_weights_file, device="cpu")


model_args = ModelArguments(
    model_name_or_path="state-spaces/mamba2-1.3b",
    version="plain",
    vision_tower="/mnt/ssd2/lmm/huggingface/hub/clip-vit-large-patch14-336",
    pretrain_mm_mlp_adapter="/mnt/ssd2/lmm/huggingface/hub/LongMamba-Stage1-1.3b-1007/mm_projector.bin",
    mm_tunable_parts="mm_mlp_adapter,mm_language_model",
    mm_vision_select_layer=-2,
    mm_projector_type="mlp2x_gelu",
    mm_use_im_start_end=False,
    mm_use_im_patch_token=False,
    mm_spatial_pool_mode="bilinear"
)

model_name = "state-spaces/mamba2-1.3b"
model = LlavaQwenForCausalLM.from_pretrained(model_name, "cpu", config=model_args)
if model_args.vision_tower is not None:
    model.get_model().initialize_vision_modules(model_args=model_args)
model.load_state_dict(state_dict)
model.cuda()


breakpoint()
