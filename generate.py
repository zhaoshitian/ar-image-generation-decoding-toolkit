import os
import sys
from functools import partial

# sys.path.append(os.path.abspath(__file__).rsplit("/", 2)[0])
sys.path.append("/mnt/petrelfs/gaopeng/zst/ar-image-generation-decoding-toolkit")
import argparse

from PIL import Image
import torch

from Lumina_mGPT.lumina_mgpt.inference_solver import FlexARInferenceSolver
from Lumina_mGPT.xllmx.util.misc import random_seed

from ensemble_logits_processor import EnsembleLogitsProcessor
from i2icfg_logits_processor import I2ILLMImageStartTriggeredUnbatchedClassifierFreeGuidanceLogitsProcessor
from minp_logits_processor import MinPLogitsWarper
from dynamicT_logits_processor import DynamicTemperatureLogitsWarper

def checkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)

def init_logits_processor(index):

    index_logits_processor_mapping = {
        1: "ensemble",
        2: "minp",
        3: "i2icfg",
        4: "dynamicT"
    }

    if index == 1: 
        params_dict = {
            "guidance_scale": cfg,
            "patch_size": 32,
        }
        # ensemblelogitsprocessor = EnsembleLogitsProcessor(
        #     guidance_scale=cfg,
        #     model=self.model,
        #     image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
        #     image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
        #     image_next_line_token_id=self.item_processor.token2id(self.item_processor.new_line_token),
        #     patch_size=32,
        #     threshold=cfg_threshold,
        # )
        partial_ensemblelogitsprocessor = partial(EnsembleLogitsProcessor, **params_dict)
    elif index == 2:
        minplogitwraper = MinPLogitsWarper(
            min_p=min_p
        )
    elif index == 3:
        # i2icfgprocessor = I2ILLMImageStartTriggeredUnbatchedClassifierFreeGuidanceLogitsProcessor(
       
        # ) 
        pass
    elif index == 4:
        # dynamicTprocessor = DynamicTemperatureLogitsWarper(
        #     temperature=1.0, 
        #     hot_temperature=30, 
        #     num_hot_tokens=1,
        #     image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
        #     image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
        #     image_next_line_token_id=self.item_processor.token2id(self.item_processor.new_line_token),
        #     patch_size=32
        # ) 

        params_dict = {
            "temperature": 1.0,
            "hot_temperature": 30,
            "num_hot_tokens": 1,
            "patch_size": 32,
        }
        # dynamicTprocessor = DynamicTemperatureLogitsWarper(
        #     temperature=1.0, 
        #     hot_temperature=30, 
        #     num_hot_tokens=1,
        #     image_start_token_id=self.item_processor.token2id(self.item_processor.image_start_token),
        #     image_end_token_id=self.item_processor.token2id(self.item_processor.image_end_token),
        #     image_next_line_token_id=self.item_processor.token2id(self.item_processor.new_line_token),
        #     patch_size=32
        # ) 
        partial_dynamicTlogitsprocessor = partial(DynamicTemperatureLogitsWarper, **params_dict)
        return [partial_dynamicTlogitsprocessor], "dynamicT"

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--lp_list", type=int, default=None)

    args = parser.parse_args()

    print("args:\n", args)

    select_set1 = [
        "Image of a dog playing water, and a water fall is in the background.",
        "A family of asian people sitting around the dinner table, eating and laughing.",
        "A high-resolution photograph of a middle-aged woman with curly hair, wearing traditional Japanese kimono, smiling gently under a cherry blossom tree in full bloom.",  # noqa
        "Image of a bustling downtown street in Tokyo at night, with neon signs, crowded sidewalks, and tall skyscrapers.",  # noqa
        "Image of a quiet European village with cobblestone streets and colorful houses, under a clear blue sky.",
    ]

    l_prompts = select_set1

    t = args.temperature
    top_k = args.top_k
    cfg = args.cfg
    n = args.n
    w, h = args.width, args.height

    inference_solver = FlexARInferenceSolver(
        model_path=args.model_path,
        precision="bf16",
    )

    additional_logits_processor, name_logits_processor = init_logits_processor(args.lp_list)
    ## just for dynamicT decoding method ##
    additional_logits_processor = additional_logits_processor[0](
        image_start_token_id=inference_solver.item_processor.token2id(inference_solver.item_processor.image_start_token),
        image_end_token_id=inference_solver.item_processor.token2id(inference_solver.item_processor.image_end_token),
        image_next_line_token_id=inference_solver.item_processor.token2id(inference_solver.item_processor.new_line_token),
    )

    generated_images_dir_path = f"T{t}_topk{top_k}_cfg{cfg}_w{w}_h{h}_lp{str(name_logits_processor)}"

    with torch.no_grad():
        l_generated_all = []
        for i, prompt in enumerate(l_prompts):
            for repeat_idx in range(n):
                random_seed(repeat_idx)
                generated = inference_solver.generate(
                    images=[],
                    qas=[[f"Generate an image of {w}x{h} according to the following prompt:\n{prompt}", None]],
                    max_gen_len=8192,
                    temperature=t,
                    logits_processor=inference_solver.create_logits_processor(cfg=cfg, image_top_k=top_k, additional_logits_processor_list=[additional_logits_processor]),
                )
                try:
                    l_generated_all.append(generated[1][0])
                    image = generated[1][0]
                    image_save_path = os.path.join(args.save_path, generated_images_dir_path)
                    checkpath(image_save_path)
                    image_save_path = os.path.join(image_save_path, f"{str(i)}.png")
                    image.save(image_save_path)

                except:
                    l_generated_all.append(Image.new("RGB", (w, h)))

        result_image = inference_solver.create_image_grid(l_generated_all, len(l_prompts), n)
        result_image.save(args.save_path)