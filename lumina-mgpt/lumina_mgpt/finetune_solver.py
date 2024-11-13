import pickle
from typing import List, Tuple

from accelerate import init_empty_weights
import torch

from model import ChameleonXLLMXConfig, ChameleonXLLMXForConditionalGeneration
from xllmx.data.item_processor import ItemProcessorBase
from xllmx.solvers.finetune import FinetuneSolverBase


class ItemProcessor(ItemProcessorBase):
    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
        assert training_mode

        if "token" in data_item and "label" in data_item:
            data_item = data_item
        else:
            assert "file" in data_item
            with open(data_item["file"], "rb") as f:
                data_item = pickle.load(f)

        tokens = data_item["token"]
        labels = data_item["label"]
        assert len(tokens) == len(labels)

        return tokens, labels

    def predict_item_token_length(self, data_item: dict) -> int:
        if "token" in data_item:
            return len(data_item["token"])
        elif "len" in data_item:
            return data_item["len"]
        else:
            raise ValueError()


class Solver(FinetuneSolverBase):
    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        # task-specific parameters
        parser.add_argument("--max_seq_len", default=4096, type=int, help="max token length")
        parser.add_argument("--mask_image_logits", default=True)
        parser.add_argument("--unmask_image_logits", action="store_false", dest="mask_image_logits")
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--z_loss_weight", type=float, default=0.0)
        parser.add_argument("--model_size", type=str, default="7B", choices=["7B", "34B"])
        return parser

    def _model_func(
        self,
        init_from: str,
    ) -> (ChameleonXLLMXForConditionalGeneration, None):

        # Only instantiate the model on rank0
        # Other ranks will receive the model weights from rank0 during FSDP wrapping (through `sync_module_states`)
        # See https://github.com/pytorch/pytorch/issues/105840
        if self.dp_rank == 0:
            model = ChameleonXLLMXForConditionalGeneration.from_pretrained(
                init_from,
                max_position_embeddings=self.args.max_seq_len,
                mask_image_logits=self.args.mask_image_logits,
                dropout=self.args.dropout,
                z_loss_weight=self.args.z_loss_weight,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
            )
        else:
            with init_empty_weights():
                config = ChameleonXLLMXConfig.from_pretrained(
                    init_from,
                    max_position_embeddings=self.args.max_seq_len,
                    mask_image_logits=self.args.mask_image_logits,
                    dropout=self.args.dropout,
                    z_loss_weight=self.args.z_loss_weight,
                    torch_dtype=torch.bfloat16,
                )
                model = ChameleonXLLMXForConditionalGeneration(config)

        del model.model.vqmodel

        return model, None

    def _item_processor_func(self) -> ItemProcessorBase:
        return ItemProcessor()

    def _make_and_save_starting_point(self, save_path: str) -> None:

        pretrained_name = {
            "7B": "Alpha-VLLM/Chameleon_7B_mGPT",
            "34B": "Alpha-VLLM/Chameleon_34B_mGPT",
        }[self.args.model_size]

        model = ChameleonXLLMXForConditionalGeneration.from_pretrained(
            pretrained_name,
            max_position_embeddings=self.args.max_seq_len,
            mask_image_logits=self.args.mask_image_logits,
            dropout=self.args.dropout,
            z_loss_weight=self.args.z_loss_weight,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )

        image_tokens = model.model.vocabulary_mapping.image_tokens
        model.lm_head.weight.data[image_tokens] = torch.zeros_like(model.lm_head.weight.data[image_tokens])

        model.save_pretrained(save_path, max_shard_size="10GB")


if __name__ == "__main__":
    args = Solver.get_args_parser().parse_args()
    solver = Solver(args)
    solver.run()
