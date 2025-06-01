import sys
import dataclasses

sys.path.append(".")

import argparse
import hydra
from omegaconf import OmegaConf
from accelerate.utils import set_seed
from pc_sam.model.pc_sam import PointCloudSAM
from pc_sam.utils.torch_utils import replace_with_fused_layernorm
from safetensors.torch import load_model
from pc_sam.ply_utils import load_ply

import numpy as np
import torch

@dataclasses.dataclass
class AuxInputs:
    coords: torch.Tensor
    features: torch.Tensor
    centers: torch.Tensor
    interp_index: torch.Tensor = None
    interp_weight: torch.Tensor = None


def repeat_interleave(x: torch.Tensor, repeats: int, dim: int):
    if repeats == 1:
        return x
    shape = list(x.shape)
    shape.insert(dim + 1, 1)
    shape[dim + 1] = repeats
    x = x.unsqueeze(dim + 1).expand(shape).flatten(dim, dim + 1)
    return x

class PointCloudProcessor:
    def __init__(self, device="cuda", batch=True, return_tensors="pt"):
        self.device = device
        self.batch = batch
        self.return_tensors = return_tensors

        self.center = None
        self.scale = None

    def __call__(self, xyz: np.ndarray, rgb: np.ndarray):
        # # The original data is z-up. Make it y-up.
        # rot = Rotation.from_euler("x", -90, degrees=True)
        # xyz = rot.apply(xyz)

        if self.center is None or self.scale is None:
            self.center = xyz.mean(0)
            self.scale = np.max(np.linalg.norm(xyz - self.center, axis=-1))

        xyz = (xyz - self.center) / self.scale
        rgb = ((rgb / 255.0) - 0.5) * 2

        if self.return_tensors == "np":
            coords = np.float32(xyz)
            feats = np.float32(rgb)
            if self.batch:
                coords = np.expand_dims(coords, 0)
                feats = np.expand_dims(feats, 0)
        elif self.return_tensors == "pt":
            coords = torch.tensor(xyz, dtype=torch.float32, device=self.device)
            feats = torch.tensor(rgb, dtype=torch.float32, device=self.device)
            if self.batch:
                coords = coords.unsqueeze(0)
                feats = feats.unsqueeze(0)
        else:
            raise ValueError(self.return_tensors)

        return coords, feats

    def normalize(self, xyz):
        return (xyz - self.center) / self.scale


class PointCloudSAMPredictor:
    input_xyz: np.ndarray
    input_rgb: np.ndarray
    prompt_coords: list[tuple[float, float, float]]
    prompt_labels: list[int]

    coords: torch.Tensor
    feats: torch.Tensor

    pc_embedding: torch.Tensor
    patches: dict[str, torch.Tensor]
    prompt_mask: torch.Tensor

    def __init__(self, args, cfg):
        print("Created model")
        model: PointCloudSAM = hydra.utils.instantiate(cfg.model)
        model.pc_encoder.patch_embed.grouper.num_groups = 1024
        model.pc_encoder.patch_embed.grouper.group_size = 128
        load_model(model, args.ckpt_path)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()

        self.model = model

        self.input_rgb = None
        self.input_xyz = None

        self.input_processor = None
        self.coords = None
        self.feats = None

        self.pc_embedding = None
        self.patches = None

        self.prompt_coords = None
        self.prompt_labels = None
        self.prompt_mask = None
        self.candidate_index = 0

    @torch.no_grad()
    def set_pointcloud(self, xyz, rgb):
        self.input_xyz = xyz
        self.input_rgb = rgb

        self.input_processor = PointCloudProcessor()
        coords, feats = self.input_processor(xyz, rgb)
        self.coords = coords
        self.feats = feats

        pc_embedding, patches = self.model.pc_encoder(self.coords, self.feats)
        self.pc_embedding = pc_embedding
        self.patches = patches
        self.prompt_mask = None

    def set_prompts(self, prompt_coords, prompt_labels):
        self.prompt_coords = prompt_coords
        self.prompt_labels = prompt_labels

    @torch.no_grad()
    def predict_mask(self):
        normalized_prompt_coords = self.input_processor.normalize(
            np.array(self.prompt_coords)
        )
        prompt_coords = torch.tensor(
            normalized_prompt_coords, dtype=torch.float32, device="cuda"
        )
        prompt_labels = torch.tensor(
            self.prompt_labels, dtype=torch.bool, device="cuda"
        )
        prompt_coords = prompt_coords.reshape(1, -1, 3)
        prompt_labels = prompt_labels.reshape(1, -1)

        multimask_output = prompt_coords.shape[1] == 1

        # [B * M, num_outputs, num_points], [B * M, num_outputs]
        def decode_masks(
            coords,
            feats,
            pc_embedding,
            patches,
            prompt_coords,
            prompt_labels,
            prompt_masks,
            multimask_output,
        ):
            pc_embeddings, patches = pc_embedding, patches
            centers = patches["centers"]
            knn_idx = patches["knn_idx"]
            coords = patches["coords"]
            feats = patches["feats"]
            aux_inputs = AuxInputs(coords=coords, features=feats, centers=centers)

            pc_pe = self.model.point_encoder.pe_layer(centers)
            sparse_embeddings = self.model.point_encoder(prompt_coords, prompt_labels)
            dense_embeddings = self.model.mask_encoder(
                prompt_masks, coords, centers, knn_idx
            )
            dense_embeddings = repeat_interleave(
                dense_embeddings,
                sparse_embeddings.shape[0] // dense_embeddings.shape[0],
                0,
            )

            logits, iou_preds = self.model.mask_decoder(
                pc_embeddings,
                pc_pe,
                sparse_embeddings,
                dense_embeddings,
                aux_inputs=aux_inputs,
                multimask_output=multimask_output,
            )
            return logits, iou_preds

        logits, scores = decode_masks(
            self.coords,
            self.feats,
            self.pc_embedding,
            self.patches,
            prompt_coords,
            prompt_labels,
            (
                self.prompt_mask[self.candidate_index].unsqueeze(0)
                if self.prompt_mask is not None
                else None
            ),
            multimask_output,
        )
        logits = logits.squeeze(0)
        scores = scores.squeeze(0)

        # if multimask_output:
        #     index = scores.argmax(0).item()
        #     logit = logits[index]
        # else:
        #     logit = logits.squeeze(0)

        # self.prompt_mask = logit.unsqueeze(0)

        # pred_mask = logit > 0
        # return pred_mask.cpu().numpy()

        # Sort according to scores
        _, indices = scores.sort(descending=True)
        logits = logits[indices]

        self.prompt_mask = logits  # [num_outputs, num_points]
        self.candidate_index = 0

        return (logits > 0).cpu().numpy()

    def set_candidate(self, index):
        self.candidate_index = index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="large", help="path to config file"
    )
    parser.add_argument("--config_dir", type=str, default="../configs")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/workspace/checkpoints/model.safetensors",
    )
    parser.add_argument("--pointcloud", type=str)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args, unknown_args = parser.parse_known_args()

    # ---------------------------------------------------------------------------- #
    # Load configuration
    # ---------------------------------------------------------------------------- #
    with hydra.initialize(args.config_dir, version_base=None):
        cfg = hydra.compose(config_name=args.config, overrides=unknown_args)
        OmegaConf.resolve(cfg)
        # print(OmegaConf.to_yaml(cfg))

    seed = cfg.get("seed", 42)
    
    set_seed(seed)
    model = PointCloudSAMPredictor(args, cfg)
    
    with torch.no_grad():
        
        # load pcd from ply file and set on model
        points = load_ply(f"{args.pointcloud}")
        xyz = points[:, :3]
        rgb = points[:, 3:6]
        
        xyz = np.array(xyz).reshape(-1, 3)
        rgb = np.array(list(rgb)).reshape(-1, 3)
        model.set_pointcloud(xyz, rgb)
        
        # set prompt coordinates and labels
        # coordinates - expected area for masking foots
        # labels - 0: no mask area / 1: mask area
        
        print()
        

    #data = {"xyz": coords, "rgb": colors, "mask": labels}
    #outputs = model(**data, is_eval=True)

if __name__ == "__main__":
    main()
