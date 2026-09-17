# V2 Phase 3A Foundation Model Provenance

This registry fixes the exact representation systems before feature extraction. Mutable aliases are not sufficient experiment identifiers; every remote model is paired with the resolved revision used by this study.

## Experiment 005 — DINOv2 ViT-B/14

| Field | Registered value |
| --- | --- |
| Model family | DINOv2 |
| Canonical Meta architecture | `dinov2_vitb14` |
| Loading checkpoint | `facebook/dinov2-base` |
| Resolved revision | `f9e44c814b77203eaa57a6bdbbd535f21ede1415` |
| Provider | Meta AI / Facebook Research |
| Library | Transformers 5.16.1 |
| Loading API | `AutoImageProcessor.from_pretrained`; `AutoModel.from_pretrained` |
| Pretraining | LVD-142M; self-supervised DINO/iBOT objectives with KoLeo regularization |
| Architecture | ViT-B/14, 12 transformer blocks |
| Native benchmark input | Resize shortest edge to 256, bicubic; center crop 224; RGB; scale by 1/255; ImageNet mean/std |
| Embedding | 768-dimensional pooled class-token representation |
| Encoder parameters | 86,580,480 (instantiated image encoder) |
| Weight file | `model.safetensors`, 346,345,912 bytes; SHA-256 `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841` |
| License | Apache-2.0 |
| Download source | `https://huggingface.co/facebook/dinov2-base` |

The Hugging Face identifier is the maintained Transformers packaging of Meta's non-register `dinov2_vitb14` LVD-142M checkpoint. The model configuration permits multiple sizes; this experiment follows the pinned processor's deterministic 224-pixel evaluation crop.

## Experiment 006 — Original OpenAI CLIP ViT-B/16

| Field | Registered value |
| --- | --- |
| Model family | CLIP |
| OpenCLIP model name | `ViT-B-16` |
| OpenCLIP pretrained identifier | `openai` |
| Pinned mirror checkpoint | `timm/vit_base_patch16_clip_224.openai` |
| Resolved mirror revision | `977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d` |
| Provider | OpenAI; loaded through OpenCLIP |
| Library | open-clip-torch 3.3.0 |
| Loading API | `open_clip.create_model_and_transforms` with pinned local safetensors and QuickGELU |
| Pretraining | Original CLIP generation; approximately 400M internet image-text pairs |
| Architecture | ViT-B/16, 12 vision transformer blocks |
| Native benchmark input | Resize shortest edge to 224, bicubic; center crop 224; RGB; scale by 1/255; OpenAI CLIP mean/std |
| Embedding | 512-dimensional projected image representation from `encode_image` |
| Encoder parameters | 86,192,640 (instantiated visual tower) |
| Weight file | `open_clip_model.safetensors`, 598,516,980 bytes; SHA-256 `4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f` |
| License | Original OpenAI CLIP repository: MIT; Hugging Face mirror metadata: Apache-2.0 |
| Original weight URL | `https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt` |
| Pinned download source | `https://huggingface.co/timm/vit_base_patch16_clip_224.openai` |

The `openai` pretrained tag is used deliberately. No LAION, DataComp, MetaCLIP, or other later OpenCLIP checkpoint is substituted.

## Experiment 007 — SigLIP2 Base Patch16/224

| Field | Registered value |
| --- | --- |
| Model family | SigLIP2 |
| Checkpoint | `google/siglip2-base-patch16-224` |
| Resolved revision | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` |
| Provider | Google |
| Library | Transformers 5.16.1 |
| Loading API | `AutoImageProcessor.from_pretrained`; `SiglipModel.from_pretrained`, retaining only its pretrained `vision_model` |
| Pretraining | Multilingual image-text sigmoid objective plus captioning, self-distillation, masked prediction, and online data curation; exact corpus composition is not fully enumerated in the model card |
| Architecture | ViT-B/16, 12 vision transformer blocks |
| Native benchmark input | Direct resize to 224 × 224, bilinear; scale by 1/255; mean `[0.5, 0.5, 0.5]`; std `[0.5, 0.5, 0.5]` |
| Embedding | 768-dimensional pooled vision representation |
| Encoder parameters | 92,884,224 (instantiated vision tower including pooling head) |
| Combined checkpoint file | `model.safetensors`, 1,500,800,904 bytes; SHA-256 `612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b` |
| License | Apache-2.0 |
| Download source | `https://huggingface.co/google/siglip2-base-patch16-224` |

The fixed-resolution checkpoint retains a SigLIP-compatible top-level configuration (`model_type: siglip`). The combined checkpoint is therefore loaded with `SiglipModel` so its nested vision configuration and weights are interpreted exactly; the text-containing wrapper is immediately discarded and only the frozen pretrained `vision_model` is retained for encoding. No tokenizer, text input, prompt, text forward pass, or zero-shot decision path participates in Experiment 007.

## Common Representation Rule

Every physical image receives one deterministic embedding with no stochastic augmentation. The extracted vectors are L2-normalized before cache storage and before the identical linear probe. This controls embedding scale but does not make the three systems causally identical: architecture, parameter count, pretraining objective/data, preprocessing, and dimensionality remain different.
