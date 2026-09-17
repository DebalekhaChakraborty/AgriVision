"""Fixed-grid occlusion and deletion-faithfulness analysis for Experiment 015."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from v2.src.datasets.transfer_transforms import build_transfer_transform
from v2.src.foundation.linear_probe import LinearProbe
from v2.src.foundation.prompt_registry import encode_class_prototypes, load_prompt_registry
from v2.src.foundation.similarity_classifier import build_zero_shot_model
from v2.src.models.transfer_registry import build_transfer_model
from v2.src.training.utils import load_config, resolve_device, save_json, sha256_file, utc_timestamp


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "v2/results/experiment_015_interpretability"
CACHE = ROOT / "v2/cache/experiment_015_interpretability"
SULTANA_ROOT = ROOT / "v2/data/external/sultana_2022/Original Image"
FRUITVISION_ROOT = ROOT / "v2/data/external/fruitvision/Fruits Original"
CLASS_NAMES = ["fresh_apple", "fresh_banana", "fresh_orange", "rotten_apple", "rotten_banana", "rotten_orange"]
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}


def root_path(value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def read_json(value: str | Path) -> Any:
    with root_path(value).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def image_path(record: dict[str, Any]) -> Path:
    base = SULTANA_ROOT if record["domain"] == "sultana" else FRUITVISION_ROOT
    return base / record["identifier"]


def regions(height: int, width: int, rows: int = 7, cols: int = 7) -> list[tuple[int, int, int, int]]:
    ys = np.linspace(0, height, rows + 1, dtype=int)
    xs = np.linspace(0, width, cols + 1, dtype=int)
    return [(int(ys[r]), int(ys[r+1]), int(xs[c]), int(xs[c+1])) for r in range(rows) for c in range(cols)]


def fixed_target_margins(scores: torch.Tensor, target_index: int) -> torch.Tensor:
    target = scores[:, target_index]
    masked = scores.clone(); masked[:, target_index] = -torch.inf
    return target - masked.max(dim=1).values


def apply_regions(image: torch.Tensor, region_list: list[tuple[int, int, int, int]], selected: list[int]) -> torch.Tensor:
    value = image.clone()
    for index in selected:
        y0, y1, x0, x1 = region_list[index]
        value[:, y0:y1, x0:x1] = 0.0
    return value


def deletion_orders(importance: np.ndarray, seed: int) -> tuple[list[int], list[int]]:
    top = np.argsort(-importance, kind="stable").tolist()
    random = np.random.default_rng(seed).permutation(len(importance)).tolist()
    return top, random


def analyze_one(
    record: dict[str, Any],
    system: str,
    image: torch.Tensor,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    config: dict[str, Any],
) -> dict[str, Any]:
    target_name = record["frozen_predictions"][system]["predicted_class"]
    target_index = CLASS_TO_IDX[target_name]
    grid_rows, grid_cols = map(int, config["attribution"]["grid"])
    region_list = regions(image.shape[1], image.shape[2], grid_rows, grid_cols)
    base_scores = score_fn(image.unsqueeze(0)).detach().cpu()
    predicted_index = int(base_scores.argmax(dim=1).item())
    if predicted_index != target_index:
        raise RuntimeError(f"Frozen prediction mismatch for {system}/{record['domain']}/{record['identifier']}: {CLASS_NAMES[predicted_index]} != {target_name}")
    base_margin = float(fixed_target_margins(base_scores, target_index).item())
    occluded = torch.stack([apply_regions(image, region_list, [i]) for i in range(len(region_list))])
    occluded_margins = fixed_target_margins(score_fn(occluded).detach().cpu(), target_index).numpy()
    importance = base_margin - occluded_margins
    seed = int(record["selection_sha256"][:8], 16) ^ int(config["attribution"]["random_seed"])
    top_order, random_order = deletion_orders(importance, seed)
    fractions = [float(v) for v in config["attribution"]["deletion_fractions"]]
    counts = [min(len(region_list), int(round(v * len(region_list)))) for v in fractions]
    top_batch = torch.stack([apply_regions(image, region_list, top_order[:count]) for count in counts])
    random_batch = torch.stack([apply_regions(image, region_list, random_order[:count]) for count in counts])
    top_margins = fixed_target_margins(score_fn(top_batch).detach().cpu(), target_index).numpy()
    random_margins = fixed_target_margins(score_fn(random_batch).detach().cpu(), target_index).numpy()
    denominator = max(abs(base_margin), 1e-8)
    top_relative = top_margins / denominator; random_relative = random_margins / denominator
    top_auc = float(np.trapezoid(top_relative, fractions)); random_auc = float(np.trapezoid(random_relative, fractions))
    positive = np.maximum(importance, 0); total_positive = float(positive.sum())
    edge = np.asarray([r in (0, grid_rows-1) or c in (0, grid_cols-1) for r in range(grid_rows) for c in range(grid_cols)])
    center = ~edge
    top5 = np.argsort(-importance, kind="stable")[:5]
    radial = [math.sqrt((i//grid_cols-(grid_rows-1)/2)**2+(i%grid_cols-(grid_cols-1)/2)**2) for i in top5]
    return {
        "system": system,
        "domain": record["domain"],
        "identifier": record["identifier"],
        "ground_truth_class": record["semantic_class"],
        "frozen_predicted_class": target_name,
        "frozen_correct": record["frozen_predictions"][system]["correct"],
        "selection_stratum": record["correctness_stratum"],
        "base_target_margin": base_margin,
        "grid": [grid_rows, grid_cols],
        "regions": [{"index":i,"row":i//grid_cols,"column":i%grid_cols,"bounds":[*region_list[i]],"importance":float(importance[i])} for i in range(len(region_list))],
        "top_region_indices": top_order,
        "random_region_indices": random_order,
        "positive_importance_sum": total_positive,
        "edge_positive_importance_share": float(positive[edge].sum()/total_positive) if total_positive else None,
        "interior_positive_importance_share": float(positive[center].sum()/total_positive) if total_positive else None,
        "top5_mean_grid_radius": float(np.mean(radial)),
        "deletion": {
            "fractions": fractions,
            "top_ranked_target_margins": top_margins.tolist(),
            "random_target_margins": random_margins.tolist(),
            "top_ranked_relative_margins": top_relative.tolist(),
            "random_relative_margins": random_relative.tolist(),
            "top_ranked_auc": top_auc,
            "random_auc": random_auc,
            "faithfulness_gap_random_minus_top": random_auc-top_auc,
        },
    }


def assemble_shared_result(
    record: dict[str, Any], system: str, base_margin: float,
    region_list: list[tuple[int, int, int, int]], importance: np.ndarray,
    top_order: list[int], random_order: list[int], fractions: list[float],
    top_margins: np.ndarray, random_margins: np.ndarray,
    grid_rows: int, grid_cols: int,
) -> dict[str, Any]:
    denominator=max(abs(base_margin),1e-8); top_relative=top_margins/denominator; random_relative=random_margins/denominator
    top_auc=float(np.trapezoid(top_relative,fractions)); random_auc=float(np.trapezoid(random_relative,fractions)); positive=np.maximum(importance,0); total_positive=float(positive.sum())
    edge=np.asarray([r in (0,grid_rows-1) or c in (0,grid_cols-1) for r in range(grid_rows) for c in range(grid_cols)]); center=~edge; top5=np.argsort(-importance,kind="stable")[:5]
    radial=[math.sqrt((i//grid_cols-(grid_rows-1)/2)**2+(i%grid_cols-(grid_cols-1)/2)**2) for i in top5]
    return {"system":system,"domain":record["domain"],"identifier":record["identifier"],"ground_truth_class":record["semantic_class"],"frozen_predicted_class":record["frozen_predictions"][system]["predicted_class"],"frozen_correct":record["frozen_predictions"][system]["correct"],"selection_stratum":record["correctness_stratum"],"base_target_margin":base_margin,"grid":[grid_rows,grid_cols],"regions":[{"index":i,"row":i//grid_cols,"column":i%grid_cols,"bounds":[*region_list[i]],"importance":float(importance[i])} for i in range(len(region_list))],"top_region_indices":top_order,"random_region_indices":random_order,"positive_importance_sum":total_positive,"edge_positive_importance_share":float(positive[edge].sum()/total_positive) if total_positive else None,"interior_positive_importance_share":float(positive[center].sum()/total_positive) if total_positive else None,"top5_mean_grid_radius":float(np.mean(radial)),"deletion":{"fractions":fractions,"top_ranked_target_margins":top_margins.tolist(),"random_target_margins":random_margins.tolist(),"top_ranked_relative_margins":top_relative.tolist(),"random_relative_margins":random_relative.tolist(),"top_ranked_auc":top_auc,"random_auc":random_auc,"faithfulness_gap_random_minus_top":random_auc-top_auc}}


def analyze_family_one(record:dict[str,Any],family:str,image:torch.Tensor,embed_fn:Callable[[torch.Tensor],torch.Tensor],probe:LinearProbe,prototypes:torch.Tensor,config:dict[str,Any])->tuple[dict[str,Any],dict[str,Any]]:
    probe_system=f"{family}_linear_probe"; zero_system=f"{family}_strict_zero_shot"; grid_rows,grid_cols=map(int,config["attribution"]["grid"]); region_list=regions(image.shape[1],image.shape[2],grid_rows,grid_cols)
    with torch.inference_mode():
        base_embedding=embed_fn(image.unsqueeze(0)); occluded=torch.stack([apply_regions(image,region_list,[i]) for i in range(len(region_list))]); occluded_embeddings=embed_fn(occluded)
        base_scores={probe_system:probe(base_embedding).cpu(),zero_system:(base_embedding@prototypes.to(base_embedding.device).T).cpu()}; occluded_scores={probe_system:probe(occluded_embeddings).cpu(),zero_system:(occluded_embeddings@prototypes.to(occluded_embeddings.device).T).cpu()}
    prepared={}; batches=[]; fractions=[float(v) for v in config["attribution"]["deletion_fractions"]]; counts=[min(len(region_list),int(round(v*len(region_list)))) for v in fractions]
    for system in (probe_system,zero_system):
        target=CLASS_TO_IDX[record["frozen_predictions"][system]["predicted_class"]]; predicted=int(base_scores[system].argmax(1).item())
        if predicted!=target: raise RuntimeError(f"Frozen prediction mismatch for {system}/{record['domain']}/{record['identifier']}: {CLASS_NAMES[predicted]} != {CLASS_NAMES[target]}")
        base_margin=float(fixed_target_margins(base_scores[system],target).item()); importance=base_margin-fixed_target_margins(occluded_scores[system],target).numpy(); seed=int(record["selection_sha256"][:8],16)^int(config["attribution"]["random_seed"]); top_order,random_order=deletion_orders(importance,seed)
        prepared[system]=(target,base_margin,importance,top_order,random_order); batches.extend([torch.stack([apply_regions(image,region_list,top_order[:count]) for count in counts]),torch.stack([apply_regions(image,region_list,random_order[:count]) for count in counts])])
    with torch.inference_mode(): deletion_embeddings=embed_fn(torch.cat(batches))
    n=len(fractions); results=[]
    for offset,system in enumerate((probe_system,zero_system)):
        target,base_margin,importance,top_order,random_order=prepared[system]; start=offset*2*n
        with torch.inference_mode(): all_scores=probe(deletion_embeddings[start:start+2*n]).cpu() if system==probe_system else (deletion_embeddings[start:start+2*n]@prototypes.to(deletion_embeddings.device).T).cpu()
        top_margins=fixed_target_margins(all_scores[:n],target).numpy(); random_margins=fixed_target_margins(all_scores[n:],target).numpy(); results.append(assemble_shared_result(record,system,base_margin,region_list,importance,top_order,random_order,fractions,top_margins,random_margins,grid_rows,grid_cols))
    return results[0],results[1]


def progress_path(system: str) -> Path:
    return CACHE / f"{system}_occlusion_progress.json"


def load_progress(system: str) -> list[dict[str, Any]]:
    value = progress_path(system)
    return read_json(value)["records"] if value.exists() else []


def save_progress(system: str, records: list[dict[str, Any]], manifest_hash: str) -> None:
    save_json({"schema_version":1,"system":system,"attribution_manifest_sha256":manifest_hash,"records":records},progress_path(system))


def run_single_system(system: str, records: list[dict[str, Any]], preprocess: Callable, score_fn: Callable, config: dict[str, Any], manifest_hash: str) -> list[dict[str, Any]]:
    completed=load_progress(system); done={(r["domain"],r["identifier"]) for r in completed}
    for number,record in enumerate(records,1):
        key=(record["domain"],record["identifier"])
        if key in done: continue
        with Image.open(image_path(record)) as raw: image=preprocess(raw.convert("RGB"))
        completed.append(analyze_one(record,system,image,score_fn,config)); save_progress(system,completed,manifest_hash)
        if len(completed)%5==0 or len(completed)==len(records): print(f"{system}: {len(completed)}/{len(records)}",flush=True)
    return completed


def load_probe(config_path: str, dimension: int, device: torch.device) -> LinearProbe:
    cfg=load_config(ROOT/config_path); checkpoint=torch.load(root_path(cfg["outputs"]["probe_checkpoint"]),map_location=device,weights_only=True)
    probe=LinearProbe(dimension,6).to(device); probe.load_state_dict(checkpoint["model_state_dict"]); probe.eval(); return probe


def run(config: dict[str, Any]) -> None:
    output=RESULTS/"occlusion_results.json"
    if output.exists(): raise FileExistsError("Refusing to overwrite completed Experiment 015 occlusion results.")
    manifest_path=root_path(config["outputs"]["attribution_manifest"]); manifest_hash=sha256_file(manifest_path); recorded=(RESULTS/"attribution_subset_manifest.sha256").read_text().split()[0]
    manifest=read_json(manifest_path)
    if manifest_hash!=recorded or manifest["status"]!="frozen_before_attribution_generation": raise RuntimeError("Attribution manifest identity failed.")
    records=manifest["records"]; device=resolve_device("auto"); CACHE.mkdir(parents=True,exist_ok=True); all_results={}

    efficient_cfg=load_config(ROOT/"v2/configs/efficientnet.yaml"); efficient=build_transfer_model(efficient_cfg["model"],load_pretrained=False).to(device); checkpoint=torch.load(root_path(efficient_cfg["outputs"]["checkpoint"]),map_location=device,weights_only=True); efficient.load_state_dict(checkpoint["model_state_dict"]); efficient.eval()
    efficient_transform=build_transfer_transform("test",efficient_cfg["preprocessing"])
    def efficient_scores(batch:torch.Tensor)->torch.Tensor:
        with torch.inference_mode(): return efficient(batch.to(device)).cpu()
    all_results["efficientnet_b0"]=run_single_system("efficientnet_b0",records,efficient_transform,efficient_scores,config,manifest_hash)
    del efficient; gc.collect()

    for family,zero_config,probe_config,dimension in (
        ("clip","v2/configs/clip_zero_shot.yaml","v2/configs/clip_linear_probe.yaml",512),
        ("siglip2","v2/configs/siglip2_zero_shot.yaml","v2/configs/siglip2_linear_probe.yaml",768),
    ):
        zero_cfg=load_config(ROOT/zero_config); model=build_zero_shot_model(zero_cfg["model"],device); registry=load_prompt_registry(root_path(zero_cfg["prompts"]["registry_path"]),zero_cfg["prompts"]["registry_sha256"]); prototypes,_=encode_class_prototypes(model,registry,"P1"); probe=load_probe(probe_config,dimension,device)
        probe_system=f"{family}_linear_probe"; zero_system=f"{family}_strict_zero_shot"
        probe_completed=load_progress(probe_system); zero_completed=load_progress(zero_system); probe_done={(r["domain"],r["identifier"]) for r in probe_completed}; zero_done={(r["domain"],r["identifier"]) for r in zero_completed}
        for record in records:
            key=(record["domain"],record["identifier"])
            if key in probe_done and key in zero_done: continue
            with Image.open(image_path(record)) as raw: image=model.preprocess(raw.convert("RGB"))
            def embedding_scores(batch:torch.Tensor)->torch.Tensor:
                with torch.inference_mode(): return F.normalize(model.encode_images(batch.to(device)).float(),p=2,dim=1)
            probe_record,zero_record=analyze_family_one(record,family,image,embedding_scores,probe,prototypes,config)
            if key not in probe_done: probe_completed.append(probe_record); save_progress(probe_system,probe_completed,manifest_hash)
            if key not in zero_done: zero_completed.append(zero_record); save_progress(zero_system,zero_completed,manifest_hash)
            if len(probe_completed)%5==0 or len(probe_completed)==len(records): print(f"{family}: probe={len(probe_completed)}, zero={len(zero_completed)}/{len(records)}",flush=True)
        all_results[probe_system]=probe_completed; all_results[zero_system]=zero_completed
        del model,probe,prototypes; gc.collect()
    aggregate=aggregate_results(all_results)
    save_json({"schema_version":1,"experiment_id":config["experiment"]["id"],"created_at":utc_timestamp(),"attribution_manifest_sha256":manifest_hash,"protocol":config["attribution"],"records_by_system":all_results,"aggregate":aggregate,"interpretation_boundary":"Occlusion measures score sensitivity under this perturbation; it is not a causal or human-ground-truth explanation."},output)
    save_json({"schema_version":1,"metric":"deletion target-margin AUC normalized to original target margin; lower top-ranked AUC and positive random-minus-top gap indicate greater perturbational faithfulness","aggregate":aggregate},RESULTS/"deletion_faithfulness.json")
    plot_faithfulness(all_results)
    print("Completed fixed-grid occlusion and deletion-faithfulness analysis.")


def aggregate_results(results: dict[str,list[dict[str,Any]]])->dict[str,Any]:
    output={}
    for system,records in results.items():
        output[system]={}
        for domain in ("all","sultana","fruitvision"):
            subset=records if domain=="all" else [r for r in records if r["domain"]==domain]
            gaps=[r["deletion"]["faithfulness_gap_random_minus_top"] for r in subset]; edges=[r["edge_positive_importance_share"] for r in subset if r["edge_positive_importance_share"] is not None]
            output[system][domain]={"images":len(subset),"faithfulness_gap_mean":float(np.mean(gaps)),"faithfulness_gap_median":float(np.median(gaps)),"faithfulness_gap_positive_fraction":float(np.mean(np.asarray(gaps)>0)),"top_ranked_auc_mean":float(np.mean([r["deletion"]["top_ranked_auc"] for r in subset])),"random_auc_mean":float(np.mean([r["deletion"]["random_auc"] for r in subset])),"edge_positive_importance_share_mean":float(np.mean(edges)),"top5_mean_grid_radius_mean":float(np.mean([r["top5_mean_grid_radius"] for r in subset]))}
    return output


def plot_faithfulness(results:dict[str,list[dict[str,Any]]])->None:
    names=list(results); labels=["EfficientNet","CLIP probe","CLIP zero-shot","SigLIP2 probe","SigLIP2 zero-shot"]; fig,axes=plt.subplots(1,2,figsize=(14,5))
    for system,label in zip(names,labels):
        records=results[system]; fractions=np.asarray(records[0]["deletion"]["fractions"]); top=np.median([r["deletion"]["top_ranked_relative_margins"] for r in records],axis=0); random=np.median([r["deletion"]["random_relative_margins"] for r in records],axis=0); axes[0].plot(fractions,top,marker="o",label=f"{label} top"); axes[0].plot(fractions,random,linestyle="--",alpha=.7,label=f"{label} random")
    axes[0].set_xlabel("Fraction of regions deleted"); axes[0].set_ylabel("Median target margin / original margin"); axes[0].set_title("Median deletion curves"); axes[0].grid(alpha=.25); axes[0].legend(fontsize=7,ncol=2)
    gaps=[np.median([r["deletion"]["faithfulness_gap_random_minus_top"] for r in results[s]]) for s in names]; positive=[np.mean(np.asarray([r["deletion"]["faithfulness_gap_random_minus_top"] for r in results[s]])>0) for s in names]; bars=axes[1].bar(np.arange(len(names)),gaps); axes[1].set_xticks(np.arange(len(names)),labels,rotation=30,ha="right"); axes[1].set_ylabel("Median random AUC − top-ranked AUC"); axes[1].set_title("Perturbational faithfulness gap"); axes[1].axhline(0,color="black",lw=.8); axes[1].grid(axis="y",alpha=.25)
    for bar,fraction in zip(bars,positive): axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height(),f"{fraction:.0%} > 0",ha="center",va="bottom",fontsize=7)
    fig.tight_layout(); fig.savefig(RESULTS/"occlusion_faithfulness.png",dpi=180); plt.close(fig)


def summarize_existing()->None:
    payload=read_json(RESULTS/"occlusion_results.json"); results=payload["records_by_system"]; summary={"schema_version":1,"spatial_interpretation":"Aggregate 7x7 input-grid sensitivity only; edge cells are a spatial proxy and not a segmentation-derived background label.","systems":{},"probe_minus_zero_shot_edge_share":{}}
    for system,records in results.items():
        summary["systems"][system]={}
        groups={"all":records}
        groups.update({f"domain:{domain}":[r for r in records if r["domain"]==domain] for domain in ("sultana","fruitvision")})
        groups.update({f"class:{name}":[r for r in records if r["ground_truth_class"]==name] for name in CLASS_NAMES})
        groups.update({f"outcome:{value}":[r for r in records if r["frozen_correct"] is value] for value in (True,False)})
        for key,subset in groups.items():
            if not subset: continue
            grids=[]
            for record in subset:
                values=np.maximum(np.asarray([region["importance"] for region in record["regions"]]),0); grids.append(values/max(float(values.sum()),1e-12))
            edge=[r["edge_positive_importance_share"] for r in subset if r["edge_positive_importance_share"] is not None]; gaps=[r["deletion"]["faithfulness_gap_random_minus_top"] for r in subset]
            summary["systems"][system][key]={"images":len(subset),"mean_normalized_positive_importance_grid":np.mean(grids,axis=0).reshape(7,7).tolist(),"edge_positive_importance_share_mean":float(np.mean(edge)),"top5_mean_grid_radius_mean":float(np.mean([r["top5_mean_grid_radius"] for r in subset])),"faithfulness_gap_median":float(np.median(gaps)),"faithfulness_positive_fraction":float(np.mean(np.asarray(gaps)>0))}
    for family in ("clip","siglip2"):
        summary["probe_minus_zero_shot_edge_share"][family]={}
        for domain in ("sultana","fruitvision"):
            probe=summary["systems"][f"{family}_linear_probe"][f"domain:{domain}"]["edge_positive_importance_share_mean"]; zero=summary["systems"][f"{family}_strict_zero_shot"][f"domain:{domain}"]["edge_positive_importance_share_mean"]
            summary["probe_minus_zero_shot_edge_share"][family][domain]=probe-zero
    save_json(summary,RESULTS/"attribution_spatial_summary.json")
    plot_faithfulness(results)
    names=list(results); labels=["EfficientNet","CLIP probe","CLIP zero-shot","SigLIP2 probe","SigLIP2 zero-shot"]; fig,axes=plt.subplots(1,5,figsize=(15,3.4)); vmax=max(np.max(summary["systems"][s]["all"]["mean_normalized_positive_importance_grid"]) for s in names)
    for ax,system,label in zip(axes,names,labels):
        im=ax.imshow(summary["systems"][system]["all"]["mean_normalized_positive_importance_grid"],vmin=0,vmax=vmax,cmap="inferno"); ax.set_title(label,fontsize=9); ax.set_xticks(range(7)); ax.set_yticks(range(7))
    fig.colorbar(im,ax=axes.ravel().tolist(),label="Mean normalized positive sensitivity",shrink=.75); fig.suptitle("Aggregate occlusion sensitivity by normalized input-grid region"); fig.savefig(RESULTS/"occlusion_spatial_distribution.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    print("Created aggregate spatial attribution summary without image overlays.")


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",default="v2/configs/representation_interpretability.yaml"); parser.add_argument("--summarize-only",action="store_true"); args=parser.parse_args(); config=load_config(root_path(args.config)); summarize_existing() if args.summarize_only else run(config); return 0


if __name__=="__main__": raise SystemExit(main())
