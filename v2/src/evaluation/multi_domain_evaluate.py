"""Locked FruitVision inference and three-domain robustness analysis for Experiment 014."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, rankdata, spearmanr
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from v2.src.datasets.fruit_dataset import default_transform
from v2.src.datasets.transfer_transforms import build_transfer_transform
from v2.src.evaluation.cross_domain_evaluate import (
    CLASS_NAMES, CLASS_TO_IDX, SYSTEMS, ManifestDataset, bootstrap_metrics,
    plot_confusion, semantic_errors, source_summary, summarize_profile, system_metrics,
)
from v2.src.foundation.linear_probe import LinearProbe
from v2.src.foundation.prompt_registry import encode_class_prototypes, load_prompt_registry
from v2.src.foundation.registry import build_foundation_encoder
from v2.src.foundation.similarity_classifier import build_zero_shot_model
from v2.src.models.transfer_registry import build_transfer_model
from v2.src.training.train import build_model as build_cnn
from v2.src.training.utils import load_config, resolve_device, save_json, set_global_seed, sha256_file, utc_timestamp


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "v2/results/experiment_014_multi_domain"
CACHE = ROOT / "v2/cache/experiment_014_multi_domain"
FRUITVISION_ROOT = ROOT / "v2/data/external/fruitvision/Fruits Original"
PHASE13 = ROOT / "v2/results/experiment_013_cross_domain"


def path(value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def read_json(value: str | Path) -> Any:
    with path(value).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_manifest(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = path(config["outputs"]["manifest"])
    recorded = path(config["outputs"]["manifest_hash"]).read_text(encoding="utf-8").split()[0]
    actual = sha256_file(manifest_path)
    manifest = read_json(manifest_path)
    if actual != recorded or manifest["status"] != "frozen_before_fruitvision_inference" or len(manifest["images"]) != 4185:
        raise RuntimeError("Frozen FruitVision manifest identity/count changed.")
    return manifest


def verify_checkpoints(config: dict[str, Any]) -> None:
    output = RESULTS / "checkpoint_verification.json"
    if output.exists():
        raise FileExistsError("Refusing to overwrite Experiment 014 checkpoint verification.")
    prior = read_json(PHASE13 / "checkpoint_verification.json")
    if prior["status"] != "PASS":
        raise RuntimeError("Experiment 013 checkpoint verification was not PASS.")
    verify_manifest(config)
    records, failures = {}, []
    for system_id, spec in SYSTEMS.items():
        frozen = prior["systems"][system_id]
        current_config_hash = sha256_file(path(spec["config"]))
        record: dict[str, Any] = {
            "condition": spec["condition"].replace("013", "014"), "name": spec["name"],
            "configuration": spec["config"], "configuration_sha256_experiment_013": frozen["configuration_sha256"],
            "configuration_sha256_current": current_config_hash, "preprocessing_identity": frozen["preprocessing_identity"],
        }
        passed = current_config_hash == frozen["configuration_sha256"]
        if spec["kind"] in {"cnn", "transfer"}:
            checkpoint_path = path(frozen["checkpoint"])
            actual = sha256_file(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            record.update({"checkpoint": frozen["checkpoint"], "checkpoint_sha256_experiment_013": frozen["checkpoint_sha256_actual"], "checkpoint_sha256_current": actual, "class_mapping": checkpoint["class_to_idx"]})
            passed &= actual == frozen["checkpoint_sha256_actual"] and checkpoint["class_to_idx"] == CLASS_TO_IDX
        elif spec["kind"] == "probe":
            probe_path = path(frozen["probe_checkpoint"])
            probe_actual = sha256_file(probe_path)
            encoder_path = Path(read_json(spec["source"])["encoder"]["checkpoint_file"]["path"])
            encoder_actual = sha256_file(encoder_path)
            checkpoint = torch.load(probe_path, map_location="cpu", weights_only=True)
            record.update({"probe_checkpoint": frozen["probe_checkpoint"], "probe_sha256_experiment_013": frozen["probe_checkpoint_sha256_actual"], "probe_sha256_current": probe_actual, "encoder_sha256_experiment_013": frozen["encoder_checkpoint_sha256_actual"], "encoder_sha256_current": encoder_actual, "encoder_revision": frozen["encoder_revision"], "class_mapping": checkpoint["class_mapping"]})
            passed &= probe_actual == frozen["probe_checkpoint_sha256_actual"] and encoder_actual == frozen["encoder_checkpoint_sha256_actual"] and checkpoint["class_mapping"] == CLASS_TO_IDX
        else:
            encoder_path = Path(frozen["encoder_checkpoint"])
            encoder_actual = sha256_file(encoder_path)
            registry_path = path(frozen["prompt_registry"])
            registry_actual = sha256_file(registry_path)
            record.update({"encoder_checkpoint": frozen["encoder_checkpoint"], "encoder_sha256_experiment_013": frozen["encoder_checkpoint_sha256_actual"], "encoder_sha256_current": encoder_actual, "encoder_revision": frozen["encoder_revision"], "prompt_registry": frozen["prompt_registry"], "prompt_registry_sha256_experiment_013": frozen["prompt_registry_sha256_actual"], "prompt_registry_sha256_current": registry_actual, "prompt_family": "P1", "class_mapping": CLASS_TO_IDX})
            passed &= encoder_actual == frozen["encoder_checkpoint_sha256_actual"] and registry_actual == frozen["prompt_registry_sha256_actual"]
        record["status"] = "PASS" if passed else "FAIL"
        records[system_id] = record
        if not passed:
            failures.append(system_id)
    payload = {"schema_version": 1, "experiment_id": config["experiment"]["id"], "verified_at": utc_timestamp(), "experiment_013_verification_sha256": sha256_file(PHASE13 / "checkpoint_verification.json"), "fruitvision_manifest_sha256": sha256_file(path(config["outputs"]["manifest"])), "same_nine_systems": list(SYSTEMS), "systems": records, "failed_conditions": failures, "status": "PASS" if not failures else "BLOCKED"}
    save_json(payload, output)
    if failures:
        raise RuntimeError(f"Checkpoint identity failures: {failures}")
    print("Verified all nine system identities against Experiment 013.")


def rows_from_scores(scores: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor, score_kind: str) -> list[dict[str, Any]]:
    top = torch.topk(scores.detach().cpu().float(), 2, dim=1)
    return [{"index": int(index), "true_index": int(label), "predicted_index": int(pred), "score": float(best), "margin": float(best-second), "score_kind": score_kind} for label, index, pred, best, second in zip(labels.tolist(), indices.tolist(), top.indices[:,0].tolist(), top.values[:,0].tolist(), top.values[:,1].tolist())]


def logits_pass(model: torch.nn.Module, dataset: ManifestDataset, device: torch.device, score_kind: str) -> tuple[list[dict[str, Any]], dict[str, float]]:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    rows, forward = [], 0.0; wall_start = time.perf_counter()
    with torch.inference_mode():
        for images, labels, indices in loader:
            images = images.to(device); start = time.perf_counter(); scores = model(images); forward += time.perf_counter()-start
            rows.extend(rows_from_scores(scores, labels, indices, score_kind))
    return rows, {"model_forward_seconds":forward,"model_forward_ms_per_image":forward*1000/len(dataset),"evaluation_loop_wall_seconds":time.perf_counter()-wall_start}


def embedding_pass(encode: Callable[[torch.Tensor], torch.Tensor], dataset: ManifestDataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    embeddings, labels_all, indices_all, forward = [], [], [], 0.0; wall_start=time.perf_counter()
    with torch.inference_mode():
        for images, labels, indices in loader:
            images=images.to(device); start=time.perf_counter(); value=F.normalize(encode(images).float(),p=2,dim=1); forward += time.perf_counter()-start
            embeddings.append(value.cpu()); labels_all.append(labels); indices_all.append(indices)
    return torch.cat(embeddings),torch.cat(labels_all),torch.cat(indices_all),{"encoder_forward_seconds":forward,"encoder_forward_ms_per_image":forward*1000/len(dataset),"evaluation_loop_wall_seconds":time.perf_counter()-wall_start}


def finalize_metrics(system_id: str, rows: list[dict[str, Any]], timing: dict[str, Any], device: torch.device, extra: dict[str, Any] | None=None) -> dict[str, Any]:
    spec=SYSTEMS[system_id]; metrics=system_metrics(rows)
    metrics.update({"condition":spec["condition"].replace("013","014"),"system":system_id,"display_name":spec["name"],"evaluated_at":utc_timestamp(),"single_fruitvision_pass":True,"zero_target_domain_adaptation":True,"timing":{"device":str(device),**timing,"scope":"Model forward after deterministic preprocessing; decoding/transforms excluded."},**(extra or {})})
    plot_confusion(metrics["confusion_matrix"],spec["name"],RESULTS/f"confusion_matrix_{metrics['condition'].lower().replace('-','_')}_{system_id}.png")
    return metrics


def save_condition(system_id: str, rows: list[dict[str, Any]], metrics: dict[str, Any], ledger: dict[str, Any], ledger_path: Path) -> None:
    save_json({"rows":rows,"metrics":metrics},CACHE/f"{system_id}_result.json")
    ledger["completed_systems"].append(system_id); ledger["last_completed_at"]=utc_timestamp(); save_json(ledger,ledger_path)
    print(f"{system_id}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}",flush=True)


def infer(config: dict[str, Any]) -> None:
    verification=read_json(RESULTS/"checkpoint_verification.json")
    if verification["status"]!="PASS": raise RuntimeError("Checkpoint verification must pass.")
    manifest=verify_manifest(config); metrics_path=RESULTS/"fruitvision_metrics.json"; predictions_path=RESULTS/"fruitvision_predictions.csv"
    if metrics_path.exists() or predictions_path.exists(): raise FileExistsError("Refusing to rerun completed FruitVision inference.")
    CACHE.mkdir(parents=True,exist_ok=True); ledger_path=RESULTS/"fruitvision_inference_access.json"
    ledger=read_json(ledger_path) if ledger_path.exists() else {"schema_version":1,"experiment_id":config["experiment"]["id"],"manifest_sha256":sha256_file(path(config["outputs"]["manifest"])),"started_at":utc_timestamp(),"completed_systems":[],"shared_encoder_policy":"CLIP and SigLIP2 each use one image traversal shared by their frozen probe and strict-P1 heads; each condition receives exactly one prediction per image."}
    device=resolve_device(str(config["evaluation"]["device"])); set_global_seed(42)
    all_rows,all_metrics={},{}
    for system_id in ledger["completed_systems"]:
        cached=read_json(CACHE/f"{system_id}_result.json"); all_rows[system_id]=cached["rows"]; all_metrics[system_id]=cached["metrics"]

    for system_id in ("custom_cnn","resnet50","efficientnet_b0","mobilenetv3_large"):
        if system_id in ledger["completed_systems"]: continue
        spec=SYSTEMS[system_id]; cfg=load_config(path(spec["config"])); print(f"Running sole FruitVision pass for {spec['name']}...",flush=True)
        if spec["kind"]=="cnn":
            transform=default_transform(int(cfg["model"]["input_size"])); model=build_cnn(cfg["model"]).to(device); checkpoint_path=path(cfg["outputs"]["checkpoint"])
        else:
            transform=build_transfer_transform("test",cfg["preprocessing"]); model=build_transfer_model(cfg["model"],load_pretrained=False).to(device); checkpoint_path=path(cfg["outputs"]["checkpoint"])
        checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=True); model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
        dataset=ManifestDataset(manifest,FRUITVISION_ROOT,transform); rows,timing=logits_pass(model,dataset,device,"uncalibrated_logit"); metrics=finalize_metrics(system_id,rows,timing,device)
        save_condition(system_id,rows,metrics,ledger,ledger_path); all_rows[system_id],all_metrics[system_id]=rows,metrics
        del model,dataset; gc.collect()

    if "dinov2_linear_probe" not in ledger["completed_systems"]:
        system_id="dinov2_linear_probe"; cfg=load_config(path(SYSTEMS[system_id]["config"])); print("Running sole FruitVision pass for DINOv2 + linear probe...",flush=True)
        encoder=build_foundation_encoder(cfg["encoder"],device); dataset=ManifestDataset(manifest,FRUITVISION_ROOT,encoder.preprocess); embeddings,labels,indices,enc_timing=embedding_pass(encoder.encode,dataset,device)
        checkpoint=torch.load(path(cfg["outputs"]["probe_checkpoint"]),map_location=device,weights_only=True); probe=LinearProbe(int(cfg["encoder"]["embedding_dim"]),6).to(device); probe.load_state_dict(checkpoint["model_state_dict"]); probe.eval()
        start=time.perf_counter(); scores=probe(embeddings.to(device)); probe_seconds=time.perf_counter()-start; rows=rows_from_scores(scores,labels,indices,"uncalibrated_linear_probe_logit")
        timing={**enc_timing,"probe_forward_seconds":probe_seconds,"combined_forward_ms_per_image":1000*(enc_timing["encoder_forward_seconds"]+probe_seconds)/len(dataset)}; metrics=finalize_metrics(system_id,rows,timing,device)
        torch.save({"embeddings":embeddings,"labels":labels,"sample_ids":[r["identifier"] for r in manifest["images"]]},CACHE/f"{system_id}_fruitvision.pt")
        save_condition(system_id,rows,metrics,ledger,ledger_path); all_rows[system_id],all_metrics[system_id]=rows,metrics; del encoder,probe,dataset,scores,embeddings; gc.collect()

    for family in ("clip","siglip2"):
        probe_id=f"{family}_linear_probe"; zero_id=f"{family}_strict_zero_shot"
        if probe_id in ledger["completed_systems"] and zero_id in ledger["completed_systems"]: continue
        if probe_id in ledger["completed_systems"] or zero_id in ledger["completed_systems"]: raise RuntimeError(f"Partial shared {family} pair cannot be rerun safely.")
        probe_cfg=load_config(path(SYSTEMS[probe_id]["config"])); zero_cfg=load_config(path(SYSTEMS[zero_id]["config"])); print(f"Running one shared {family.upper()} image pass for frozen probe and strict P1 heads...",flush=True)
        model=build_zero_shot_model(zero_cfg["model"],device); registry=load_prompt_registry(path(zero_cfg["prompts"]["registry_path"]),zero_cfg["prompts"]["registry_sha256"]); prototypes,prototype_meta=encode_class_prototypes(model,registry,"P1")
        dataset=ManifestDataset(manifest,FRUITVISION_ROOT,model.preprocess); embeddings,labels,indices,enc_timing=embedding_pass(model.encode_images,dataset,device)
        checkpoint=torch.load(path(probe_cfg["outputs"]["probe_checkpoint"]),map_location=device,weights_only=True); probe=LinearProbe(int(probe_cfg["encoder"]["embedding_dim"]),6).to(device); probe.load_state_dict(checkpoint["model_state_dict"]); probe.eval()
        start=time.perf_counter(); probe_scores=probe(embeddings.to(device)); probe_seconds=time.perf_counter()-start
        start=time.perf_counter(); zero_scores=embeddings.to(device)@prototypes.to(device).t(); similarity_seconds=time.perf_counter()-start
        probe_rows=rows_from_scores(probe_scores,labels,indices,"uncalibrated_linear_probe_logit"); zero_rows=rows_from_scores(zero_scores,labels,indices,"cosine_similarity")
        probe_metrics=finalize_metrics(probe_id,probe_rows,{**enc_timing,"probe_forward_seconds":probe_seconds,"combined_forward_ms_per_image":1000*(enc_timing["encoder_forward_seconds"]+probe_seconds)/len(dataset)},device,{"shared_encoder_traversal_with":zero_id})
        zero_metrics=finalize_metrics(zero_id,zero_rows,{**enc_timing,"similarity_forward_seconds":similarity_seconds,"combined_forward_ms_per_image":1000*(enc_timing["encoder_forward_seconds"]+similarity_seconds)/len(dataset)},device,{"prompt_family":"P1","prompt_registry_sha256":registry.sha256,"prototype_generation":prototype_meta,"classifier_training":False,"shared_encoder_traversal_with":probe_id})
        torch.save({"embeddings":embeddings,"labels":labels,"sample_ids":[r["identifier"] for r in manifest["images"]]},CACHE/f"{probe_id}_fruitvision.pt")
        for system_id,rows,metrics in ((probe_id,probe_rows,probe_metrics),(zero_id,zero_rows,zero_metrics)):
            save_condition(system_id,rows,metrics,ledger,ledger_path); all_rows[system_id],all_metrics[system_id]=rows,metrics
        del model,probe,dataset,embeddings,probe_scores,zero_scores; gc.collect()

    fieldnames=["condition","system","image_identifier","ground_truth_class","predicted_class","correct","fruit_identity_correct","freshness_condition_correct","score","margin","score_kind","score_warning"]
    with predictions_path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fieldnames); writer.writeheader()
        for system_id,spec in SYSTEMS.items():
            for item in all_rows[system_id]:
                true_name,pred_name=CLASS_NAMES[item["true_index"]],CLASS_NAMES[item["predicted_index"]]
                writer.writerow({"condition":spec["condition"].replace("013","014"),"system":system_id,"image_identifier":manifest["images"][item["index"]]["identifier"],"ground_truth_class":true_name,"predicted_class":pred_name,"correct":true_name==pred_name,"fruit_identity_correct":true_name.split("_",1)[1]==pred_name.split("_",1)[1],"freshness_condition_correct":true_name.split("_",1)[0]==pred_name.split("_",1)[0],"score":item["score"],"margin":item["margin"],"score_kind":item["score_kind"],"score_warning":"Uncalibrated and not comparable across systems."})
    save_json({"schema_version":1,"experiment_id":config["experiment"]["id"],"manifest_sha256":ledger["manifest_sha256"],"systems":all_metrics},metrics_path)
    ledger.update({"completed_at":utc_timestamp(),"status":"COMPLETE_ONE_PASS_PER_SYSTEM","fruitvision_passes_per_system":1}); save_json(ledger,ledger_path)


def load_predictions() -> dict[str,list[dict[str,str]]]:
    rows={key:[] for key in SYSTEMS}
    with (RESULTS/"fruitvision_predictions.csv").open(newline="",encoding="utf-8") as handle:
        for row in csv.DictReader(handle): rows[row["system"]].append(row)
    return rows


def rank_values(values: dict[str,float]) -> dict[str,float]:
    keys=list(SYSTEMS); ranks=rankdata([-values[key] for key in keys],method="average")
    return {key:float(rank) for key,rank in zip(keys,ranks)}


def multi_domain_figures(summary: dict[str,Any], ranking: dict[str,Any], class_data: dict[str,Any]) -> None:
    keys=list(SYSTEMS); names=[SYSTEMS[k]["name"] for k in keys]; x=np.arange(len(keys))
    fig,ax=plt.subplots(figsize=(14,6)); width=.25
    for offset,(domain,label) in zip((-width,0,width),(("source_accuracy","Source"),("sultana_accuracy","Sultana"),("fruitvision_accuracy","FruitVision"))): ax.bar(x+offset,[summary[k][domain] for k in keys],width,label=label)
    ax.set_xticks(x,names,rotation=35,ha="right"); ax.set_ylim(0,1); ax.set_ylabel("Accuracy"); ax.legend(); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(RESULTS/"multi_domain_accuracy.png",dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,6)); ax.bar(x-.18,[summary[k]["sultana_retention"] for k in keys],.36,label="Sultana"); ax.bar(x+.18,[summary[k]["fruitvision_retention"] for k in keys],.36,label="FruitVision"); ax.set_xticks(x,names,rotation=35,ha="right"); ax.set_ylabel("Accuracy retention vs source"); ax.legend(); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(RESULTS/"external_retention.png",dpi=170); plt.close(fig)
    ordered=ranking["robustness_order"]; fig,ax=plt.subplots(figsize=(11,6)); ax.barh([SYSTEMS[k]["name"] for k in ordered][::-1],[summary[k]["worst_external_accuracy"] for k in ordered][::-1]); ax.set_xlim(0,1); ax.set_xlabel("Worst external accuracy"); ax.grid(axis="x",alpha=.25); fig.tight_layout(); fig.savefig(RESULTS/"worst_domain_accuracy.png",dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6)); domains=("source","sultana","fruitvision");
    for key in keys: ax.plot(domains,[ranking["domain_ranks"][d][key] for d in domains],marker="o",label=SYSTEMS[key]["name"])
    ax.invert_yaxis(); ax.set_ylabel("Accuracy rank (1 is best)"); ax.grid(alpha=.25); ax.legend(ncol=3,fontsize=8); fig.tight_layout(); fig.savefig(RESULTS/"rank_stability.png",dpi=170); plt.close(fig)
    matrix=[]; row_labels=[]
    for key in keys:
        for domain in ("source","sultana","fruitvision"):
            matrix.append([class_data["systems"][key][c][f"{domain}_recall"] for c in CLASS_NAMES]); row_labels.append(f"{SYSTEMS[key]['name']} | {domain}")
    fig,ax=plt.subplots(figsize=(10,14)); im=ax.imshow(matrix,vmin=0,vmax=1,cmap="viridis",aspect="auto"); fig.colorbar(im,ax=ax,label="Recall"); ax.set_xticks(range(6),CLASS_NAMES,rotation=35,ha="right"); ax.set_yticks(range(len(row_labels)),row_labels,fontsize=7); fig.tight_layout(); fig.savefig(RESULTS/"class_recall_heatmap.png",dpi=170); plt.close(fig)


def pca_figure() -> dict[str,Any]:
    models=(("DINOv2","dinov2_linear_probe","v2/cache/dinov2/test.pt"),("SigLIP2","siglip2_linear_probe","v2/cache/siglip2/test.pt")); fig,axes=plt.subplots(2,2,figsize=(14,11)); result={}; colors=plt.get_cmap("tab10")
    for row,(name,system_id,source_file) in enumerate(models):
        source=torch.load(path(source_file),map_location="cpu",weights_only=False); sultana=torch.load(ROOT/f"v2/cache/experiment_013_cross_domain/{system_id}_external.pt",map_location="cpu",weights_only=False); fruitvision=torch.load(CACHE/f"{system_id}_fruitvision.pt",map_location="cpu",weights_only=False)
        embeddings=[]; labels=[]; domains=[]; rng=np.random.default_rng(42)
        for domain,payload in (("Source",source),("Sultana",sultana),("FruitVision",fruitvision)):
            y=payload["labels"].numpy(); selected=[]
            for c in range(6): selected.extend(rng.choice(np.where(y==c)[0],200,replace=False).tolist())
            embeddings.append(payload["embeddings"][selected].numpy()); labels.extend(y[selected]); domains.extend([domain]*1200)
        combined=np.vstack(embeddings); reducer=PCA(n_components=2,random_state=42); projected=reducer.fit_transform(combined); domains=np.asarray(domains); labels=np.asarray(labels)
        for domain in ("Source","Sultana","FruitVision"): axes[row,0].scatter(projected[domains==domain,0],projected[domains==domain,1],s=5,alpha=.3,label=domain)
        for c,class_name in enumerate(CLASS_NAMES): axes[row,1].scatter(projected[labels==c,0],projected[labels==c,1],s=5,alpha=.3,label=class_name,color=colors(c))
        axes[row,0].set_title(f"{name}: domain"); axes[row,1].set_title(f"{name}: class"); axes[row,0].legend(markerscale=2); axes[row,1].legend(markerscale=2,fontsize=8)
        result[name.lower()]={"samples_per_domain":1200,"explained_variance_ratio":reducer.explained_variance_ratio_.tolist()}
    fig.suptitle("Descriptive PCA of frozen representations across three domains"); fig.tight_layout(); fig.savefig(RESULTS/"pca_three_domains.png",dpi=170); plt.close(fig); result["boundary"]="Exploratory only; not used for selection or adaptation."; return result


def analyze(config: dict[str,Any]) -> None:
    fruit_metrics=read_json(RESULTS/"fruitvision_metrics.json")["systems"]; sultana_metrics=read_json(PHASE13/"metrics_by_system.json")["systems"]; rows=load_predictions(); summary={}; domain_values={"source":{},"sultana":{},"fruitvision":{}}; class_data={"schema_version":1,"systems":{}}
    for system_id,spec in SYSTEMS.items():
        source_acc,source_f1,source_recalls=source_summary(spec); a=sultana_metrics[system_id]; b=fruit_metrics[system_id]
        domain_values["source"][system_id]=source_acc; domain_values["sultana"][system_id]=a["accuracy"]; domain_values["fruitvision"][system_id]=b["accuracy"]
        a_ret,b_ret=a["accuracy"]/source_acc,b["accuracy"]/source_acc
        summary[system_id]={"display_name":spec["name"],"source_accuracy":source_acc,"sultana_accuracy":a["accuracy"],"fruitvision_accuracy":b["accuracy"],"mean_external_accuracy":(a["accuracy"]+b["accuracy"])/2,"worst_external_accuracy":min(a["accuracy"],b["accuracy"]),"source_weighted_f1":source_f1,"sultana_weighted_f1":a["weighted_f1"],"fruitvision_weighted_f1":b["weighted_f1"],"mean_external_weighted_f1":(a["weighted_f1"]+b["weighted_f1"])/2,"worst_external_weighted_f1":min(a["weighted_f1"],b["weighted_f1"]),"worst_external_macro_f1":min(a["macro_f1"],b["macro_f1"]),"sultana_retention":a_ret,"fruitvision_retention":b_ret,"mean_external_retention":(a_ret+b_ret)/2,"worst_external_retention":min(a_ret,b_ret)}
        class_data["systems"][system_id]={}
        for name in CLASS_NAMES:
            ar=float(a["per_class"][name]["recall"]); br=float(b["per_class"][name]["recall"]); sr=source_recalls[name]
            class_data["systems"][system_id][name]={"source_recall":sr,"sultana_recall":ar,"fruitvision_recall":br,"mean_external_recall":(ar+br)/2,"worst_external_recall":min(ar,br),"external_recall_range":abs(ar-br)}
    robustness_order=sorted(SYSTEMS,key=lambda key:(-summary[key]["worst_external_accuracy"],-summary[key]["mean_external_accuracy"],key)); ranks={domain:rank_values(values) for domain,values in domain_values.items()}
    volatility={key:{"source_rank":ranks["source"][key],"sultana_rank":ranks["sultana"][key],"fruitvision_rank":ranks["fruitvision"][key],"best_rank":min(ranks[d][key] for d in ranks),"worst_rank":max(ranks[d][key] for d in ranks),"rank_range":max(ranks[d][key] for d in ranks)-min(ranks[d][key] for d in ranks)} for key in SYSTEMS}
    ranking={"schema_version":1,"protocol":{"primary":"worst_external_accuracy","secondary":"mean_external_accuracy","status":"frozen_before_fruitvision_results"},"robustness_order":robustness_order,"domain_ranks":ranks,"rank_volatility":volatility}; save_json(ranking,RESULTS/"ranking_analysis.json")
    correlations={}
    for first,second in (("source","sultana"),("source","fruitvision"),("sultana","fruitvision")):
        x=[ranks[first][k] for k in SYSTEMS]; y=[ranks[second][k] for k in SYSTEMS]; sp=spearmanr(x,y); kt=kendalltau(x,y)
        correlations[f"{first}_vs_{second}"]={"spearman_rho":float(sp.statistic),"spearman_p_value_descriptive":float(sp.pvalue),"kendall_tau_b":float(kt.statistic),"kendall_p_value_descriptive":float(kt.pvalue),"tie_method":"average ranks; Kendall tau-b"}
    save_json({"schema_version":1,"correlations":correlations,"inferential_boundary":"Descriptive ranking stability for nine fixed systems; p-values are not used for model selection."},RESULTS/"rank_correlations.json")

    manifest=verify_manifest(config); true=np.asarray([CLASS_TO_IDX[r["class"]] for r in manifest["images"]]); rng=np.random.default_rng(42); class_indices=[np.where(true==c)[0] for c in range(6)]; samples=np.concatenate([rng.choice(idx,size=(5000,len(idx)),replace=True) for idx in class_indices],axis=1); bootstrap={}; boot_acc={}
    for system_id in SYSTEMS:
        pred=np.asarray([CLASS_TO_IDX[r["predicted_class"]] for r in rows[system_id]]); acc,mf1=bootstrap_metrics(true,pred,samples); boot_acc[system_id]=acc
        bootstrap[system_id]={"accuracy":{"estimate":fruit_metrics[system_id]["accuracy"],"lower":float(np.quantile(acc,.025)),"upper":float(np.quantile(acc,.975))},"macro_f1":{"estimate":fruit_metrics[system_id]["macro_f1"],"lower":float(np.quantile(mf1,.025)),"upper":float(np.quantile(mf1,.975))}}
    save_json({"schema_version":1,"seed":42,"resamples":5000,"method":"95% class-stratified bootstrap interval over the FruitVision evaluation sample.","systems":bootstrap},RESULTS/"fruitvision_bootstrap_intervals.json")
    paired={}
    for left,right in config["uncertainty"]["paired_comparisons"]:
        delta=boot_acc[left]-boot_acc[right]; paired[f"{left}_minus_{right}"]={"left":left,"right":right,"observed_accuracy_difference":fruit_metrics[left]["accuracy"]-fruit_metrics[right]["accuracy"],"mean_paired_bootstrap_accuracy_difference":float(delta.mean()),"lower":float(np.quantile(delta,.025)),"upper":float(np.quantile(delta,.975))}
    save_json({"schema_version":1,"seed":42,"resamples":5000,"method":"Paired 95% class-stratified bootstrap interval over the FruitVision evaluation sample.","comparisons":paired},RESULTS/"paired_bootstrap_comparisons.json")

    aggregate_classes={}
    for name in CLASS_NAMES:
        values=[value for system in class_data["systems"].values() for value in (system[name]["sultana_recall"],system[name]["fruitvision_recall"])]
        ranges=[system[name]["external_recall_range"] for system in class_data["systems"].values()]
        aggregate_classes[name]={"mean_recall_across_models_and_external_domains":float(np.mean(values)),"minimum_recall":float(np.min(values)),"mean_between_external_domain_range":float(np.mean(ranges)),"maximum_between_external_domain_range":float(np.max(ranges))}
    class_data["aggregate_by_class"]=aggregate_classes; save_json(class_data,RESULTS/"class_recall_across_domains.json")
    fruit_errors={system_id:semantic_errors([{"true_index":CLASS_TO_IDX[r["ground_truth_class"]],"predicted_index":CLASS_TO_IDX[r["predicted_class"]]} for r in values]) for system_id,values in rows.items()}; sultana_errors=read_json(PHASE13/"semantic_error_analysis.json")["systems"]
    save_json({"schema_version":1,"systems":{key:{"sultana":sultana_errors[key],"fruitvision":fruit_errors[key]} for key in SYSTEMS}},RESULTS/"semantic_error_comparison.json")
    save_json({"schema_version":1,"ranking_protocol":ranking["protocol"],"systems":summary,"zero_shot_minus_probe":{"clip":{"sultana":summary["clip_strict_zero_shot"]["sultana_accuracy"]-summary["clip_linear_probe"]["sultana_accuracy"],"fruitvision":summary["clip_strict_zero_shot"]["fruitvision_accuracy"]-summary["clip_linear_probe"]["fruitvision_accuracy"]},"siglip2":{"sultana":summary["siglip2_strict_zero_shot"]["sultana_accuracy"]-summary["siglip2_linear_probe"]["sultana_accuracy"],"fruitvision":summary["siglip2_strict_zero_shot"]["fruitvision_accuracy"]-summary["siglip2_linear_probe"]["fruitvision_accuracy"]}}},RESULTS/"multi_domain_summary.json")
    prior_profile=read_json(PHASE13/"domain_profile.json"); external_records=read_json(RESULTS/"fruitvision_dataset_audit.json")["records_for_manifest"]; save_json({"schema_version":1,"source_test":prior_profile["source_test"],"sultana_originals":prior_profile["external_originals"],"fruitvision_originals":summarize_profile(external_records),"processing":"Decoded RGB; 64x64 BOX-resized descriptive summaries only; evaluation inputs were not harmonized.","aggregate_shift_not_causal_attribution":True},RESULTS/"domain_profile.json")
    save_json(pca_figure(),RESULTS/"pca_analysis.json"); multi_domain_figures(summary,ranking,class_data)
    print("Completed FruitVision uncertainty and all three-domain robustness analyses.")


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("stage",choices=("verify","infer","analyze")); parser.add_argument("--config",default="v2/configs/multi_domain_robustness.yaml"); args=parser.parse_args(); config=load_config(path(args.config)); {"verify":verify_checkpoints,"infer":infer,"analyze":analyze}[args.stage](config); return 0


if __name__=="__main__": raise SystemExit(main())
