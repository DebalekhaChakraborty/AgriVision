# FruitVision Dataset Provenance — Experiment 014

## Registered dataset release

| Field | Verified value |
|---|---|
| Dataset | FruitVision: A Benchmark Dataset for Fresh, Rotten, and Formalin-mixed Fruit Detection |
| Contributors | Md Hasan Imam Bijoy; Syeda Zarin Tasnim; Syed Ali Awsaf; Md Zahid Hasan |
| Repository | Mendeley Data |
| DOI | [10.17632/xkbjx8959c.2](https://data.mendeley.com/datasets/xkbjx8959c/2) |
| Version | 2 |
| Dataset publication date | 15 January 2025 |
| **Dataset licence** | **CC BY-NC-ND 4.0**, as displayed by Mendeley for Version 2 |
| Institutions | Daffodil International University; Leading University |
| Associated article | [Data in Brief 61 (2025), 111752](https://pmc.ncbi.nlm.nih.gov/articles/PMC12221502/) |

The dataset licence is not inferred from the associated article. The article itself uses a different Creative Commons licence; Experiment 014 follows the official Mendeley dataset metadata and treats the image data as CC BY-NC-ND 4.0. Images are used locally for research evaluation and are not redistributed.

## Collection and taxonomy

The associated article reports collection from fruit markets and gardens across Sylhet, Bangladesh, from 18 April through 31 July 2024. The devices were Apple iPhone 15 Pro Max, Redmi POCO M2 Reloaded, and Redmi Note 9 Pro. The release contains five fruits—apple, banana, grape, mango, and orange—under fresh, rotten, and formalin-mixed conditions, for 15 folders/classes. Agricultural experts assisted classification.

The article reports 10,154 originals. Its table labels the post-augmentation counts as 81,232; because those per-class figures include eight times the original counts, this document describes that number exactly as the article reports it rather than deriving a separate net-new count.

## Authoritative raw/augmented packaging gate

The official Mendeley Version 2 file API exposes two distinct archives, so separation is reliable before download:

| Official file | File ID | Bytes | SHA-256 | Experiment 014 policy |
|---|---|---:|---|---|
| `Original Image.zip` | `7d97a2e2-d501-4b8c-9fff-699433c001ea` | 898,826,798 | `c72a8d8d9b1f44f8f4a7f31e1420da7f356de5eb4f18e698bbe692ca4d1a18fc` | Eligible source archive; hash and CRC must pass |
| `Augmented-Resized Image.zip` | `5dfc18cf-0f8b-4c06-ac53-141c17ee0b76` | 4,787,094,558 | `06f550197ec8ab47eff8abea0eac6051565fe3f0a0ceaeb3e3bd3c5132d0a38c` | Prohibited; never download or extract |

## Registered six-class subset

Only these raw/original folders are eligible:

| Canonical class | Article original count |
|---|---:|
| `fresh_apple` | 765 |
| `rotten_apple` | 630 |
| `fresh_banana` | 749 |
| `rotten_banana` | 632 |
| `fresh_orange` | 753 |
| `rotten_orange` | 656 |
| **Expected eligible total** | **4,185** |

All grape and mango folders are excluded. Every formalin-mixed folder is excluded without remapping, calibration, or analysis. Folder spelling and final counts will be verified directly against the archive and recorded in the frozen manifest.

## Local verification outcome

The official `Original Image.zip` download matched the registered byte size and SHA-256 above, and a complete ZIP CRC test passed. The six eligible folders were extracted without importing any grape, mango, or formalin-mixed folder. Direct enumeration reproduced the article counts exactly: 765 fresh apple, 630 rotten apple, 749 fresh banana, 632 rotten banana, 753 fresh orange, and 656 rotten orange photographs (4,185 total). Four `desktop.ini` auxiliary files were excluded because they are not images.

All 4,185 selected records decoded as RGB JPEG. The raw archive was removed after verified extraction to avoid retaining an unnecessary duplicate; the selected local photographs remain under the Git-ignored external-data path. The separately packaged augmented archive was never downloaded. The immutable evaluation inventory and per-image hashes are in `v2/results/experiment_014_multi_domain/fruitvision_manifest.json` (SHA-256 `71651aee8816b01baaa1a752f1bcc6327ae26e98578ed95c2150276ad445cb3f`).

## Redistribution and claim boundary

`v2/data/external/` and `v2/cache/` are Git-ignored. No archive, source photograph, thumbnail, or embedding cache may be tracked. Experiment 014 evaluates visible fresh/rotten labels only and makes no formalin-detection, chemical-contamination, food-safety, toxicity, or edibility claim.
