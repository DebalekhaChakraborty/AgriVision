# External Dataset Provenance — Experiment 013

## Registered release

| Field | Frozen value |
|---|---|
| Dataset | Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality |
| Authors | Nusrat Sultana; Musfika Jahan; Mohammad Shorif Uddin |
| Repository | Mendeley Data |
| DOI | [10.17632/bdd69gyhv8.1](https://data.mendeley.com/datasets/bdd69gyhv8/1) |
| Version | 1 |
| Publication date | 8 April 2022 |
| Licence | CC BY 4.0 |
| Source institution | Jahangirnagar University, Bangladesh |
| Associated article | [An extensive dataset for successful recognition of fresh and rotten fruits](https://pmc.ncbi.nlm.nih.gov/articles/PMC9469664/) |
| Download/integrity date | 2 September 2026 |

The associated Data in Brief article describes photographs collected from fruit shops and fields from 16–31 March 2022 with an agricultural-domain specialist, using a Nikon D5600. These statements describe the published collection and do not establish that its visible fresh/rotten annotation protocol is identical to the source dataset's independently produced protocol.

## Original-image identification

The official Mendeley file API exposes two distinct files at the registered version. This packaging makes original/augmented separation reliable:

| Official file | File ID | Bytes | SHA-256 | Experiment 013 disposition |
|---|---|---:|---|---|
| `Original Image.zip` | `de93ba06-6a58-45e3-913d-837b2ae52acb` | 2,794,170,228 | `f89d67d4c4b24810bcd8406db877db6b1ac9834cea0b0cb56d65acebe59e66ff` | Downloaded, hash-verified, CRC-tested, then six folders extracted |
| `Augmented Image.zip` | `ccd1f142-03b2-473a-8c78-78920e63b8bd` | 803,272,180 | `5c748b5904932cd8cf631a0b776253d6217d577067d9477c267014d03ac16e10` | Never downloaded or extracted |

The published dataset has 3,200 originals: 200 in each of 16 class folders. The article reports 12,335 augmentation-derived images separately. Experiment 013 selected files only from `Original Image.zip`; it did not attempt to infer originality from filenames or pixels.

The verified archive was removed after extraction to recover local disk space. It remains reproducibly obtainable from the DOI and its exact identity is frozen above. No archive or photograph is tracked by Git.

## Folder structure and class mapping

Local ignored root: `v2/data/external/sultana_2022/Original Image/`.

| Official folder | Canonical class | Eligible originals |
|---|---|---:|
| `FreshApple` | `fresh_apple` | 200 |
| `RottenApple` | `rotten_apple` | 200 |
| `FreshBanana` | `fresh_banana` | 200 |
| `RottenBanana` | `rotten_banana` | 200 |
| `FreshOrange` | `fresh_orange` | 200 |
| `RottenOrange` | `rotten_orange` | 200 |
| **Total** | | **1,200** |

Non-overlapping grape, guava, jujube, pomegranate, and strawberry fresh/rotten folders were excluded. The entire official augmented archive was excluded. Every eligible selected original is evaluation-only: no target-domain train, validation, calibration, adaptation, prototype, or prompt-selection subset exists.

## Local-data boundary

`.gitignore` excludes `v2/data/external/` and `v2/cache/`. The repository retains provenance, hashes, image identifiers, numeric statistics, predictions, and aggregate plots only. It contains no external image thumbnails or redistributed source photographs.

