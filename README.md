# Towards Visual Query Segmentation in the Wild

Official code and benchmark release for the NeurIPS 2026 (Datasets & Benchmarks) paper
**_Towards Visual Query Segmentation in the Wild_**.

Bing Fan\*, Minghao Li\*, Hanzhi Zhang\*, Shaohua Dong, Naga Prudhvi Mareedu,
Xiaoqiong Liu, Weishi Shi, Yunhe Feng, Yan Huang, Heng Fan†
<br>Department of Computer Science and Engineering, University of North Texas
<br><sub>\*Equal contribution &nbsp;&nbsp;†Corresponding author</sub>

📄 **Paper:** [arXiv:2603.08898](https://arxiv.org/pdf/2603.08898)
&nbsp;&nbsp;|&nbsp;&nbsp;📦 **Dataset:** [`bf0820/VQS-4K` on Hugging Face](https://huggingface.co/datasets/bf0820/VQS-4K)

---

## What is visual query segmentation?

You are given an untrimmed video and a **visual query**: one image plus a mask of the target,
taken from **outside** that video. The task is to produce a pixel mask for the object in
**every frame where it appears**, and nothing where it does not.

- **vs. VQL** (visual query localization) — VQL returns a single bounding box on the target's
  *last* appearance. VQS returns pixel masks for *all* appearances.
- **vs. VOS** (video object segmentation) — VOS takes its reference mask from the video's own
  first frame. Here the query is **external**, so pose, scale, lighting and background can
  differ sharply from anything in the video.

**VQS-4K** is the benchmark: 4,111 video–query pairs, 1,304,473 search-video frames at 6 fps,
222,173 annotated masks, 222 fine-grained categories. **VQ-SAM** is the baseline in this
repository — it extends SAM 2 with target-specific and background-distractor memory, evolved
through a multi-stage framework with an adaptive memory generation (AMG) module.

## Results on VQS-4K test

| Method | mSTIoU | mSTIoU₅₀ | mSTIoU₇₅ | mTIoU | mTIoU₅₀ | mTIoU₇₅ | Rec | Succ | FPS |
|---|---|---|---|---|---|---|---|---|---|
| Cutie-b <sub>CVPR'24</sub> | 18.5 | 16.5 | 8.2 | 21.0 | 16.9 | 9.2 | 38.4 | 32.3 | 36 |
| OASIS <sub>ICCV'25</sub> | 15.8 | 13.6 | 6.7 | 21.3 | 17.4 | 9.5 | 34.3 | 29.4 | 48 |
| LOMM <sub>ICCV'25</sub> | 12.5 | 10.2 | 5.0 | 12.7 | 7.9 | 4.3 | 26.1 | 25.8 | 10 |
| SAM 2 <sub>ICLR'25</sub> | 12.9 | 10.6 | 5.2 | 13.0 | 8.3 | 4.6 | 26.7 | 26.3 | 22 |
| SAM2.1++ <sub>CVPR'25</sub> | 12.5 | 10.2 | 5.1 | 14.6 | 10.0 | 5.5 | 28.8 | 25.8 | 16 |
| SAM2Long <sub>ICCV'25</sub> | 18.6 | 16.5 | 8.2 | 24.4 | 20.6 | 11.2 | 38.6 | 32.5 | 19 |
| SAM 3 <sub>ICLR'26</sub> | 13.3 | 10.9 | 5.4 | 14.1 | 9.5 | 5.2 | 27.7 | 26.6 | 15 |
| VQLoC <sub>NeurIPS'23</sub> | 10.1 | 12.5 | 6.3 | 21.5 | 17.5 | 8.4 | 31.7 | 23.2 | 36 |
| PRVQL <sub>ICCV'25</sub> | 13.6 | 14.4 | 7.1 | 22.1 | 18.4 | 8.7 | 32.6 | 26.9 | 30 |
| REN <sub>NeurIPS'25</sub> | 16.8 | 17.7 | 8.9 | 23.6 | 19.8 | 9.5 | 34.8 | 30.5 | 12 |
| **VQ-SAM (ours)** | **26.0** | **23.8** | **12.7** | **29.6** | **26.8** | **14.4** | **43.6** | **42.1** | 29 |

## Repository layout

```
sam2/                SAM 2 model code. Our additions live in
                       modeling/spatio_temporal_transformer.py   (STTB)
                       modeling/memory_attention.py              (AMG / adaptive gating)
                     plus the training configs under configs/sam2.1_training/
training/            training loop, VOS dataset/sampler, loss functions
tools/               inference, visualisation and benchmarking entry points
eval/                official metric implementation (tAP, stAP, Succ, Rec, t-IoU, st-IoU)
splits/              official split files, both protocols, one sequence id per line
configs_repro/       fully resolved Hydra configs from the runs reported in the paper
docs/                method design notes, train/test guides, env setup
```

## Install

Verified on 4× RTX A6000 (49 GB), driver 565.57.01, CUDA 12.7.

```bash
conda create -y -n vqsam python=3.10.13
conda activate vqsam

pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -r requirements.txt
pip install -e .
```

> **Run commands from the repository root.** `tools/` and `training/` scripts put their own
> directory on `sys.path`, not the repo root, so if you have another `sam2` installed the
> import can silently resolve to it. If `python -c "import sam2; print(sam2.__file__)"` does
> not point inside this repository, prefix your commands with
> `PYTHONPATH=$(pwd)`.

## Get the data

The dataset is packed **one tar per sequence** on Hugging Face, so you can take a single video
instead of all 76 GB.

```bash
# whole test split
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("bf0820/VQS-4K", repo_type="dataset", local_dir="data/VQS-4K")
PY
```

Then unpack the archives so that the tree looks like this — every tar extracts into its own
`<seq_id>/` folder, so unpacking them all in place rebuilds the original layout:

```
data/VQS-4K/
├── images/<seq_id>/00000.jpg   # 00000 is the QUERY frame, not part of the video
│                   00001.jpg   # the search video starts here
└── masks/<seq_id>/00000.png    # query mask (model input, never scored)
                   00001.png    # GT mask, only where the target is visible
```

Three things to know:

1. **Frame `00000` is the query, not part of the search video.** It is excluded from all metrics.
2. **Masks are sparse on purpose.** A mask file exists only for frames where the object is
   present. *No mask file means the object is absent and the correct prediction is an empty
   mask* — this is the point of the untrimmed setting, so do not treat missing masks as
   unlabelled.
3. **Sequence ids are `<supercategory>-<category>-set_<n>`.** `category` alone is not unique
   (`mouse` exists under both `electronics` and `land_animal`), so key on the pair.

See the [dataset card](https://huggingface.co/datasets/bf0820/VQS-4K) for `metadata.csv`,
the per-category breakdown and a ready-made PyTorch `Dataset`.

Finally, fetch the SAM 2.1 initialisation checkpoint into `checkpoints/`:

```bash
cd checkpoints && ./download_ckpts.sh && cd ..
```

## Train

The paper's main model, on 4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python training/train.py \
  -c configs/sam2.1_training/sam2.1_hiera_b+_DAM_AdaptiveGate_Jan28.yaml \
  --use-cluster 0 --num-gpus 4
```

Paths in that config are relative to the repository root
(`./data/VQS-4K/{images,masks}`, `./splits/train_t-IoU.txt`,
`./checkpoints/sam2.1_hiera_base_plus.pt`); edit the `dataset:` block if your data lives
elsewhere. Checkpoints, logs and TensorBoard events go to `sam2_logs/<experiment>/`.

Note the Hydra `-c` path is `configs/...`, **not** `sam2/configs/...` — it resolves inside the
`sam2` package.

### Ablations

| Config | Ablation |
|---|---|
| `sam2.1_hiera_b+_DAM_AdaptiveGate_Jan28.yaml` | **main model** |
| `..._DAM_AdaptiveGate_DRM2_Jan31.yaml` / `..._DRM3_Jan31.yaml` | distractor memory slots = 2 / 3 |
| `..._DAM_AdaptiveGate_RAM0_Feb8.yaml` / `..._RAM1_Feb8.yaml` / `..._RAM4_Feb6.yaml` | recent-memory slots = 0 / 1 / 4 |
| `..._DAM_Gate_QualityWeight_Jan15.yaml` | quality-weighted RAM, 8 frames/segment |
| `..._DAM_Gate_QualityWeight_Jan15_nf12.yaml` / `..._nf16.yaml` | 12 / 16 frames per segment |

For `nf16`, `spatio_temporal_transformer.max_temporal_pos_frames` must be
`>= scratch.num_frames - 1`, and `memory_attention.use_activation_checkpointing: true` is
recommended to fit in VRAM. The inference segment length must match the training clip length.

## Inference

```bash
python tools/test_video_segmentation.py \
  --sam2_cfg sam2/configs/sam2.1_training/sam2.1_hiera_b+_DAM_AdaptiveGate_Jan28.yaml \
  --sam2_checkpoint sam2_logs/two_stage_mem_dam_adaptive_gating_Jan28/checkpoints/checkpoint.pt \
  --base_video_dir ./data/VQS-4K/images \
  --input_mask_dir ./data/VQS-4K/masks \
  --video_list_file ./splits/test_t-IoU.txt \
  --output_mask_dir ./outputs/predictions \
  --num_frames_per_segment 8 \
  --segments_per_batch 2
```

Multi-GPU: `bash tools/test_video_segmentation_multi_gpu.sh --num-gpus 4 ...` (same flags in
`--kebab-case`). Predictions are written as one binary PNG per frame under
`outputs/predictions/<seq_id>/`, using the same frame indices as the GT masks.

## Evaluate

```bash
bash eval/scripts/run_metrics.sh \
  --gt-folder ./data/VQS-4K/masks \
  --pred-folder ./outputs/predictions \
  --video-list ./splits/test_t-IoU.txt \
  --output results.json
```

Always report **which split file and GT folder** you used — the two split protocols
(`*_t-IoU.txt` and `*_t-IoU_by_class.txt`) are alternatives covering all 4,111 sequences, not
nested subsets, and they give different numbers.

Metric definitions, and the mapping from the script's output keys to the `mSTIoU` / `mTIoU` /
`Rec` / `Succ` names used in the paper, are in [`eval/METRICS.md`](eval/METRICS.md). Run on the
VQ-SAM predictions, this code reproduces every column of the main table above.

## Licenses

- **Code** — Apache 2.0, inherited from [SAM 2](https://github.com/facebookresearch/sam2).
  See [`LICENSE`](LICENSE).
- **Dataset** — CC BY-NC 4.0, research use only. See [`DATASET_LICENSE`](DATASET_LICENSE).

All source videos were collected from YouTube under Creative Commons licenses. If the license
of a source video changes, please contact us and we will take appropriate action.

## Citation

```bibtex
@inproceedings{fan2026vqs,
  title     = {Towards Visual Query Segmentation in the Wild},
  author    = {Fan, Bing and Li, Minghao and Zhang, Hanzhi and Dong, Shaohua and
               Mareedu, Naga Prudhvi and Liu, Xiaoqiong and Shi, Weishi and
               Feng, Yunhe and Huang, Yan and Fan, Heng},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)
               Datasets and Benchmarks Track},
  year      = {2026},
  eprint    = {2603.08898},
  archivePrefix = {arXiv}
}
```

## Acknowledgements

Built on [SAM 2](https://github.com/facebookresearch/sam2). The evaluation metrics follow the
Ego4D VQ2D protocol. We thank the annotators and volunteers who inspected and refined VQS-4K.
