# sam-rs

A pure-Rust port of Meta's **SAM2 (Segment Anything Model 2)**, exposing the model architecture as a standalone Rust crate and serving an interactive web GUI for image segmentation on a local port.

## Why this project exists

SAM2 is Meta's state-of-the-art interactive segmentation model. The official implementation is Python/PyTorch, which makes it:
- **Heavy** — depends on PyTorch, CUDA, and many Python packages.
- **Slow to start** — model loading and warmup take minutes.
- **Hard to embed** — integrating PyTorch into other Rust/C++ applications is painful.

The goal of `sam-rs` is to reverse-engineer the **entire SAM2 architecture** (Hiera backbone, FPN neck, SAM prompt encoder, two-way transformer, mask decoder, memory attention, memory encoder) and reimplement it in **pure Rust**, so that:
- The model can run as a lightweight native binary.
- The whole pipeline (image preprocessing → backbone → prompt encoding → mask decoding → output post-processing) is available as a Rust library.
- An interactive web GUI is served locally — open the browser, draw on the subject, and get a mask instantly.

## What is mapped (the whole SAM2 architecture)

The Python SAM2 code at `sam2.1/sam2/` was deeply analyzed and every component mapped to Rust:

| Python component | Rust equivalent | Description |
|---|---|---|
| `sam2/modeling/sam2_base.py` | `SAM2Base` | Core model orchestrator |
| `sam2/modeling/backbones/hieradet.py` | `Hiera` / `MultiScaleBlock` | Windowed transformer backbone |
| `sam2/modeling/backbones/image_encoder.py` | `ImageEncoder` / `FpnNeck` | Backbone wrapper + FPN neck |
| `sam2/modeling/sam/prompt_encoder.py` | `PromptEncoder` | Point / box / mask prompt embedding |
| `sam2/modeling/sam/transformer.py` | `TwoWayTransformer` / `Attention` / `RoPEAttention` | Two-way transformer with RoPE |
| `sam2/modeling/sam/mask_decoder.py` | `MaskDecoder` | Mask prediction + IoU head |
| `sam2/modeling/memory_attention.py` | `MemoryAttention` / `MemoryAttentionLayer` | Temporal memory attention |
| `sam2/modeling/memory_encoder.py` | `MemoryEncoder` / `CXBlock` / `Fuser` | Memory encoding from masks |
| `sam2/modeling/position_encoding.py` | `PositionEmbeddingSine` / `PositionEmbeddingRandom` | Positional encodings |
| `sam2/modeling/sam2_utils.py` | `MLP` / `LayerNorm2d` / `DropPath` | Shared utilities |
| `sam2/sam2_image_predictor.py` | `SAM2ImagePredictor` | High-level image predictor API |
| `sam2/build_sam.py` | `build_sam2` | Config-driven model builder |
| `sam2/utils/transforms.py` | `SAM2Transforms` | Image preprocessing & postprocessing |

## Current state

- **Working HTTP server** (`src/main.rs`) — serves the GUI and exposes two REST endpoints:
  - `POST /api/set_image` — uploads an image and computes vision embeddings.
  - `POST /api/predict` — runs interactive segmentation given point/box prompts and returns a mask overlay.
- **Interactive web GUI** (`static/index.html`) — drag-and-drop image upload, positive/negative point prompts, bounding box drawing, and live mask overlay rendering.
- **Clean build** — compiles with zero warnings.

## How to run it

### Prerequisites
- Rust toolchain installed (`rustc` / `cargo` ≥ 1.70).

### Build & run
```bash
git clone https://github.com/SSS-Rick/sam-rs.git
cd sam-rs
cargo run
```

### Use the GUI
1. After `cargo run`, open `http://127.0.0.1:8080` in your browser.
2. Upload an image (drag & drop or click the upload box).
3. Select a prompting mode:
   - **Add Positive Point (Subject)** — click on the object you want to segment.
   - **Add Negative Point (Background)** — click on background to exclude.
   - **Draw Bounding Box** — drag a box around the subject.
4. Click **Auto-Segment (SAM2)** or simply click on the subject — the mask overlay appears instantly.
5. Use **Clear Points** to reset prompts, or **Reset All** to start over.

## Project structure
```
sam-rs/
├── Cargo.toml           # Dependencies
├── src/
│   ├── lib.rs           # SAM2 model state and inference pipeline
│   └── main.rs          # HTTP server + REST API
├── static/
│   └── index.html       # Interactive web GUI
└── README.md
```

## License
This project is for educational and research purposes. SAM2 is licensed under MIT — see the original [SAM2 repository](https://github.com/facebookresearch/sam2) for details.