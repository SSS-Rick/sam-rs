[package]
name = "sam-rs"
version = "0.1.0"
edition = "2021"

[dependencies]
tch = "0.114"  # torch-rs/chroma - Rust bindings for PyTorch
tch-rs = { version = "0.114", features = ["cuda", "float16"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
base64 = "0.21"
image = "0.25"
imageproc = "0.34"
winit = "0.28"
web-sys = "0.3.143"
console_error_overlay = "0.2.1"
log = "0.4"
env_logger = "0.10"
```

Now let me create the main architecture module that maps the SAM2 PyTorch code to Rust:
<tool_call>
<function=Write>
<parameter=content>
// sam-rs: Rust port of Meta SAM2 (Segment Anything Model 2)
// Maps every component from the PyTorch implementation to Rust

/// Core SAM2 model architecture
pub struct SAM2 {
    /// Image encoder (backbone + neck)
    pub image_encoder: ImageEncoder,
    /// Memory attention module
    pub memory_attention: MemoryAttention,
    /// Memory encoder module
    pub memory_encoder: MemoryEncoder,
    /// SAM prompt encoder
    pub sam_prompt_encoder: PromptEncoder,
    /// SAM mask decoder
    pub sam_mask_decoder: MaskDecoder,
    /// Config parameters
    pub config: SAM2Config,
}

impl SAM2 {
    /// Build SAM2 from config and weights
    pub fn build(config_path: &str, weights_path: &str) -> Result<Self, String> {
        // Load config
        let config_str = std::fs::read_to_string(config_path)
            .map_err(|e| format!("Failed to read config: {}", e))?;
        let config: SAM2Config = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config: {}", e))?;

        // Initialize model components
        let image_encoder = ImageEncoder::build(&config.image_encoder)?;
        let memory_attention = MemoryAttention::build(&config.memory_attention)?;
        let memory_encoder = MemoryEncoder::build(&config.memory_encoder)?;
        let sam_prompt_encoder = PromptEncoder::build(&config.sam_prompt_encoder)?;
        let sam_mask_decoder = MaskDecoder::build(&config.sam_mask_decoder)?;

        // Load weights
        let weights = std::fs::read(weights_path)
            .map_err(|e| format!("Failed to read weights: {}", e))?;
        // TODO: Load weights into model

        Ok(SAM2 {
            image_encoder,
            memory_attention,
            memory_encoder,
            sam_prompt_encoder,
            sam_mask_decoder,
            config,
        })
    }

    /// Set an image for segmentation
    pub fn set_image(&mut self, image: &Tensor) -> Result<(), String> {
        // Preprocess image
        let input = self.preprocess_image(image)?;

        // Get image embeddings
        let backbone_out = self.image_encoder.forward(&input)?;

        // Prepare backbone features
        let (feats, vision_feats, vision_pos_embeds, feat_sizes) =
            self._prepare_backbone_features(backbone_out)?;

        // Store features for tracking
        // ... (track in predictor state)

        Ok(())
    }

    /// Predict masks given points
    pub fn predict_masks(
        &self,
        point_coords: &[f32],
        point_labels: &[i32],
    ) -> Result<(Tensor, Tensor, Tensor), String> {
        // Embed prompts
        let sparse_embeddings = self.sam_prompt_encoder.embed_points(
            point_coords,
            point_labels,
        )?;
        let dense_embeddings = self.sam_prompt_encoder.embed_no_mask()?;

        // Predict masks
        let high_res_features = self.get_high_res_features()?;
        let low_res_masks = self.sam_mask_decoder.predict_masks(
            &self.image_embed,
            &self.image_pe,
            &sparse_embeddings,
            &dense_embeddings,
            false, // repeat_image
            &high_res_features,
        )?;

        Ok(low_res_masks)
    }

    /// Track across video frames
    pub fn track_frame(
        &mut self,
        frame_idx: usize,
        is_init_cond_frame: bool,
        current_features: &[Tensor],
        point_inputs: Option<(&[f32], &[i32])>,
        mask_inputs: Option<&Tensor>,
        track_in_reverse: bool,
    ) -> Result<TrackingOutput, String> {
        // ... complex tracking logic
        todo!()
    }

    fn _prepare_backbone_features(
        &self,
        backbone_out: &Dict<String, Tensor>,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>, Vec<Tensor>, Vec<(usize, usize)>), String> {
        // Extract features from backbone output
        // ... maps to SAM2Base._prepare_backbone_features
        todo!()
    }

    fn preprocess_image(&self, image: &Tensor) -> Result<Tensor, String> {
        // Resize to config.image_size
        // Normalize with mean/std
        // Convert to CHW format
        todo!()
    }
}

/// SAM2 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SAM2Config {
    pub model: ModelConfig,
    pub image_encoder: ImageEncoderConfig,
    pub memory_attention: MemoryAttentionConfig,
    pub memory_encoder: MemoryEncoderConfig,
    pub sam_prompt_encoder: PromptEncoderConfig,
    pub sam_mask_decoder: MaskDecoderConfig,
    // ... other config params
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub _target_: String,
    pub num_maskmem: usize,
    pub image_size: usize,
    pub backbone_stride: usize,
    // ... all SAM2Base params
}

// Image encoder component
pub struct ImageEncoder {
    pub trunk: Hiera,
    pub neck: FpnNeck,
    pub scalp: usize,
}

impl ImageEncoder {
    fn forward(&self, input: &Tensor) -> Result<Dict<String, Tensor>, String> {
        // Forward through backbone
        let features = self.trunk.forward(input)?;
        // Forward through neck
        let neck_out = self.neck.forward(&features)?;
        Ok(neck_out)
    }
}

/// Hiera backbone (from hieradet.py)
pub struct Hiera {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub stages: Vec<usize>,
    pub window_spec: Vec<usize>,
    pub global_att_blocks: Vec<usize>,
    pub blocks: Vec<MultiScaleBlock>,
}

impl Hiera {
    fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>, String> {
        // ... forward pass through all blocks
        todo!()
    }
}

/// Multi-scale attention block
pub struct MultiScaleBlock {
    pub dim: usize,
    pub dim_out: usize,
    pub num_heads: usize,
    pub norm1: nn::LayerNorm,
    pub attn: MultiScaleAttention,
    pub norm2: nn::LayerNorm,
    pub mlp: MLP,
}

#[derive(Debug, Clone)]
pub struct MultiScaleAttention {
    pub dim: usize,
    pub dim_out: usize,
    pub num_heads: usize,
    pub q_pool: Option<nn::MaxPool2d>,
    pub qkv: nn::Linear,
    pub proj: nn::Linear,
}

impl MultiScaleAttention {
    fn forward(&self, x: &Tensor) -> Tensor {
        let (b, h, w, _) = x.shape();
        // qkv projection
        let qkv = self.qkv.forward(x);
        let (q, k, v) = qkv.split(3, Dim::Dim(-1));
        // ... attention computation
        todo!()
    }
}

/// Prompt encoder
pub struct PromptEncoder {
    pub embed_dim: usize,
    pub input_image_size: [usize; 2],
    pub image_embedding_size: [usize; 2],
    pub pe_layer: PositionEmbeddingRandom,
    pub point_embeddings: ModuleList<nn::Embedding>,
    pub not_a_point_embed: nn::Embedding,
    pub mask_input_size: [usize; 2],
    pub mask_downscaling: nn::Sequential,
    pub no_mask_embed: nn::Embedding,
}

impl PromptEncoder {
    fn embed_points(&self, points: &[f32], labels: &[i32]) -> Result<Tensor, String> {
        // Embed point prompts
        todo!()
    }

    fn embed_no_mask(&self) -> Tensor {
        // Return no mask embedding
        todo!()
    }

    fn get_dense_pe(&self) -> Tensor {
        // Get dense positional encoding
        todo!()
    }
}

/// SAM mask decoder
pub struct MaskDecoder {
    pub transformer_dim: usize,
    pub transformer: TwoWayTransformer,
    pub num_multimask_outputs: usize,
    pub iou_token: nn::Embedding,
    pub mask_tokens: nn::Embedding,
    pub output_upscaling: nn::Sequential,
    pub output_hypernetworks_mlps: ModuleList<MLP>,
    pub iou_prediction_head: MLP,
    pub dynamic_multimask_via_stability: bool,
    pub dynamic_multimask_stability_delta: f32,
    pub dynamic_multimask_stability_thresh: f32,
}

impl MaskDecoder {
    fn predict_masks(
        &self,
        image_embeddings: &Tensor,
        image_pe: &Tensor,
        sparse_prompt_embeddings: &Tensor,
        dense_prompt_embeddings: &Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Option<&[Tensor]>,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor), String> {
        // ... mask prediction logic
        todo!()
    }
}

/// Two-way transformer
pub struct TwoWayTransformer {
    pub depth: usize,
    pub embedding_dim: usize,
    pub num_heads: usize,
    pub mlp_dim: usize,
    pub layers: ModuleList<TwoWayAttentionBlock>,
}

impl TwoWayTransformer {
    fn forward(
        &self,
        image_embedding: &Tensor,
        image_pe: &Tensor,
        point_embedding: &Tensor,
    ) -> (Tensor, Tensor) {
        // ... transformer forward
        todo!()
    }
}

/// Transformer attention
pub struct Attention {
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub out_proj: nn::Linear,
    pub num_heads: usize,
    pub internal_dim: usize,
}

impl Attention {
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
        // ... scaled dot-product attention
        todo!()
    }
}

/// RoPE attention (Rotary Positional Encoding)
pub struct RoPEAttention {
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub out_proj: nn::Linear,
    pub num_heads: usize,
    pub internal_dim: usize,
    pub rope_theta: f32,
    pub rope_k_repeat: bool,
    pub feat_sizes: [usize; 2],
}

impl RoPEAttention {
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
        // ... attention with RoPE
        todo!()
    }
}

/// Memory attention module
pub struct MemoryAttention {
    pub d_model: usize,
    pub layers: ModuleList<MemoryAttentionLayer>,
    pub pos_enc_at_input: bool,
    pub batch_first: bool,
}

impl MemoryAttention {
    fn forward(
        &self,
        curr: &Tensor,
        memory: &Tensor,
        curr_pos: Option<&Tensor>,
        memory_pos: Option<&Tensor>,
        num_obj_ptr_tokens: usize,
    ) -> Tensor {
        // ... memory attention forward
        todo!()
    }
}

/// Memory attention layer
pub struct MemoryAttentionLayer {
    pub d_model: usize,
    pub dim_feedforward: usize,
    pub dropout: f32,
    pub pos_enc_at_attn: bool,
    pub pos_enc_at_cross_attn_keys: bool,
    pub pos_enc_at_cross_attn_queries: bool,
    pub self_attention: RoPEAttention,
    pub cross_attn_image: RoPEAttention,
    pub norm1: nn::LayerNorm,
    pub norm2: nn::LayerNorm,
    pub norm3: nn::LayerNorm,
    pub linear1: nn::Linear,
    pub linear2: nn::Linear,
    pub activation: fn(Tensor) -> Tensor,
}

impl MemoryAttentionLayer {
    fn forward(
        &self,
        tgt: &Tensor,
        memory: &Tensor,
        query_pos: Option<&Tensor>,
        pos: Option<&Tensor>,
        num_k_exclude_rope: usize,
    ) -> Tensor {
        // ... layer forward
        todo!()
    }
}

/// Memory encoder
pub struct MemoryEncoder {
    pub mask_downsampler: MaskDownSampler,
    pub pix_feat_proj: nn::Conv2d,
    pub fuser: Fuser,
    pub position_encoding: PositionEncodingSine,
    pub out_proj: nn::Conv2d,
    pub mem_dim: usize,
}

impl MemoryEncoder {
    fn forward(
        &self,
        pix_feat: &Tensor,
        masks: &Tensor,
        skip_mask_sigmoid: bool,
    ) -> Result<Dict<String, Tensor>, String> {
        // ... memory encoder forward
        todo!()
    }
}

/// Mask downsampler
pub struct MaskDownSampler {
    pub encoder: nn::Sequential,
}

impl MaskDownSampler {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.encoder.forward(x)
    }
}

/// CXBlock (from memory_encoder.py)
pub struct CXBlock {
    pub dwconv: nn::Conv2d,
    pub norm: LayerNorm2d,
    pub pwconv1: nn::Linear,
    pub act: nn::GELU,
    pub pwconv2: nn::Linear,
    pub gamma: nn::Parameter,
    pub drop_path: DropPath,
}

impl CXBlock {
    fn forward(&self, x: &Tensor) -> Tensor {
        // ... CXBlock forward
        todo!()
    }
}

/// Fuser module
pub struct Fuser {
    pub layers: ModuleList<CXBlock>,
}

impl Fuser {
    fn forward(&self, x: &Tensor) -> Tensor {
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }
}

/// Positional encoding sine
pub struct PositionEncodingSine {
    pub num_pos_feats: usize,
    pub temperature: f32,
    pub normalize: bool,
    pub scale: f32,
}

impl PositionEncodingSine {
    fn forward(&self, x: &Tensor) -> Tensor {
        // ... positional encoding
        todo!()
    }
}

/// DropPath
pub struct DropPath {
    pub drop_prob: f32,
}

impl DropPath {
    fn forward(&self, x: &Tensor) -> Tensor {
        if self.drop_prob == 0.0 || !x.requires_grad() {
            return x.clone();
        }
        let keep_prob = 1.0 - self.drop_prob;
        let shape = [x.shape()[0]] + vec![1; x.dim() - 1];
        let random_tensor = x.new_empty(shape).bernoulli_(keep_prob);
        if keep_prob > 0.0 {
            random_tensor.div_(keep_prob);
        }
        x * random_tensor
    }
}

/// MLP (Multi-Layer Perceptron)
pub struct MLP {
    pub layers: ModuleList<nn::Linear>,
    pub act: nn::GELU,
    pub sigmoid_output: bool,
}

impl MLP {
    fn forward(&self, x: &Tensor) -> Tensor {
        for (i, layer) in self.layers.iter().enumerate() {
            x = if i < self.layers.len() - 1 {
                self.act.forward(layer.forward(x))
            } else {
                layer.forward(x)
            };
        }
        if self.sigmoid_output {
            x.sigmoid()
        } else {
            x
        }
    }
}

/// LayerNorm2d
pub struct LayerNorm2d {
    pub weight: nn::Parameter,
    pub bias: nn::Parameter,
    pub eps: f32,
}

impl LayerNorm2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let u = x.mean_dim(1, KeepDim::Yes);
        let s = (x - &u).pow(2).mean_dim(1, KeepDim::Yes);
        let x = (x - &u) / (s + self.eps).sqrt();
        self.weight.broadcast_as(x.shape()) * x + self.bias.broadcast_as(x.shape())
    }
}

/// Tracking output
pub struct TrackingOutput {
    pub pred_masks: Tensor,
    pub pred_masks_high_res: Tensor,
    pub obj_ptr: Tensor,
    pub object_score_logits: Option<Tensor>,
}

/// Image features state
pub struct ImageFeatures {
    pub image_embed: Tensor,
    pub high_res_feats: Vec<Tensor>,
    pub image_pe: Tensor,
}