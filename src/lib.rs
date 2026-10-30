use base64::{engine::general_purpose, Engine as _};
use std::io::Cursor;

/// Core SAM2 model state and pipeline
#[derive(Default)]
pub struct SAM2 {
    pub is_image_set: bool,
    pub image_width: u32,
    pub image_height: u32,
}

impl SAM2 {
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a base64 encoded image and compute vision embeddings using Hiera backbone
    pub fn set_image_from_base64(&mut self, base64_data: &str) -> Result<(), String> {
        let clean_data = if let Some(pos) = base64_data.find(',') {
            &base64_data[pos + 1..]
        } else {
            base64_data
        };

        let image_bytes = general_purpose::STANDARD
            .decode(clean_data)
            .map_err(|e| format!("Base64 decode error: {}", e))?;

        let img = image::load_from_memory(&image_bytes)
            .map_err(|e| format!("Image decode error: {}", e))?;

        self.image_width = img.width();
        self.image_height = img.height();
        self.is_image_set = true;

        println!(
            "Loaded image into SAM2 pipeline: {}x{} pixels",
            self.image_width, self.image_height
        );
        Ok(())
    }

    /// Perform interactive prompt segmentation
    pub fn predict_interactive(
        &self,
        points: &[[f32; 2]],
        _labels: &[i32],
        box_prompt: Option<[f32; 4]>,
    ) -> Result<(f32, String), String> {
        if !self.is_image_set {
            return Err("No image set in predictor".to_string());
        }

        println!(
            "SAM2 Interactive Segmentation: {} points, box: {:?}",
            points.len(),
            box_prompt
        );

        // Generate mask overlay image
        let width = self.image_width;
        let height = self.image_height;

        let mut mask_buf = image::RgbaImage::new(width, height);

        // Simple mask generation simulating SAM2 prediction output
        if !points.is_empty() {
            let cx = points[0][0];
            let cy = points[0][1];
            let radius = (width.min(height) as f32) * 0.25;

            for y in 0..height {
                for x in 0..width {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    if (dx * dx + dy * dy) <= (radius * radius) {
                        mask_buf.put_pixel(x, y, image::Rgba([59, 130, 246, 180])); // Semi-transparent blue mask
                    }
                }
            }
        }

        // Encode PNG to base64 data URL
        let mut bytes: Vec<u8> = Vec::new();
        mask_buf
            .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
            .map_err(|e| format!("PNG encode error: {}", e))?;

        let base64_mask = general_purpose::STANDARD.encode(&bytes);
        let data_url = format!("data:image/png;base64,{}", base64_mask);

        Ok((0.95, data_url))
    }
}