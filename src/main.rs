use sam_rs::SAM2;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tiny_http::{Header, Method, Response, Server};

#[derive(Deserialize)]
struct SetImageRequest {
    image_data: String,
}

#[derive(Serialize)]
struct SetImageResponse {
    status: String,
    message: String,
}

#[derive(Deserialize)]
struct PredictRequest {
    points: Vec<[f32; 2]>,
    labels: Vec<i32>,
    bbox: Option<[f32; 4]>,
    multimask_output: bool,
}

#[derive(Serialize)]
struct PredictResponse {
    status: String,
    message: String,
    iou_score: f32,
    mask_overlay_data: String,
}

fn main() {
    env_logger::init();
    println!("Initializing SAM2 Rust Model Engine...");

    let model = Arc::new(Mutex::new(SAM2::default()));

    let address = "127.0.0.1:8080";
    let server = Server::http(address).expect("Failed to start HTTP server");
    println!("====================================================");
    println!(" SAM2 Rust Interactive GUI running!");
    println!(" Open your browser at: http://{}", address);
    println!("====================================================");

    for request in server.incoming_requests() {
        let url = request.url().to_string();
        let method = request.method().clone();

        match (method, url.as_str()) {
            (Method::Get, "/") | (Method::Get, "/index.html") => {
                let html = include_str!("../static/index.html");
                let header = Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap();
                let response = Response::from_string(html).with_header(header);
                let _ = request.respond(response);
            }
            (Method::Post, "/api/set_image") => {
                let mut body = String::new();
                let mut req = request;
                if req.as_reader().read_to_string(&mut body).is_ok() {
                    if let Ok(data) = serde_json::from_str::<SetImageRequest>(&body) {
                        println!("Processing new image in SAM2 Rust pipeline...");
                        let mut model_lock = model.lock().unwrap();
                        let res = match model_lock.set_image_from_base64(&data.image_data) {
                            Ok(_) => SetImageResponse {
                                status: "ok".to_string(),
                                message: "Image embeddings computed successfully".to_string(),
                            },
                            Err(e) => SetImageResponse {
                                status: "error".to_string(),
                                message: e,
                            },
                        };
                        let json = serde_json::to_string(&res).unwrap();
                        let header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
                        let response = Response::from_string(json).with_header(header);
                        let _ = req.respond(response);
                    }
                }
            }
            (Method::Post, "/api/predict") => {
                let mut body = String::new();
                let mut req = request;
                if req.as_reader().read_to_string(&mut body).is_ok() {
                    if let Ok(data) = serde_json::from_str::<PredictRequest>(&body) {
                        println!("Running SAM2 inference for prompt...");
                        let model_lock = model.lock().unwrap();
                        let res = match model_lock.predict_interactive(&data.points, &data.labels, data.bbox) {
                            Ok((score, mask_base64)) => PredictResponse {
                                status: "ok".to_string(),
                                message: "Success".to_string(),
                                iou_score: score,
                                mask_overlay_data: mask_base64,
                            },
                            Err(e) => PredictResponse {
                                status: "error".to_string(),
                                message: e,
                                iou_score: 0.0,
                                mask_overlay_data: "".to_string(),
                            },
                        };
                        let json = serde_json::to_string(&res).unwrap();
                        let header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
                        let response = Response::from_string(json).with_header(header);
                        let _ = req.respond(response);
                    }
                }
            }
            _ => {
                let response = Response::from_string("Not Found").with_status_code(404);
                let _ = request.respond(response);
            }
        }
    }
}