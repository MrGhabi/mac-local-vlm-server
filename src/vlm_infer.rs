use anyhow::{Context, Result};
use std::process::Command;

use crate::config::Config;

pub struct VlmInfer {
    config: Config,
}

impl VlmInfer {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    pub fn infer(&self, image_path: &str, prompt: &str) -> Result<String> {
        let venv_path = Config::expand_path(&self.config.model.venv_path);
        let python_bin = format!("{}/bin/python", venv_path);
        
        // Get path to vlm_infer.py
        // Priority: 1. Same dir as executable, 2. Project root, 3. Current dir
        let mut script_path = None;
        
        // Try same directory as executable
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                let path = exe_dir.join("vlm_infer.py");
                if path.exists() {
                    script_path = Some(path);
                }
            }
        }
        
        // Try current directory
        if script_path.is_none() {
            let path = std::path::PathBuf::from("vlm_infer.py");
            if path.exists() {
                script_path = Some(path);
            }
        }
        
        // Try local-llm-server directory (for development)
        if script_path.is_none() {
            let path = std::path::PathBuf::from("tools/local-llm-server/vlm_infer.py");
            if path.exists() {
                script_path = Some(path);
            }
        }
        
        let script_path = script_path
            .context("vlm_infer.py not found. Make sure it's in the same directory as the executable.")?;

        eprintln!("Running inference...");
        eprintln!("  Model: {}", self.config.model.default_model);
        eprintln!("  Image: {}", image_path);
        
        let output = Command::new(&python_bin)
            .arg(script_path)
            .arg("--image")
            .arg(image_path)
            .arg("--prompt")
            .arg(prompt)
            .arg("--model")
            .arg(&self.config.model.default_model)
            .arg("--max-tokens")
            .arg(self.config.inference.max_tokens.to_string())
            .arg("--temperature")
            .arg(self.config.inference.temperature.to_string())
            .output()
            .context("Failed to execute vlm_infer.py")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Inference failed: {}", stderr);
        }

        let result = String::from_utf8(output.stdout)
            .context("Invalid UTF-8 output")?;
        
        Ok(result)
    }
}
