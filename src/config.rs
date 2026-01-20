use anyhow::Result;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub server: ServerConfig,
    pub model: ModelConfig,
    pub inference: InferenceConfig,
    pub paths: PathsConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    pub port: u16,
    pub host: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfig {
    pub default_model: String,
    pub venv_path: String,
    #[serde(default)]
    pub allowed_models: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct InferenceConfig {
    pub max_tokens: u32,
    pub temperature: f32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct PathsConfig {
    pub pid_file: String,
    pub log_file: String,
}

impl Config {
    pub fn load() -> Result<Self> {
        let config_path = Self::get_config_path()?;
        let content = fs::read_to_string(&config_path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }

    fn get_config_path() -> Result<PathBuf> {
        let exe_path = std::env::current_exe()?;
        let exe_dir = exe_path.parent().unwrap();
        let config_path = exe_dir.join("config.toml");
        
        if !config_path.exists() {
            // Fallback to current directory
            let fallback = PathBuf::from("config.toml");
            if fallback.exists() {
                return Ok(fallback);
            }
        }
        
        Ok(config_path)
    }

    pub fn expand_path(path: &str) -> String {
        if path.starts_with("~/") {
            if let Some(home) = std::env::var("HOME").ok() {
                return path.replacen("~", &home, 1);
            }
        }
        path.to_string()
    }
}
