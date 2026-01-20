mod config;
mod service;
mod vlm_infer;

use anyhow::Result;
use clap::{Parser, Subcommand};
use config::Config;
use service::ServiceManager;
use vlm_infer::VlmInfer;

#[derive(Parser)]
#[command(name = "llm-server")]
#[command(about = "Local LLM Server Management CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the VLM server
    Start {
        /// Optional model to use (overrides config)
        /// Examples: mlx-community/Qwen3-VL-4B-Instruct-4bit (default)
        ///           mlx-community/Qwen3-VL-8B-Instruct-4bit
        #[arg(short, long)]
        model: Option<String>,
        
        /// Optional port to run on (overrides config)
        #[arg(short, long)]
        port: Option<u16>,
    },
    
    /// Stop the mlx-vlm server
    Stop,
    
    /// Check server status
    Status,
    
    /// Restart the server
    Restart,
    
    /// Run direct inference without starting server
    Infer {
        /// Path to image file
        #[arg(long)]
        image: String,
        
        /// Prompt text
        #[arg(long)]
        prompt: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = Config::load()?;

    match cli.command {
        Commands::Start { model, port } => {
            let manager = ServiceManager::new(config);
            manager.start(model.as_deref(), port)?;
        }
        Commands::Stop => {
            let manager = ServiceManager::new(config);
            manager.stop()?;
        }
        Commands::Status => {
            let manager = ServiceManager::new(config);
            manager.status()?;
        }
        Commands::Restart => {
            let manager = ServiceManager::new(config);
            manager.restart()?;
        }
        Commands::Infer { image, prompt } => {
            let infer = VlmInfer::new(config);
            let result = infer.infer(&image, &prompt)?;
            println!("{}", result);
        }
    }

    Ok(())
}
