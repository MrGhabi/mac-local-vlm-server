use anyhow::{Context, Result};
use std::fs;
use std::net::TcpStream;
use std::process::Command;
use std::time::Duration;
use sysinfo::{System, Pid, ProcessRefreshKind};

use crate::config::Config;

pub struct ServiceManager {
    config: Config,
}

impl ServiceManager {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    pub fn start(&self, model: Option<&str>, port_override: Option<u16>) -> Result<()> {
        let port = port_override.unwrap_or(self.config.server.port);

        if self.is_running(port)? {
            println!("Service is already running on port {}", port);
            return Ok(());
        }

        let venv_path = Config::expand_path(&self.config.model.venv_path);
        let python_bin = format!("{}/bin/python", venv_path);
        
        // Use our custom local_vlm_server.py instead of mlx_vlm.server
        let server_script = std::env::current_exe()?
            .parent()
            .context("Failed to get executable directory")?
            .join("local_vlm_server.py");
        
        if !server_script.exists() {
            anyhow::bail!(
                "local_vlm_server.py not found at {:?}. Please install llm-server properly.",
                server_script
            );
        }

        // Determine which model to use
        let model_to_use = model.unwrap_or(&self.config.model.default_model);
        
        // Serialize allowed models to semicolon-separated string
        let allowed_models_str = self.config.model.allowed_models.join(";");

        println!("Starting local VLM server on port {}...", port);
        println!("Model: {}", model_to_use);
        if !allowed_models_str.is_empty() {
             println!("Allowed Models: {:?}", self.config.model.allowed_models);
        }
        
        // Check if log file path needs to be unique per port?
        // Ideally we should append port to log filename if running multiple ports, 
        // to avoid conflict.
        let log_file_expanded = Config::expand_path(&self.config.paths.log_file);
        // If non-default port, append port to log name e.g. .llm-server-58081.log
        let final_log_file = if port != self.config.server.port {
            // naive replacement extension
            if log_file_expanded.ends_with(".log") {
                log_file_expanded.replace(".log", &format!("-{}.log", port))
            } else {
                format!("{}-{}", log_file_expanded, port)
            }
        } else {
            log_file_expanded
        };

        // Same for PID file
        let pid_file_expanded = Config::expand_path(&self.config.paths.pid_file);
        let final_pid_file = if port != self.config.server.port {
             if pid_file_expanded.ends_with(".pid") {
                pid_file_expanded.replace(".pid", &format!("-{}.pid", port))
            } else {
                format!("{}-{}", pid_file_expanded, port)
            }
        } else {
            pid_file_expanded
        };

        // Set HF_HOME to project directory so models are loaded from ./data
        let exe_path = std::env::current_exe()?;
        let project_dir = exe_path
            .parent()
            .and_then(|p| p.parent())
            .context("Failed to get project directory")?;
        let hf_cache = project_dir.join("data");

        let child = Command::new(&python_bin)
            .arg(&server_script)
            .env("PYTHONUNBUFFERED", "1")
            .env("VLM_PORT", port.to_string())
            .env("VLM_MODEL", model_to_use)
            .env("ALLOWED_MODELS", allowed_models_str)
            .env("HF_HOME", hf_cache)
            .stdout(std::process::Stdio::null())
            .stderr(std::fs::File::create(&final_log_file)?)
            .spawn()
            .context("Failed to start server")?;

        let pid = child.id();
        fs::write(&final_pid_file, pid.to_string())?;
        
        println!("✓ Service started (PID: {})", pid);
        println!("  Port: {}", port);
        println!("  Logs: {}", final_log_file);
        println!("  PID File: {}", final_pid_file); // Helpful for debugging
        println!("\nUse 'llm-server status' to check when service is ready.");

        Ok(())
    }

    pub fn stop(&self) -> Result<()> {
        let pid_file_path = Config::expand_path(&self.config.paths.pid_file);
        
        if !std::path::Path::new(&pid_file_path).exists() {
            println!("Service is not running (no PID file)");
            return Ok(());
        }

        let pid_str = fs::read_to_string(&pid_file_path)
            .context("Failed to read PID file")?;
        let pid: u32 = pid_str.trim().parse()
            .context("Invalid PID in file")?;

        println!("Stopping service (PID: {})...", pid);

        // Try to kill the process
        #[cfg(unix)]
        {
            
            Command::new("kill")
                .arg(pid.to_string())
                .status()
                .context("Failed to kill process")?;
        }

        #[cfg(not(unix))]
        {
            // Windows fallback (not tested)
            Command::new("taskkill")
                .args(&["/PID", &pid.to_string(), "/F"])
                .status()
                .context("Failed to kill process")?;
        }

        // Remove PID file
        fs::remove_file(&pid_file_path)
            .context("Failed to remove PID file")?;

        println!("✓ Service stopped");
        Ok(())
    }

    pub fn status(&self) -> Result<()> {
        let pid_file_path = Config::expand_path(&self.config.paths.pid_file);
        
        if !std::path::Path::new(&pid_file_path).exists() {
            println!("Service: NOT RUNNING (no PID file)");
            return Ok(());
        }

        let pid_str = fs::read_to_string(&pid_file_path)?;
        let pid: u32 = pid_str.trim().parse()
            .context("Invalid PID in file")?;

        // Check if process exists
        let mut sys = System::new();
        sys.refresh_processes_specifics(ProcessRefreshKind::new());
        
        if let Some(process) = sys.process(Pid::from_u32(pid)) {
            println!("Service: RUNNING");
            println!("  PID: {}", pid);
            println!("  Port: {}", self.config.server.port);
            println!("  Memory: {:.1} MB", process.memory() as f64 / 1024.0 / 1024.0);
            
            if self.is_running(self.config.server.port)? {
                println!("  Status: ✓ Responding");
            } else {
                println!("  Status: ⚠ Not responding on port {}", self.config.server.port);
            }
        } else {
            println!("Service: STALE (PID file exists but process not found)");
            println!("  Run 'llm-server stop' to clean up");
        }

        Ok(())
    }

    pub fn restart(&self) -> Result<()> {
        println!("Restarting service...");
        let _ = self.stop();
        std::thread::sleep(Duration::from_millis(500));
        self.start(None, None)
    }

    fn check_port(&self, port: u16) -> Result<bool> {
        let addr = format!("{}:{}", self.config.server.host, port);
        match TcpStream::connect_timeout(
            &addr.parse().context("Invalid address")?,
            Duration::from_millis(500)
        ) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
    
    fn is_running(&self, port: u16) -> Result<bool> {
        self.check_port(port)
    }
}
