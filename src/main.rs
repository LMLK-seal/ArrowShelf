use anyhow::Result;
use dashmap::DashMap;
use log::{error, info};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::signal;
use uuid::Uuid;

// The store maps a simple key to a file path.
type Db = Arc<DashMap<String, PathBuf>>;

// --- V2.0 Protocol with ListKeys ---
#[derive(Serialize, Deserialize, Debug)] 
struct Command { 
    action: Action, 
    key: Option<String> 
}

#[derive(Serialize, Deserialize, Debug, PartialEq)] 
enum Action { 
    RequestPath, 
    GetPath, 
    Delete, 
    ListKeys, // <-- Re-added
    Ping, 
    Shutdown 
}

#[derive(Serialize, Deserialize, Debug)] 
struct Response { 
    status: Status, 
    message: Option<String>, 
    path: Option<PathBuf>, 
    keys: Option<Vec<String>> // <-- Re-added
}

#[derive(Serialize, Deserialize, Debug)] 
enum Status { Ok, Error }

const BIND_ADDRESS: &str = "127.0.0.1:56789";

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let listener = TcpListener::bind(BIND_ADDRESS).await?;
    info!("ArrowShelf V2 (Shared Memory) daemon listening on tcp://{}", BIND_ADDRESS);
    info!("Press Ctrl+C to shut down.");

    let db = Arc::new(DashMap::new());

    loop {
        tokio::select! {
            Ok((stream, addr)) = listener.accept() => {
                info!("Accepted connection from: {}", addr);
                let db_clone = db.clone();
                tokio::spawn(async move {
                    let _ = handle_connection(stream, db_clone).await;
                });
            }
            _ = signal::ctrl_c() => {
                info!("Ctrl+C received, cleaning up shared memory files and shutting down.");
                for item in db.iter() {
                    let path = item.value();
                    info!("Cleaning up: {:?}", path);
                    let _ = fs::remove_file(path);
                }
                break;
            }
        }
    }
    Ok(())
}

async fn handle_connection(stream: TcpStream, db: Db) -> Result<()> {
    let mut reader = BufReader::new(stream);
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line).await {
            Ok(0) | Err(_) => break, 
            _ => {}
        }

        let cmd: Command = match serde_json::from_str(line.trim()) { Ok(c) => c, _ => continue };

        match cmd.action {
            Action::RequestPath => {
                let key = Uuid::new_v4().to_string();
                let filename = format!("arrowshelf_{}", key);
                let path = std::env::temp_dir().join(filename);

                db.insert(key.clone(), path.clone());
                info!("Reserved path {:?} for key {}", path, key);
                let resp = Response { status: Status::Ok, message: Some(key), path: Some(path), keys: None };
                let resp_json = serde_json::to_string(&resp)? + "\n";
                reader.get_mut().write_all(resp_json.as_bytes()).await?;
            }
            Action::GetPath => {
                let key = cmd.key.ok_or_else(|| anyhow::anyhow!("'GetPath' requires 'key'"))?;
                if let Some(entry) = db.get(&key) {
                    let path = entry.value().clone();
                    let resp = Response { status: Status::Ok, message: None, path: Some(path), keys: None };
                    let resp_json = serde_json::to_string(&resp)? + "\n";
                    reader.get_mut().write_all(resp_json.as_bytes()).await?;
                } else {
                    let resp = Response { status: Status::Error, message: Some("Key not found".to_string()), path: None, keys: None };
                    let resp_json = serde_json::to_string(&resp)? + "\n";
                    reader.get_mut().write_all(resp_json.as_bytes()).await?;
                }
            }
            Action::Delete => {
                let key = cmd.key.ok_or_else(|| anyhow::anyhow!("'Delete' requires 'key'"))?;
                if let Some((_, path)) = db.remove(&key) {
                    info!("Deleting file for key {}: {:?}", key, path);
                    if let Err(e) = fs::remove_file(&path) {
                        error!("Failed to delete file {:?}: {}", path, e);
                    }
                }
                let resp = Response { status: Status::Ok, message: Some("Deleted".to_string()), path: None, keys: None };
                let resp_json = serde_json::to_string(&resp)? + "\n";
                reader.get_mut().write_all(resp_json.as_bytes()).await?;
            }
            // --- Re-added ListKeys Logic ---
            Action::ListKeys => {
                let keys: Vec<String> = db.iter().map(|entry| entry.key().clone()).collect();
                let resp = Response { status: Status::Ok, message: None, path: None, keys: Some(keys) };
                let resp_json = serde_json::to_string(&resp)? + "\n";
                reader.get_mut().write_all(resp_json.as_bytes()).await?;
            }
            Action::Ping => {
                let resp = Response { status: Status::Ok, message: Some("pong".to_string()), path: None, keys: None };
                let resp_json = serde_json::to_string(&resp)? + "\n";
                reader.get_mut().write_all(resp_json.as_bytes()).await?;
            }
            Action::Shutdown => {
                 info!("Shutdown command received. Exiting immediately.");
                 std::process::exit(0);
            }
        }
    }
    Ok(())
}