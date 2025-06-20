use anyhow::Result;
use dashmap::DashMap;
use log::{error, info};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::signal;
use uuid::Uuid;

type Db = Arc<DashMap<String, Vec<u8>>>;

#[derive(Serialize, Deserialize, Debug)] struct Command { action: Action, key: Option<String>, data_len: Option<usize> }
#[derive(Serialize, Deserialize, Debug, PartialEq)] enum Action { Put, Get, Delete, ListKeys, Ping, Shutdown }
#[derive(Serialize, Deserialize, Debug)] struct Response { status: Status, message: Option<String>, keys: Option<Vec<String>>, data_len: Option<usize> }
#[derive(Serialize, Deserialize, Debug)] enum Status { Ok, Error }

const BIND_ADDRESS: &str = "127.0.0.1:56789";

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let listener = TcpListener::bind(BIND_ADDRESS).await?;
    info!("ArrowShelf daemon listening on tcp://{}", BIND_ADDRESS);
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
                info!("Ctrl+C received, shutting down gracefully.");
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
        let bytes_read_result = reader.read_line(&mut line).await;

        match bytes_read_result {
            Ok(0) => {
                info!("Client disconnected cleanly.");
                break;
            }
            Ok(_) => {}
            Err(e) => {
                if e.kind() == std::io::ErrorKind::ConnectionReset {
                    info!("Connection reset by client (expected for multiprocessing workers).");
                } else {
                    error!("Connection read error: {}", e);
                }
                break;
            }
        }

        let cmd: Command = match serde_json::from_str(line.trim()) {
            Ok(c) => c,
            Err(e) => {
                let resp = Response { status: Status::Error, message: Some(format!("Invalid command JSON: {}", e)), keys: None, data_len: None };
                let resp_json = serde_json::to_string(&resp)? + "\n";
                reader.get_mut().write_all(resp_json.as_bytes()).await?;
                continue;
            }
        };

        match cmd.action {
            Action::Ping => {
                let resp = Response { status: Status::Ok, message: Some("pong".to_string()), keys: None, data_len: None };
                let resp_json = serde_json::to_string(&resp)? + "\n";
                reader.get_mut().write_all(resp_json.as_bytes()).await?;
            }
            Action::Put => {
                let key = Uuid::new_v4().to_string();
                let data_len = cmd.data_len.ok_or_else(|| anyhow::anyhow!("'put' requires 'data_len'"))?;
                let mut data = vec![0u8; data_len];
                reader.read_exact(&mut data).await?;
                db.insert(key.clone(), data);
                let resp = Response { status: Status::Ok, message: Some(key), keys: None, data_len: None };
                let resp_json = serde_json::to_string(&resp)? + "\n";
                reader.get_mut().write_all(resp_json.as_bytes()).await?;
            }
            Action::Get => {
                let key = cmd.key.ok_or_else(|| anyhow::anyhow!("'get' requires 'key'"))?;
                if let Some(data_entry) = db.get(&key) {
                    let data = data_entry.value();
                    let resp = Response { status: Status::Ok, message: None, keys: None, data_len: Some(data.len()) };
                    let resp_json = serde_json::to_string(&resp)? + "\n";
                    reader.get_mut().write_all(resp_json.as_bytes()).await?;
                    reader.get_mut().write_all(data).await?;
                } else {
                    let resp = Response { status: Status::Error, message: Some("Key not found".to_string()), keys: None, data_len: None };
                    let resp_json = serde_json::to_string(&resp)? + "\n";
                    reader.get_mut().write_all(resp_json.as_bytes()).await?;
                }
            }
            Action::Delete => {
                let key = cmd.key.ok_or_else(|| anyhow::anyhow!("'delete' requires 'key'"))?;
                db.remove(&key);
                let resp = Response { status: Status::Ok, message: Some(format!("Deleted key {}", key)), keys: None, data_len: None };
                let resp_json = serde_json::to_string(&resp)? + "\n";
                reader.get_mut().write_all(resp_json.as_bytes()).await?;
            }
            Action::ListKeys => {
                let keys: Vec<String> = db.iter().map(|entry| entry.key().clone()).collect();
                let resp = Response { status: Status::Ok, message: None, keys: Some(keys), data_len: None };
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