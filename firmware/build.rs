use std::env;
use std::fs;
use std::path::PathBuf;

#[derive(Copy, Clone, Debug)]
enum ModeSetting {
    PumpGated,
    Continuous,
}

impl ModeSetting {
    fn parse(raw: &str) -> Option<Self> {
        let mut normalized = raw.trim().to_ascii_lowercase();
        normalized.retain(|c| c != '_' && c != '-');
        match normalized.as_str() {
            "continuous" => Some(Self::Continuous),
            "pumpgated" => Some(Self::PumpGated),
            _ => None,
        }
    }

    fn env_value(self) -> &'static str {
        match self {
            Self::Continuous => "Continuous",
            Self::PumpGated => "PumpGated",
        }
    }

    fn cfg_flag(self) -> &'static str {
        match self {
            Self::Continuous => "mode_continuous",
            Self::PumpGated => "mode_pump_gated",
        }
    }
}

fn main() {
    // Ensure proper linking for ESP32-S3 and ESP-IDF app descriptor
    println!("cargo:rustc-link-arg=-Tlinkall.x");

    // Register cfg for PSRAM mode detection
    println!("cargo:rustc-check-cfg=cfg(psram_mode_octal)");
    println!("cargo:rustc-check-cfg=cfg(psram_mode_quad)");
    println!("cargo:rustc-check-cfg=cfg(mode_continuous)");
    println!("cargo:rustc-check-cfg=cfg(mode_pump_gated)");

    // Export selected variables from .env so code can use env!(...) at compile time
    export_dotenv_vars();
}

fn export_dotenv_vars() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.parent().unwrap_or(&manifest_dir).to_path_buf();

    // Register rerun triggers for all candidate paths up-front so a later-created
    // .env invalidates the build-script cache even if none existed on the first run.
    let manifest_dotenv = manifest_dir.join(".env");
    let workspace_dotenv = workspace_root.join(".env");
    println!("cargo:rerun-if-changed={}", manifest_dotenv.display());
    println!("cargo:rerun-if-changed={}", workspace_dotenv.display());

    // Priority: FIRMWARE_DOTENV_PATH > firmware/.env > workspace/.env
    let dotenv_path = env::var("FIRMWARE_DOTENV_PATH")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .or_else(|| {
            if manifest_dotenv.exists() { Some(manifest_dotenv.clone()) } else { None }
        })
        .or_else(|| {
            if workspace_dotenv.exists() { Some(workspace_dotenv.clone()) } else { None }
        });

    let mut had_ssid = false;
    let mut had_password = false;
    let mut had_server_ip = false;
    let mut had_server_port = false;
    let mut empty_ssid = false;
    let mut empty_password = false;
    let mut empty_server_ip = false;
    let mut empty_server_port = false;
    let mut invalid_mode: Option<String> = None;
    let mut selected_mode: Option<ModeSetting> = None;

    if let Some(path) = dotenv_path.clone() {
        println!("cargo:warning=Using dotenv: {}", path.display());
        if let Ok(content) = fs::read_to_string(&path) {
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if let Some((k, v)) = parse_kv(line) {
                    match k {
                        // Accept both legacy and current names and export canonical SSID/PASSWORD
                        "SSID" => {
                            println!("cargo:rustc-env=SSID={v}");
                            had_ssid = true;
                            if v.is_empty() {
                                empty_ssid = true;
                            }
                        }
                        "WIFI_SSID" => {
                            println!("cargo:rustc-env=SSID={v}");
                            had_ssid = true;
                            if v.is_empty() {
                                empty_ssid = true;
                            }
                        }
                        "PASSWORD" => {
                            println!("cargo:rustc-env=PASSWORD={v}");
                            had_password = true;
                            if v.is_empty() {
                                empty_password = true;
                            }
                        }
                        "WIFI_PASSWORD" => {
                            println!("cargo:rustc-env=PASSWORD={v}");
                            had_password = true;
                            if v.is_empty() {
                                empty_password = true;
                            }
                        }
                        // Optional server endpoint
                        "SERVER_IP" => {
                            println!("cargo:rustc-env=SERVER_IP={v}");
                            had_server_ip = true;
                            if v.is_empty() {
                                empty_server_ip = true;
                            }
                        }
                        "SERVER_PORT" => {
                            println!("cargo:rustc-env=SERVER_PORT={v}");
                            had_server_port = true;
                            if v.is_empty() {
                                empty_server_port = true;
                            }
                        }
                        // Optional device identity/version
                        "DEVICE_ID" => println!("cargo:rustc-env=DEVICE_ID={v}"),
                        "DEVICE_VERSION" => println!("cargo:rustc-env=DEVICE_VERSION={v}"),
                        // Optional firmware log level (error|warn|info|debug|trace)
                        "LOG_LEVEL" => println!("cargo:rustc-env=LOG_LEVEL={v}"),
                        // Optional recording mode override (Continuous|PumpGated)
                        "MODE" => {
                            if v.is_empty() {
                                println!(
                                    "cargo:warning=MODE is set but empty. Falling back to PumpGated"
                                );
                            } else if let Some(mode) = ModeSetting::parse(v) {
                                selected_mode = Some(mode);
                            } else if invalid_mode.is_none() {
                                invalid_mode = Some(v.to_string());
                            }
                        }
                        // MQTT configuration (optional, for mqtt feature)
                        "MQTT_BROKER_IP" => println!("cargo:rustc-env=MQTT_BROKER_IP={v}"),
                        "MQTT_BROKER_PORT" => println!("cargo:rustc-env=MQTT_BROKER_PORT={v}"),
                        "MQTT_TOPIC" => println!("cargo:rustc-env=MQTT_TOPIC={v}"),
                        "MQTT_CLIENT_ID" => println!("cargo:rustc-env=MQTT_CLIENT_ID={v}"),
                        "MQTT_USERNAME" => println!("cargo:rustc-env=MQTT_USERNAME={v}"),
                        "MQTT_PASSWORD" => println!("cargo:rustc-env=MQTT_PASSWORD={v}"),
                        _ => {}
                    }
                }
            }
        }
        // Also watch the resolved path (covers FIRMWARE_DOTENV_PATH pointing elsewhere).
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!("cargo:rerun-if-env-changed=FIRMWARE_DOTENV_PATH");
    println!("cargo:rerun-if-env-changed=SSID");
    println!("cargo:rerun-if-env-changed=PASSWORD");
    println!("cargo:rerun-if-env-changed=SERVER_IP");
    println!("cargo:rerun-if-env-changed=SERVER_PORT");

    // Also consider already-set environment variables in the build environment
    if !had_ssid {
        if let Ok(val) = env::var("SSID") {
            if val.is_empty() {
                empty_ssid = true;
            }
            had_ssid = true;
        } else if let Ok(val) = env::var("WIFI_SSID") {
            if val.is_empty() {
                empty_ssid = true;
            }
            had_ssid = true;
            // Normalize to SSID for downstream compile-time env!
            println!("cargo:rustc-env=SSID={val}");
        }
    }
    if !had_password {
        if let Ok(val) = env::var("PASSWORD") {
            if val.is_empty() {
                empty_password = true;
            }
            had_password = true;
        } else if let Ok(val) = env::var("WIFI_PASSWORD") {
            if val.is_empty() {
                empty_password = true;
            }
            had_password = true;
            println!("cargo:rustc-env=PASSWORD={val}");
        }
    }
    if !had_server_ip {
        if let Ok(val) = env::var("SERVER_IP") {
            if val.is_empty() {
                empty_server_ip = true;
            }
            had_server_ip = true;
        }
    }
    if !had_server_port {
        if let Ok(val) = env::var("SERVER_PORT") {
            if val.is_empty() {
                empty_server_port = true;
            }
            had_server_port = true;
        }
    }

    if selected_mode.is_none() {
        if let Ok(val) = env::var("MODE") {
            if val.is_empty() {
                println!(
                    "cargo:warning=MODE is set in environment but empty. Falling back to PumpGated"
                );
            } else if let Some(mode) = ModeSetting::parse(&val) {
                selected_mode = Some(mode);
            } else if invalid_mode.is_none() {
                invalid_mode = Some(val);
            }
        }
    }

    // Emit helpful diagnostics before the compile step fails on env!(..)
    if !had_ssid || !had_password || !had_server_ip || !had_server_port {
        if dotenv_path.is_none() {
            println!(
                "cargo:warning=No .env found. Set FIRMWARE_DOTENV_PATH, or create firmware/.env or workspace .env"
            );
        }
        if !had_ssid {
            println!(
                "cargo:warning=Missing SSID. Provide SSID in .env (SSID=...) or export SSID before build"
            );
        }
        if !had_password {
            println!(
                "cargo:warning=Missing PASSWORD. Provide PASSWORD in .env (PASSWORD=...) or export PASSWORD before build"
            );
        }
        if !had_server_ip {
            println!(
                "cargo:warning=Missing SERVER_IP. Provide SERVER_IP in .env (e.g., 192.168.0.10) or export SERVER_IP before build"
            );
        }
        if !had_server_port {
            println!(
                "cargo:warning=Missing SERVER_PORT. Provide SERVER_PORT in .env (e.g., 3000) or export SERVER_PORT before build"
            );
        }
        println!(
            "cargo:warning=Dotenv precedence: FIRMWARE_DOTENV_PATH > firmware/.env > workspace .env"
        );
    }
    if empty_ssid {
        println!("cargo:warning=SSID is set but empty. Check your .env or environment");
    }
    if empty_password {
        println!("cargo:warning=PASSWORD is set but empty. Check your .env or environment");
    }
    if empty_server_ip {
        println!("cargo:warning=SERVER_IP is set but empty. Check your .env or environment");
    }
    if empty_server_port {
        println!("cargo:warning=SERVER_PORT is set but empty. Check your .env or environment");
    }

    if let Some(raw) = invalid_mode {
        println!(
            "cargo:warning=Unsupported MODE value '{raw}'. Using PumpGated. Accepts: Continuous or PumpGated"
        );
    }

    let final_mode = selected_mode.unwrap_or(ModeSetting::PumpGated);
    println!("cargo:rustc-env=MODE={}", final_mode.env_value());
    println!("cargo:rustc-cfg={}", final_mode.cfg_flag());
}

fn parse_kv(line: &str) -> Option<(&str, &str)> {
    let mut it = line.splitn(2, '=');
    let k = it.next()?.trim();
    let mut v = it.next()?.trim();
    // strip quotes if present
    if (v.starts_with('\"') && v.ends_with('\"')) || (v.starts_with('\'') && v.ends_with('\'')) {
        v = &v[1..v.len() - 1];
    }
    Some((k, v))
}
