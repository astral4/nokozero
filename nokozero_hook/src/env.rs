//! Environment variable configuration.

use crate::log::fatal;
use crate::practice::MAX_CHARACTER;
use std::env::var_os;

pub(crate) struct Config {
    /// The driver address (`host:port`) for this instance.
    pub(crate) connect_addr: String,
    /// The character to select at the character-select screen.
    pub(crate) character: u32,
    /// Whether the game should run in headless mode.
    pub(crate) headless: bool,
}

impl Config {
    /// Reads and validates the environment variables. Aborts if there are errors.
    pub(crate) fn from_env() -> Self {
        let mut errors = Vec::new();

        let connect_addr = required("NOKOZERO_CONNECT", &mut errors);

        let mut character = None;
        if let Some(raw) = required("NOKOZERO_CHARACTER", &mut errors) {
            match raw.parse() {
                Ok(value) if value <= MAX_CHARACTER => character = Some(value),
                _ => errors.push(format!(
                    "NOKOZERO_CHARACTER must be 0-{MAX_CHARACTER}, got {raw:?}"
                )),
            }
        }

        let mut headless = None;
        if let Some(raw) = required("NOKOZERO_HEADLESS", &mut errors) {
            match raw.to_ascii_lowercase().as_str() {
                "1" | "true" => headless = Some(true),
                "0" | "false" => headless = Some(false),
                _ => errors.push(format!(
                    "NOKOZERO_HEADLESS must be 0, 1, false, or true, got {raw:?}"
                )),
            }
        }

        let (Some(connect_addr), Some(character), Some(headless)) =
            (connect_addr, character, headless)
        else {
            fatal!("invalid configuration:\n  {}", errors.join("\n  "));
        };

        eprintln!(
            "nokozero_hook::env: connect={connect_addr} character={character} headless={headless}"
        );

        Self {
            connect_addr,
            character,
            headless,
        }
    }
}

/// Reads `name` as UTF-8, recording an error and returning `None` if it is unset or malformed.
fn required(name: &str, errors: &mut Vec<String>) -> Option<String> {
    let Some(raw) = var_os(name) else {
        errors.push(format!("{name} is not set"));
        return None;
    };
    #[expect(clippy::unnecessary_debug_formatting)]
    match raw.into_string() {
        Ok(value) => Some(value),
        Err(raw) => {
            errors.push(format!("{name} must be valid UTF-8, got {raw:?}"));
            None
        }
    }
}
