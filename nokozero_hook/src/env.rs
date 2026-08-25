//! Environment variable configuration.

use std::env::var;

/// Returns whether the game should run in headless mode.
pub(crate) fn headless() -> bool {
    var("NOKOZERO_HEADLESS").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// Returns the driver address (`host:port`) for this instance.
pub(crate) fn connect_addr() -> Option<String> {
    var("NOKOZERO_CONNECT").ok()
}
