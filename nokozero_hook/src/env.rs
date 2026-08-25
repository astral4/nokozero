//! Environment variable configuration.

use std::env::var;

/// Returns the driver address (`host:port`) for this instance.
pub(crate) fn connect_addr() -> Option<String> {
    var("NOKOZERO_CONNECT").ok()
}
