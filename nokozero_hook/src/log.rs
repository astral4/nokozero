//! Helper for logging an error and aborting the process.

/// Reports a message to stderr and aborts the process.
macro_rules! fatal {
    ($($arg:tt)*) => {{
        ::std::eprintln!(
            "{}: {}",
            ::std::module_path!(),
            ::std::format_args!($($arg)*),
        );
        ::std::process::abort()
    }};
}

pub(crate) use fatal;
