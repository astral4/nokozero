use std::env::var;
use std::path::PathBuf;

const TARGET: &str = "i686-pc-windows-gnu";

fn main() {
    let manifest_dir = var("CARGO_MANIFEST_DIR").unwrap();
    let artifact_dir = PathBuf::from(manifest_dir)
        .parent()
        .expect("could not find workspace root")
        .join("target")
        .join(TARGET)
        .join("release");

    let hook_path = artifact_dir.join("nokozero_hook.dll");

    assert!(
        hook_path.exists(),
        "Hook library was not found at {}. \
         Build the nokozero_hook crate first: \
         cargo build -p nokozero_hook --target {TARGET} --release",
        hook_path.display()
    );

    println!("cargo:rustc-env=HOOK_PATH={}", hook_path.display());
    println!("cargo:rerun-if-changed=../nokozero_hook/src");
    println!("cargo:rerun-if-changed=../nokozero_hook/Cargo.toml");
}
