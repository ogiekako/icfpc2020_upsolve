#![allow(unused)]

use anyhow::{anyhow, bail, Result};
use std::io::Read;

// Convert lambda expressions to SKI combinator.
fn main() {
    let child = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || run())
        .unwrap();
    child.join().unwrap();
}

fn run() -> Result<()> {
    let i = 0;
    println!("{}", 42);
    Ok(())
}
