#![allow(unused)]

use anyhow::{anyhow, bail, Result};
use std::io::BufRead;
use std::io::Read;

use itertools::Itertools;

// Convert lambda expressions to SKI combinator.
fn main() {
    let child = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || run().unwrap())
        .unwrap();
    child.join().unwrap();
}

fn run() -> Result<()> {
    // TODO
    Ok(())
}
