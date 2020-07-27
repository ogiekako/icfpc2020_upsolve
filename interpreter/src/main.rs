#![allow(unused)]

extern crate interpreter;

// extern crate clap;

use interpreter::*;

use anyhow::{anyhow, bail, Result};
use std::io::Read;

fn main() {
    let child = std::thread::Builder::new().stack_size(256 * 1024 * 1024).spawn(move || run()).unwrap();
    child.join().unwrap();
}

fn galaxy(input: &str) -> String {
    format!("ap ap ap interact galaxy {} ap ap vec 0 0", input)
}

fn run() {
    let start = std::time::Instant::now();

    let g = interpreter::G::new();

    let state =  "ap ap cons 3 ap ap cons ap ap cons 0 ap ap cons ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 nil ap ap cons nil ap ap cons 0 nil ap ap cons 0 ap ap cons nil nil".into();
    let vector = (0, 0);

    let next_state = g.galaxy(state, vector.0, vector.1, "").state();
    assert_eq!(next_state, "ap ap cons 3 ap ap cons ap ap cons 0 ap ap cons ap ap cons 1 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 2 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 nil ap ap cons nil ap ap cons 0 nil ap ap cons 0 ap ap cons nil nil");

    let d = std::time::Instant::now() - start;
    eprintln!("computed in {:?}", d);
}
