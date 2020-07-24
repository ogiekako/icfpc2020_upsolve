#![allow(unused)]

extern crate anyhow;
extern crate itertools;
#[macro_use]
extern crate lazy_static;
extern crate im_rc;
// extern crate im;
extern crate interpreter;

use interpreter::*;

use anyhow::{anyhow, bail, Result};
use std::io::Read;

fn main() {
    let child = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || run())
        .unwrap();
    child.join().unwrap();
}

fn run() {
    let mut input = String::new();

    std::io::stdin().read_to_string(&mut input);
    if input.is_empty() {
        input = "ap ap ap interact galaxy nil ap ap vec 0 0".into();
    }
    let input = input.trim();

    let mut env = default_env();

    let e = parse_string(&env, &input);
    eprintln!("evaluating {}", &e);
    let res = e.eval(&env).expr;
    println!("result: {}", res.reduce(&env));
}
