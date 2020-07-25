#![allow(unused)]

extern crate interpreter;

extern crate console_error_panic_hook;

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

fn galaxy(input: &str) -> String {
    format!("ap ap ap interact galaxy {} ap ap vec 1000 1000", input)
}

fn run() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    let mut input = String::new();

    std::io::stdin().read_to_string(&mut input);
    if input.is_empty() {
        input = galaxy("nil");
        input = galaxy(
            "ap ap cons 2 ap ap cons ap ap cons 1 ap ap cons 1 nil ap ap cons 0 ap ap cons nil nil",
        );
    } else {
        input = galaxy(&input.trim());
    }

    let mut env = default_env();

    let e = parse_string(&env, &input);
    eprintln!("evaluating {}", &e);
    let res = e.eval(&env).expr;
    println!("result: {}", res.reduce(&env));
}
