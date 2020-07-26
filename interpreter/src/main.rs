#![allow(unused)]

extern crate interpreter;

// extern crate clap;

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
    format!("ap ap ap interact galaxy {} ap ap vec 0 0", input)
}

fn run() {
    {
        // let state = "ap ap cons 2 ap ap cons ap ap cons 1 ap ap cons -1 nil ap ap cons 0 ap ap cons nil nil";
        let state = "ap ap cons 5 ap ap cons ap ap cons 2 ap ap cons 0 ap ap cons nil ap ap cons nil ap ap cons nil ap ap cons nil ap ap cons nil ap ap cons 0 nil ap ap cons 0 ap ap cons nil nil";
        let (next_state, images) = interpreter::galaxy(state.into(), (0, 0));
        eprintln!("finished computing galaxy: {} {:?}", next_state, images);

        return;
    }

    // let matches = clap::App::new("interpreter")
    //     .arg(
    //         clap::Arg::with_name("piston")
    //             .short("p")
    //             .long("piston")
    //             .help("Use piston game engine"),
    //     )
    //     .get_matches();

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
