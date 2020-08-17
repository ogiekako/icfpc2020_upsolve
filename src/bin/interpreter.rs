#![allow(unused)]

use app::*;

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
    let start = std::time::Instant::now();

    for (name, eval) in vec![
        (
            "reduce_evaluator",
            Box::new(reduce_evaluator::Eval::new()) as Box<dyn common::Evaluator>,
        ),
        ("js_gen_evaluator", Box::new(gen_js::GalaxyEvaluator::new())),
    ] {
        let g = common::G::new(eval);

        let state =  "ap ap cons 3 ap ap cons ap ap cons 0 ap ap cons ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 nil ap ap cons nil ap ap cons 0 nil ap ap cons 0 ap ap cons nil nil";
        let vector = (0, 0);
        let want_state = "ap ap cons 3 ap ap cons ap ap cons 0 ap ap cons ap ap cons 1 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 2 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 nil ap ap cons nil ap ap cons 0 nil ap ap cons 0 ap ap cons nil nil";

        let next_state = g
            .galaxy(state.into(), vector.0, vector.1, "".into())
            .state();

        assert_eq!(next_state, want_state);

        let d = std::time::Instant::now() - start;
        eprintln!("computed in {:?}", d);
    }
}
