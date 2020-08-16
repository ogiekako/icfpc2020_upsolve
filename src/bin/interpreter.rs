#![allow(unused)]

use app::galaxy::*;

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

    let g = G::new();

    let state =  "ap ap cons 3 ap ap cons ap ap cons 0 ap ap cons ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 nil ap ap cons nil ap ap cons 0 nil ap ap cons 0 ap ap cons nil nil";
    let vector = (0, 0);
    let want_state = Some("ap ap cons 3 ap ap cons ap ap cons 0 ap ap cons ap ap cons 1 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 2 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 nil ap ap cons nil ap ap cons 0 nil ap ap cons 0 ap ap cons nil nil");

    // let state = "ap ap cons 5 ap ap cons ap ap cons 2 ap ap cons 0 ap ap cons nil ap ap cons nil ap ap cons nil ap ap cons nil ap ap cons nil ap ap cons 0 nil ap ap cons 8 ap ap cons nil nil";
    // let vector = (18, 3);
    // let want_state: Option<&str> = None;

    let next_state = g
        .galaxy(
            state.into(),
            vector.0,
            vector.1,
            API_KEY.lock().unwrap().as_str(),
        )
        .state();

    if let Some(want_state) = want_state {
        assert_eq!(next_state, want_state);
    }

    let d = std::time::Instant::now() - start;
    eprintln!("computed in {:?}", d);
}
