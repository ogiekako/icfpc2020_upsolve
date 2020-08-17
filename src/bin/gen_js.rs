use anyhow::Result;
use app::*;
use std::io::prelude::*;

// Convert lambda expressions to SKI combinator.
fn main() {
    let child = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || run2())
        .unwrap();
    child.join().unwrap().unwrap();
}

fn run() -> Result<()> {
    let mut env = gen_js::Env::new();
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line.contains(" = ") {
            env.add_parse(&line)?;
            continue;
        }
        let v: gen_js::Value = line.parse()?;
        let res = gen_js::evaluate(&env, &v)?;
        println!("{}", res);
    }
    Ok(())
}

fn run2() -> Result<()> {
    let start = std::time::Instant::now();

    let g = common::G::new(Box::new(gen_js::GalaxyEvaluator::new()));

    let state =  "ap ap cons 3 ap ap cons ap ap cons 0 ap ap cons ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 nil ap ap cons nil ap ap cons 0 nil ap ap cons 0 ap ap cons nil nil";
    let vector = (0, 0);
    let want_state = "ap ap cons 3 ap ap cons ap ap cons 0 ap ap cons ap ap cons 1 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 2 ap ap cons 0 ap ap cons 0 ap ap cons 0 ap ap cons 0 nil ap ap cons nil ap ap cons 0 nil ap ap cons 0 ap ap cons nil nil";

    let got = g
        .galaxy(state.into(), vector.0, vector.1, "".into())
        .state();

    assert_eq!(got, want_state);

    let d = std::time::Instant::now() - start;
    eprintln!("computed in {:?}", d);

    Ok(())
}
