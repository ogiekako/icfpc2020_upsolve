use app::*;

use anyhow::Result;
use std::io::prelude::*;

// Convert lambda expressions to SKI combinator.
fn main() {
    let child = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || run())
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
