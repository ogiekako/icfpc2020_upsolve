use crate::*;
use anyhow::{anyhow, Result};
use std::io::prelude::*;
use std::{
    collections::HashMap,
    process::{Command, Stdio},
    str::FromStr,
};
use wasm_bindgen::prelude::*;

pub struct GalaxyEvaluator {
    env: Env,
}

impl GalaxyEvaluator {
    pub fn new() -> Self {
        let env = Env::new_galaxy();
        Self { env }
    }
}

impl common::Evaluator for GalaxyEvaluator {
    fn evaluate(&self, expr: &str) -> common::Node {
        let v: Value = expr.parse().unwrap();
        let res = evaluate(&self.env, &v).unwrap();
        res.parse().unwrap()
    }
    fn add_def(&mut self, s: &str) {
        self.env.add_parse(s).unwrap()
    }
}

pub struct Env(Vec<(String, Value)>);

impl Env {
    pub fn new() -> Self {
        Self(Vec::new())
    }
    pub fn add_parse(&mut self, line: &str) -> Result<()> {
        let v: Vec<_> = line.split(" = ").map(str::trim).collect();
        let name = format!("{}", v[0].parse::<Value>()?);
        self.0.push((name, v[1].parse()?));
        Ok(())
    }
    fn new_galaxy() -> Self {
        let mut env = Self::new();
        for line in include_str!("../galaxy.txt").split("\n") {
            env.add_parse(line).unwrap();
        }
        env
    }
}

#[cfg(target_os = "linux")]
fn eval_js(prog: &str) -> Result<String> {
    // for debug.
    let mut f = std::fs::File::create("/tmp/hoge.js")?;
    write!(f, "{}", prog).unwrap();

    let p = Command::new("node")
        // 100M.
        .args(&["--stack-size=100000"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    p.stdin.unwrap().write_all(js.as_bytes())?;
    let mut res = String::new();
    p.stdout.unwrap().read_to_string(&mut res)?;
    res = res.trim().into();
    eprintln!("Got result: {}", res);
    Ok(res)
}

#[cfg(target_arch = "wasm32")]
fn eval_js(prog: &str) -> Result<String> {
    Ok(js_eval_js(prog))
}

#[wasm_bindgen(module = "/js/wasm_define.js")]
#[cfg(target_arch = "wasm32")]
extern "C" {
    fn js_eval_js(s: &str) -> String;
}

pub fn evaluate(env: &Env, expr: &Value) -> Result<String> {
    let js = to_js_program(&env, &expr);
    eval_js(&js)
}

fn to_js_program(env: &Env, expr: &Value) -> String {
    use std::fmt::Write; // for write! to work for String

    let pre = include_str!("../js/prelude.js").to_string();

    let mut js = String::new();

    for (k, v) in env.0.iter() {
        writeln!(js, "const {} = new Lazy(() => {});", k, v).unwrap();
    }
    writeln!(js, "const result = to_string(eval({}));", expr).unwrap();
    writeln!(js, "console.log(result);").unwrap();
    writeln!(js, "return result;");

    format!("{}\n{}", pre, js)
}

#[derive(Debug, Eq, PartialEq)]
pub enum Value {
    Ap(Box<Value>, Box<Value>),
    Num(i64),
    Var(String),
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Ap(x, y) => write!(f, "ap({},{})", x, y),
            Value::Num(x) => write!(f, "{}n", x),
            Value::Var(x) => write!(f, "{}", x.replace(":", "f")),
        }
    }
}

impl FromStr for Value {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Value::parse(&mut s.split(" ").into_iter())
    }
}

impl Value {
    fn parse<'a>(i: &mut impl Iterator<Item = &'a str>) -> Result<Self> {
        Ok(match i.next().ok_or(anyhow!("iterator expected"))? {
            "ap" => Value::Ap(Box::new(Value::parse(i)?), Box::new(Value::parse(i)?)),
            s => match s.parse::<i64>() {
                Ok(x) => Value::Num(x),
                Err(_) => Value::Var(s.into()),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_parse() {
        use Value::*;

        for tc in vec![
            ("1", Num(1)),
            ("ap inc 1", Ap(Var("inc".into()).into(), Num(1).into())),
        ] {
            let got: Value = tc.0.parse().unwrap();
            assert_eq!(got, tc.1);
        }
    }

    #[test]
    fn test_evaluate() {
        let mut env = Env::new();
        for line in vec!["pwr2 = ap ap s ap ap c ap eq 0 1 ap ap b ap mul 2 ap ap b pwr2 ap add -1"]
        {
            env.add_parse(line).unwrap();
        }

        for tc in vec![
            ("1", "1"),
            ("ap pwr2 3", "8"),
            ("ap ap t 1 2", "1"),
            ("ap ap t 1 ap ap ap s i i ap ap s i i", "1"),
            ("ap ap cons 1 2", "ap ap cons 1 2"),
        ] {
            eprintln!("{}", tc.0);
            let expr: Value = tc.0.parse().unwrap();
            let got = evaluate(&env, &expr).unwrap();
            assert_eq!(got, tc.1);
        }
    }
}
