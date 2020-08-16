#![allow(unused)]

use anyhow::{anyhow, bail, Result};
use std::io::Read;

// Convert lambda expressions to SKI combinator.
fn main() {
    let child = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || run())
        .unwrap();
    child.join().unwrap();
}

fn run() -> Result<()> {
    let expr = Term::parse(&mut std::io::stdin().bytes().map(|c| c.unwrap().into()));
    println!("{:?}", expr.to_ski());
    Ok(())
}

impl Term {
    fn parse(i: &mut impl Iterator<Item = char>) -> Term {
        match i.next().unwrap() {
            // TODO: check the number of )'s more strictly.
            ')' | ' ' => Term::parse(i),
            '(' => {
                let x = Term::parse(i);
                let y = Term::parse(i);
                Term::Ap(Box::new(x), Box::new(y))
            }
            '\\' => {
                let v = i.take_while(|c| c.is_alphabetic()).collect::<String>();
                let t = Term::parse(i);
                Term::Lambda(v, Box::new(t))
            }
            'S' => Term::S,
            'K' => Term::K,
            'I' => Term::I,
            c => {
                let v = std::iter::once(c)
                    .chain(i.take_while(|c| c.is_alphanumeric()))
                    .collect::<String>();
                Term::Var(v)
            }
        }
    }

    // Converts any lambda term to SKI form.
    // https://en.wikipedia.org/wiki/Combinatory_logic#Completeness_of_the_S-K_basis
    fn to_ski(self) -> Self {
        match self {
            Term::Ap(x, y) => Term::Ap(x.to_ski().into(), y.to_ski().into()), // Rule 2.
            Term::Lambda(x, y) => {
                if !has_free(&y, &x) {
                    // Rule 3.
                    return Term::Ap(Term::K.into(), y.to_ski().into());
                };
                match *y {
                    // Rule 4.
                    Term::Var(_) => Term::I,
                    // Rule 5. T[λx.λy.E] => T[λx.T[λy.E]]
                    Term::Lambda(y, e) => {
                        Term::Lambda(x, Term::Lambda(y, e).to_ski().into()).to_ski()
                    }
                    // Rule 6. T[λx.(E₁ E₂)] => ((S T[λx.E₁]) T[λx.E₂])
                    Term::Ap(e1, e2) => Term::Ap(
                        Term::Ap(
                            Term::S.into(),                              //
                            Term::Lambda(x.clone(), e1).to_ski().into(), //
                        )
                        .into(),
                        Term::Lambda(
                            x.clone(), //
                            e2,        //
                        )
                        .to_ski()
                        .into(),
                    ),
                    y => y,
                }
            }
            t => t, // Rule 1.
        }
    }
}

fn has_free(t: &Term, v: &str) -> bool {
    match t {
        Term::Ap(x, y) => has_free(x, v) || has_free(y, v),
        Term::Lambda(x, y) => v != x && has_free(y, v),
        Term::Var(x) => v == x,
        _ => false,
    }
}

#[derive(Eq, PartialEq, Debug)]
enum Term {
    // (t t)
    Ap(Box<Term>, Box<Term>),
    // \x t
    Lambda(String, Box<Term>),
    // x
    Var(String),
    // SKI
    S,
    K,
    I,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_parse() {
        use Term::*;
        for tc in &[
            (r"\x x", Lambda("x".into(), Var("x".into()).into())),
            (
                r"\x (x x)",
                Lambda(
                    "x".into(),
                    Ap(Var("x".into()).into(), Var("x".into()).into()).into(),
                ),
            ),
            (
                r"\x \y ((x y) (y x))",
                Lambda(
                    "x".into(),
                    Lambda(
                        "y".into(),
                        Ap(
                            Ap(Var("x".into()).into(), Var("y".into()).into()).into(),
                            Ap(Var("y".into()).into(), Var("x".into()).into()).into(),
                        )
                        .into(),
                    )
                    .into(),
                ),
            ),
        ] {
            let got = Term::parse(&mut tc.0.chars());
            assert_eq!(got, tc.1);
        }
    }

    #[test]
    fn test_has_free() {
        use Term::*;
        for tc in &[
            (r"\x x", "x", false),
            (r"\y x", "x", true),
            (r"\y (y y)", "x", false),
            (r"\y (y x)", "x", true),
        ] {
            eprintln!("{}", tc.0);
            let t = Term::parse(&mut tc.0.chars());
            let got = has_free(&t, tc.1);
            assert_eq!(got, tc.2);
        }
    }

    #[test]
    fn test_to_ski() {
        use Term::*;
        for tc in &[(r"\x \y (y x)", "((S (K (S I))) ((S (K K)) I))")] {
            eprintln!("{}", tc.0);
            let t = Term::parse(&mut tc.0.chars());
            let got = t.to_ski();
            let want = Term::parse(&mut tc.1.chars());
            assert_eq!(got, want);
        }
    }
}
