#![allow(unused)]
#[macro_use]
extern crate lazy_static;
use std::fmt::Formatter;
#[derive(Clone, Eq, PartialEq, Debug)]
pub enum Expr {
    Ap(CachedExpr, CachedExpr),
    Op(String, Vec<CachedExpr>),
    Num(isize),
    Var(String),
    Image(std::collections::BTreeSet<(isize, isize)>),
    Mod(CachedExpr),
}

use Expr::*;

impl Into<CachedExpr> for Expr {
    fn into(self) -> CachedExpr {
        CachedExpr {
            expr: std::rc::Rc::new(std::cell::RefCell::new(self)),
        }
    }
}

lazy_static! {
    static ref CNT: std::sync::Mutex<usize> = std::sync::Mutex::new(0);
}

impl Expr {
    fn num(&self) -> isize {
        match self {
            Expr::Num(x) => *x,
            _ => panic!("not a num: {}", self),
        }
    }
    fn must_op(&self) -> (&str, &[CachedExpr]) {
        match self {
            Op(s, v) => (s.as_str(), v.as_slice()),
            _ => panic!("not op"),
        }
    }

    fn cons(hd: CachedExpr, tl: CachedExpr) -> Expr {
        Ap(Ap(Op("cons".into(), vec![]).into(), hd).into(), tl)
    }
    fn nil() -> Expr {
        Op("nil".into(), vec![])
    }
    fn image(v: Vec<(isize, isize)>) -> Expr {
        let mut img = std::collections::BTreeSet::new();
        for p in v {
            img.insert(p);
        }
        Image(img)
    }
    fn modulate(&self, env: &Env) -> Expr {
        Mod(self.clone().into())
        // let s = self.eval(env);
        // match self.must_op() {
        //     ("cons", [x0, x1]) => {
        //         let x0 = x0.expr.borrow().modulate(env);
        //         let x1 = x1.expr.borrow().modulate(env);
        //     }
        //     ("nil", []) => {}
        // }
    }
    fn demodulate(&self, env: &Env) -> Expr {
        match self {
            Mod(x) => x.expr.borrow().clone(),
            _ => panic!("not Mod"),
        }
    }

    pub fn reduce(&self, env: &Env) -> Expr {
        let x = self.eval(env).expr;
        match x {
            Op(s, v) => Op(
                s,
                v.iter()
                    .map(|e| e.expr.borrow().reduce(env).into())
                    .collect(),
            ),
            Mod(_) | Num(_) | Image(_) => x,
            _ => panic!("unexpected expr after eval"),
        }
    }
    pub fn eval(&self, env: &Env) -> EvalResult {
        // if *CNT.lock().unwrap() > 100000 {
        //     panic!("too much eval");
        // }
        // *CNT.lock().unwrap() += 1;
        // eprintln!("{:?}", self);

        let ap = |x: &CachedExpr, y: &CachedExpr| Ap(x.clone(), y.clone());
        let t = Op("t".to_string(), vec![]);
        let f = Op("f".to_string(), vec![]);
        let bb = |b| if b { t.clone() } else { f.clone() };

        match self {
            Ap(l, r) => l.eval(env).then(|l| {
                let e = match l {
                    Op(name, v) => {
                        let mut v = v.clone();
                        v.push(r.clone());
                        Op(name.to_string(), v)
                    }
                    _ => panic!("non op l: {:?}", l),
                };
                e.eval(env)
            }),
            Var(ref name) => env.get(name).unwrap().eval(env),
            Op(name, v) => match (name.as_str(), &v.as_slice()) {
                ("add", [x, y]) => x.eval(env).then(|x| {
                    y.eval(env)
                        .then(|y| EvalResult::new(Num(x.num() + y.num()), true))
                }),
                ("mul", [x, y]) => x.eval(env).then(|x| {
                    y.eval(env)
                        .then(|y| EvalResult::new(Num(x.num() * y.num()), true))
                }),
                ("div", [x, y]) => x.eval(env).then(|x| {
                    y.eval(env)
                        .then(|y| EvalResult::new(Num(x.num() / y.num()), true))
                }),
                ("lt", [x, y]) => x.eval(env).then(|x| {
                    y.eval(env)
                        .then(|y| EvalResult::new(bb(x.num() < y.num()), true))
                }),
                ("eq", [x, y]) => x.eval(env).then(|x| {
                    y.eval(env)
                        .then(|y| EvalResult::new(bb(x.num() == y.num()), true))
                }),
                ("neg", [x]) => x.eval(env).then(|x| EvalResult::new(Num(-x.num()), true)),

                ("b", [x0, x1, x2]) => ap(x0, &ap(x1, x2).into()).eval(env),
                ("c", [x0, x1, x2]) => ap(&ap(x0, x2).into(), x1).eval(env),
                ("s", [x0, x1, x2]) => ap(&ap(x0, x2).into(), &ap(x1, x2).into()).eval(env),

                ("i", [x0]) => x0.eval(env),

                ("f", [x0, x1]) => x1.eval(env),
                ("t", [x0, x1]) => x0.eval(env),

                (s, [x0, x1, x2]) if s == "cons" || s == "vec" => {
                    ap(&ap(x2, x0).into(), x1).eval(env)
                }

                ("car", [x2]) => ap(x2, &t.into()).eval(env),
                ("cdr", [x2]) => ap(x2, &f.into()).eval(env),

                ("nil", [x0]) => t.eval(env),
                ("isnil", [x0]) => x0.eval(env).then(|x0| match x0 {
                    Op(s, v) if s == "nil" && v.len() == 0 => t.eval(env),
                    Op(s, v) if s == "cons" && v.len() == 2 => f.eval(env),
                    _ => panic!("unexpected x0: {:?}", x0),
                }),
                ("if0", &[x0, x1, x2]) => x0.eval(env).then(|x0| match x0 {
                    Num(0) => x1.eval(env),
                    Num(1) => x2.eval(env),
                    _ => {
                        // eprintln!("invalid if0 arg {:?}", x0);
                        x2.eval(env)
                    }
                }),
                ("mod", &[x0]) => x0
                    .expr
                    .borrow()
                    .eval(env)
                    .then(|x| x.modulate(env).eval(env)),
                ("dem", &[x0]) => x0
                    .expr
                    .borrow()
                    .eval(env)
                    .then(|x| x.demodulate(env).eval(env)),
                ("send", &[x0]) => panic!("send is not implemented."),
                ("modem", &[x0]) => {
                    // let mut env = env.clone();
                    // env.insert("x0".into(), x0.clone());
                    // let expr = parse_string(&env, "ap dem ap mod x0");
                    // let expr = replace_vars_once(&env, expr);
                    // expr.eval(&env)
                    x0.expr.borrow().clone().eval(env)
                }
                ("f38", &[x2, x0]) => {
                    let mut env = env.clone();
                    env.insert("x0".into(), x0.clone());
                    env.insert("x2".into(), x2.clone());
                    let f = parse_string(&env, "ap ap ap if0 ap car x0 ( ap modem ap car ap cdr x0 , ap multipledraw ap car ap cdr ap cdr x0 ) ap ap ap interact x2 ap modem ap car ap cdr x0 ap send ap car ap cdr ap cdr x0");
                    let expr = replace_vars_once(&env, f);
                    expr.eval(&env)
                }
                ("interact", &[x2, x4, x3]) => {
                    let mut env = env.clone();
                    env.insert("x2".into(), x2.clone());
                    env.insert("x3".into(), x3.clone());
                    env.insert("x4".into(), x4.clone());
                    let expr = parse_string(&env, "ap ap f38 x2 ap ap x2 x4 x3");
                    let expr = replace_vars_once(&env, expr);
                    expr.eval(&env)
                }
                ("multipledraw", &[x0]) => x0.eval(env).then(|x0| match x0.clone() {
                    Op(s, v) => match (s.as_str(), v.as_slice()) {
                        ("nil", []) => EvalResult::new(x0.clone(), true),
                        ("cons", [x0, x1]) => {
                            // eager eval.
                            Op("draw".into(), vec![x0.clone()]).eval(env).then(|img| {
                                Op("multipledraw".into(), vec![x1.clone()])
                                    .eval(env)
                                    .then(|lst| {
                                        Op(
                                            "cons".into(),
                                            vec![img.clone().into(), lst.clone().into()],
                                        )
                                        .eval(env)
                                    })
                            })
                        }
                        _ => panic!("unexpected list: {:?}", x0),
                    },
                    _ => panic!("unexpected list: {:?}", x0),
                }),
                ("draw", &[x0]) => x0.eval(env).then(|mut lst| {
                    let mut dummy = EvalResult::new(Num(0), true);
                    let mut img = std::collections::BTreeSet::new();

                    let mut lst = lst.clone();

                    loop {
                        let (name, xs) = if let Op(name, xs) = &lst {
                            (name, xs)
                        } else {
                            panic!("unexpected lst: {:?}", lst)
                        };
                        match lst.clone() {
                            Op(s, v) => match (s.as_str(), v.as_slice()) {
                                ("nil", []) => return EvalResult::new(Image(img), dummy.cacheable),
                                ("cons", [hd, tl]) => {
                                    dummy = dummy.then(|_| {
                                        hd.eval(env)
                                            .then(|hd| match hd {
                                                Op(s, v)
                                                    if (s == "cons" || s == "vec")
                                                        && v.len() == 2 =>
                                                {
                                                    v[0].eval(env).then(|x| {
                                                        v[1].eval(env).then(|y| {
                                                            img.insert((x.num(), y.num()));
                                                            tl.eval(env)
                                                        })
                                                    })
                                                }
                                                _ => panic!("unexpected point: {:?}", hd),
                                            })
                                            .then(|tl| {
                                                lst = tl.clone();
                                                // dummy result
                                                EvalResult::new(Num(0), true)
                                            })
                                    });
                                }
                                _ => panic!("unexpected lst: {:?}", lst),
                            },
                            _ => panic!("unexpected lst: {:?}", lst),
                        };
                    }
                }),

                _ => EvalResult::new(self.clone(), true),
            },
            Mod(_) | Num(_) | Image(_) => EvalResult::new(self.clone(), true),
        }
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct CachedExpr {
    expr: std::rc::Rc<std::cell::RefCell<Expr>>,
}
pub struct EvalResult {
    pub expr: Expr,
    cacheable: bool,
}

impl EvalResult {
    fn new(expr: Expr, cacheable: bool) -> Self {
        Self { expr, cacheable }
    }
    fn then(&self, op: impl FnOnce(&Expr) -> Self) -> Self {
        let EvalResult { expr, cacheable } = op(&self.expr);
        EvalResult {
            expr,
            cacheable: cacheable && self.cacheable,
        }
    }
}

pub type Env = im_rc::HashMap<String, CachedExpr>;

pub fn default_env() -> Env {
    let mut env = Env::new();
    for (name, expr) in vec![
        ("checkerboard", "ap ap s ap ap b s ap ap c ap ap b c ap ap b ap c ap c ap ap s ap ap b s ap ap b ap b ap ap s i i lt eq ap ap s mul i nil ap ap s ap ap b s ap ap b ap b cons ap ap s ap ap b s ap ap b ap b cons ap c div ap c ap ap s ap ap b b ap ap c ap ap b b add neg ap ap b ap s mul div ap ap c ap ap b b checkerboard ap ap c add 2"),
        ("pwr2", "ap ap s ap ap c ap eq 0 1 ap ap b ap mul 2 ap ap b pwr2 ap add -1"),
        ("statelessdraw", "ap ap c ap ap b b ap ap b ap b ap cons 0 ap ap c ap ap b b cons ap ap c cons nil ap ap c ap ap b cons ap ap c cons nil nil"),
        (":67108929", "ap ap b ap b ap ap s ap ap b ap b ap cons 0 ap ap c ap ap b b cons ap ap c cons nil ap ap c cons nil ap c cons")
    ] {
        env.insert(name.to_string(), CachedExpr{expr: std::rc::Rc::new(std::cell::RefCell::new(Num(0)))});
        env.insert(name.to_string(), parse_string(&env, &expr).into());
    }
    for line in include_str!("../galaxy.txt").split("\n") {
        let ss = line.split(" = ").collect::<Vec<_>>();
        let (name, expr) = (ss[0], ss[1]);
        let e = parse_string(&env, expr);
        env.insert(name.to_string(), e.into());
    }

    env
}

impl CachedExpr {
    fn eval(&self, env: &Env) -> EvalResult {
        let EvalResult { expr, cacheable } = self.expr.borrow().eval(env);
        if cacheable {
            *self.expr.borrow_mut() = expr.clone();
        }
        EvalResult { expr, cacheable }
    }
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Ap(l, r) => write!(f, "ap {} {}", l.expr.borrow(), r.expr.borrow()),
            Expr::Op(s, v) => {
                write!(f, "{}", s)?;
                for e in v.iter() {
                    write!(f, " {}", e.expr.borrow().clone())?;
                }
                Ok(())
            }
            Expr::Num(i) => write!(f, "{}", i),
            Expr::Var(s) => write!(f, "{}", s),
            Expr::Image(_) => write!(f, "<image>"),
            Expr::Mod(x) => write!(f, "ap mod {}", x.expr.borrow()),
        }
    }
}

fn replace_vars_once(env: &Env, expr: Expr) -> Expr {
    match expr {
        Ap(l, r) => {
            let l = replace_vars_once(env, l.expr.borrow().clone());
            let r = replace_vars_once(env, r.expr.borrow().clone());
            Ap(l.into(), r.into())
        }
        Var(s) => env.get(&s).unwrap().expr.borrow().clone(),
        _ => expr,
    }
}

pub fn parse_string(env: &Env, expr: &str) -> Expr {
    parse(
        env,
        &mut expr.split(" ").map(String::from).into_iter().peekable(),
    )
}

fn parse(
    env: &Env,
    mut it: &mut std::iter::Peekable<impl std::iter::Iterator<Item = String>>,
) -> Expr {
    use Expr::*;

    let s = it.next().expect("iterator exhausted");
    match s.as_str() {
        "(" => {
            let mut lst = vec![];
            loop {
                if it.peek().unwrap() == ")" {
                    it.next().unwrap();
                    break;
                }
                lst.push(parse(env, it));
                if it.peek().unwrap() == "," {
                    it.next().unwrap();
                }
            }
            let mut res = Expr::nil();
            for x in lst.into_iter().rev() {
                res = Expr::cons(x.into(), res.into());
            }
            res
        }
        s if s.starts_with("[") => {
            let mut img = std::collections::BTreeSet::new();
            for p in s.trim_matches(|c| c == '[' || c == ']').split(";") {
                if p.is_empty() {
                    break;
                }
                let xy = p
                    .split(",")
                    .map(|s| s.parse::<isize>().unwrap())
                    .collect::<Vec<_>>();
                img.insert((xy[0], xy[1]));
            }
            Image(img)
        }
        "ap" => Ap(parse(env, &mut it).into(), parse(env, &mut it).into()),
        "add" | "b" | "c" | "car" | "cdr" | "cons" | "div" | "eq" | "i" | "isnil" | "lt" | "f"
        | "mod" | "dem" | "send" | "multipledraw" | "modem" | "interact" | "f38" | "vec"
        | "draw" | "if0" | "mul" | "neg" | "nil" | "s" | "t" => Op(s.into(), vec![]).into(),
        s => {
            if let Ok(i) = s.parse::<isize>() {
                Num(i)
            } else if env.contains_key(s) || s.chars().next().unwrap() == ':' {
                Var(s.to_string())
            } else {
                panic!("unknown var {}", s);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::iter::FromIterator;

    #[test]
    fn test() {
        for tc in vec![
            ("ap ap add 1 2", "3"),
            ("ap ap add 1 ap ap add 2 3", "6"),
            ("ap ap ap if0 0 2 :1", "2"),
            ("ap ap ap if0 1 :1 3", "3"),
            ("ap ap t 1 :1", "1"),
            ("ap ap f :1 1", "1"),
            ("ap pwr2 0", "1"),
            ("ap pwr2 1", "2"),
            ("ap pwr2 4", "16"),
            ("( )", "nil"),
            ("( 1 , 2 )", "ap ap cons 1 ap ap cons 2 nil"),
            ("ap car ( 1 , 2 )", "1"),
            ("ap draw nil", "[]"),
            ("ap draw ap ap cons ap ap vec 1 1 nil", "[1,1]"),
            ("ap draw ap ap checkerboard 3 0", "[0,0;0,2;1,1;2,0;2,2]"),
            ("ap multipledraw nil", "nil"),
            (
                "ap multipledraw ap ap cons ( ap ap vec 1 1 ) nil",
                "( [1,1] )",
            ),
            ("ap multipledraw ( ( ap ap vec 1 1 ) )", "( [1,1] )"),
            ("ap modem ( 1 , 1 )", "( 1 , 1 )"),
            (
                "ap ap f38 nil ( 0 , 0 , ( ( ap ap vec 1 1 ) ) )",
                "( 0 , ( [1,1] ) )",
            ),
            (
                "ap ap ap interact statelessdraw nil ap ap vec 1 0",
                "( nil , ( [1,0] ) )",
            ),
            (
                "ap ap ap interact :67108929 nil ap ap vec 0 0",
                "( ( ap ap vec 0 0 ) , ( [0,0] ) )",
            ),
            (
                "ap ap ap interact :67108929 ( ap ap vec 0 0 ) ap ap vec 2 3",
                "( ( ap ap vec 2 3 , ap ap vec 0 0 ) , ( [0,0;2,3] ) )",
            ),
            // (
            //     "ap ap ap interact galaxy nil ap ap vec 0 0",
            //     "ap cons ap cons 0 ap cons ap cons 0 nil ap cons 0 ap cons nil nil ap cons ap cons [] ap cons [] ap cons [] nil nil"
            // )
        ] {
            *CNT.lock().unwrap() = 0;

            eprintln!("--- testing: {}", tc.0);
            let env = default_env();

            let e1 = parse_string(&env, tc.0);
            let e2 = parse_string(&env, tc.1);

            eprintln!("e1: {}", e1);
            eprintln!("e2: {}", e2);

            let e1 = e1.reduce(&env);
            let e2 = e2.reduce(&env);

            eprintln!("e1.eval: {}", e1);
            eprintln!("e2.eval: {}", e2);

            assert_eq!(e1, e2);
        }
    }
}
