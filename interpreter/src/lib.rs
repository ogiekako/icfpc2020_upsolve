#![allow(unused)]

extern crate anyhow;
extern crate im_rc as im;
extern crate itertools;
#[macro_use]
extern crate lazy_static;
extern crate serde;
extern crate wasm_bindgen;

use std::cell::RefCell;
use std::fmt::Formatter;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum Expr {
    Ap(CachedExpr, CachedExpr),
    Op(String, Vec<CachedExpr>),
    Num(i64),
    Var(String),
    Image(std::collections::BTreeSet<(i64, i64)>),
    Mod(CachedExpr),
    Func(Vec<String>, Rc<Expr>, Vec<CachedExpr>),
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
    fn num(&self) -> i64 {
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
    fn must_list_rev(&self) -> Vec<Expr> {
        match self.must_op() {
            ("nil", []) => vec![],
            ("cons", [x0, x1]) => {
                let mut res = x1.expr.borrow().must_list_rev();
                res.push(x0.expr.borrow().clone());
                res
            }
            _ => panic!("not list"),
        }
    }
    fn must_list(&self) -> Vec<Expr> {
        self.must_list_rev().into_iter().rev().collect()
    }
    fn must_image(&self) -> Vec<(i64, i64)> {
        match self {
            Image(img) => {
                let mut v = vec![];
                for p in img.iter() {
                    v.push(*p);
                }
                v
            }
            _ => panic!("not image"),
        }
    }

    fn cons(hd: CachedExpr, tl: CachedExpr) -> Expr {
        Ap(Ap(Op("cons".into(), vec![]).into(), hd).into(), tl)
    }
    fn nil() -> Expr {
        Op("nil".into(), vec![])
    }
    fn image(v: Vec<(i64, i64)>) -> Expr {
        let mut img = std::collections::BTreeSet::new();
        for p in v {
            img.insert(p);
        }
        Image(img)
    }
    fn modulate(&self, env: &Env) -> Expr {
        Mod(self.clone().into())
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
            Func(args, body, v) => Func(
                args,
                body,
                v.iter()
                    .map(|e| e.expr.borrow().reduce(env).into())
                    .collect(),
            ),
            Mod(_) | Num(_) | Image(_) => x,
            _ => panic!("unexpected expr after eval"),
        }
    }
    pub fn eval(&self, env: &Env) -> EvalResult {
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
                    Func(args, expr, v) => {
                        let mut v = v.clone();
                        v.push(r.clone());
                        Func(args.clone(), expr.clone(), v)
                    }
                    _ => panic!("not op or func l: {:?}", l),
                };
                e.eval(env)
            }),
            Var(name) => env.get(name).unwrap().eval(env),
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
                        eprintln!("invalid if0 arg {:?}", x0);
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
                ("modem", &[x0]) => x0.expr.borrow().clone().eval(env),
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
            Func(args, body, v) if args.len() == v.len() => {
                let mut env2 = im::hashmap! {};
                // let mut env = env.clone();
                for (a, x) in args.iter().zip(v.iter()) {
                    env2.insert(a.clone(), x.expr.borrow().clone());
                }
                replace_vars_once(&env2, body.as_ref().clone()).eval(&env)
            }
            _ => EvalResult::new(self.clone(), true),
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

pub type Env = im_rc::HashMap<String, Expr>;

pub fn default_env() -> Env {
    let mut env = Env::new();
    let data = vec![
        ("checkerboard", "", "ap ap s ap ap b s ap ap c ap ap b c ap ap b ap c ap c ap ap s ap ap b s ap ap b ap b ap ap s i i lt eq ap ap s mul i nil ap ap s ap ap b s ap ap b ap b cons ap ap s ap ap b s ap ap b ap b cons ap c div ap c ap ap s ap ap b b ap ap c ap ap b b add neg ap ap b ap s mul div ap ap c ap ap b b checkerboard ap ap c add 2"),
        ("pwr2", "", "ap ap s ap ap c ap eq 0 1 ap ap b ap mul 2 ap ap b pwr2 ap add -1"),
        ("statelessdraw", "", "ap ap c ap ap b b ap ap b ap b ap cons 0 ap ap c ap ap b b cons ap ap c cons nil ap ap c ap ap b cons ap ap c cons nil nil"),
        (":67108929", "", "ap ap b ap b ap ap s ap ap b ap b ap cons 0 ap ap c ap ap b b cons ap ap c cons nil ap ap c cons nil ap c cons"),
        ("f38" ,"x2 x0","ap ap ap if0 ap car x0 ( ap modem ap car ap cdr x0 , ap multipledraw ap car ap cdr ap cdr x0 ) ap ap ap interact x2 ap modem ap car ap cdr x0 ap send ap car ap cdr ap cdr x0"),
        ("interact", "x2 x4 x3", "ap ap f38 x2 ap ap x2 x4 x3"),
    ];
    for (name, _, _) in data.iter() {
        // add dummy to recognize as vars.
        env.insert(name.to_string(), Num(0));
    }
    for (name, args, expr) in data {
        let expr = parse_string(&env, expr);
        let args = if args.is_empty() {
            vec![]
        } else {
            args.split(' ').into_iter().map(str::to_string).collect()
        };
        env.insert(name.to_string(), Func(args, Rc::new(expr), vec![]));
    }
    for line in include_str!("../galaxy.txt").split("\n") {
        let ss = line.split(" = ").collect::<Vec<_>>();
        let (name, expr) = (ss[0], ss[1]);
        let e = parse_string(&env, expr);
        env.insert(name.to_string(), Func(vec![], Rc::new(e), vec![]));
    }

    env
}

impl CachedExpr {
    fn eval(&self, env: &Env) -> EvalResult {
        let EvalResult { expr, cacheable } = self.expr.borrow().eval(env);
        if cacheable {
            self.expr.replace(expr.clone());
        }
        EvalResult { expr, cacheable }
    }
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Ap(l, r) => write!(f, "ap {} {}", l.expr.borrow(), r.expr.borrow()),
            Expr::Op(s, v) => {
                let mut res = s.to_string();
                for e in v.iter() {
                    res = format!("ap {} {}", res, e.expr.borrow().clone());
                }
                write!(f, "{}", res)
            }
            Expr::Func(args, body, v) => write!(f, "<func>"),
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
        Var(s) => {
            if let Some(e) = env.get(&s) {
                e.clone()
            } else {
                Var(s)
            }
        }
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
                    .map(|s| s.parse::<i64>().unwrap())
                    .collect::<Vec<_>>();
                img.insert((xy[0], xy[1]));
            }
            Image(img)
        }
        "ap" => Ap(parse(env, &mut it).into(), parse(env, &mut it).into()),
        "add" | "b" | "c" | "car" | "cdr" | "cons" | "div" | "eq" | "i" | "isnil" | "lt" | "f"
        | "mod" | "dem" | "send" | "multipledraw" | "modem" | "vec" | "draw" | "if0" | "mul"
        | "neg" | "nil" | "s" | "t" => Op(s.into(), vec![]).into(),
        s => {
            if let Ok(i) = s.parse::<i64>() {
                Num(i)
            } else if env.contains_key(s)
                || s.chars().next().unwrap() == ':'
                || s.chars().next().unwrap() == 'x'
            {
                Var(s.to_string())
            } else {
                panic!("unknown var {}", s);
            }
        }
    }
}

#[wasm_bindgen]
pub struct GalaxyResult {
    state: String,
    images: Vec<Vec<(i64, i64)>>,
}

#[wasm_bindgen]
impl GalaxyResult {
    pub fn state(&self) -> String {
        self.state.clone()
    }
    pub fn image_count(&self) -> usize {
        self.images.len()
    }
    pub fn image(&self, i: usize) -> Image {
        Image {
            img: self.images[i]
                .iter()
                .map(|p| Point {
                    x: p.0 as _,
                    y: p.1 as _,
                })
                .collect(),
        }
    }
}

#[wasm_bindgen]
pub struct Image {
    img: Vec<Point>,
}

#[wasm_bindgen]
impl Image {
    pub fn count(&self) -> usize {
        self.img.len()
    }
    pub fn point(&self, i: usize) -> Point {
        self.img[i]
    }
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

#[wasm_bindgen]
pub fn galaxy_wasm(state: String, x: i32, y: i32) -> GalaxyResult {
    let (state, images) = galaxy(state, (x as _, y as _));
    GalaxyResult { state, images }
}

pub fn galaxy(state: String, vec: (i64, i64)) -> (String, Vec<Vec<(i64, i64)>>) {
    let env = default_env();

    let input = format!(
        "ap ap ap interact galaxy {} ap ap vec {} {}",
        state, vec.0, vec.1
    );
    let expr = parse_string(&env, &input);
    let v = expr.reduce(&env).must_list();

    let next_state = format!("{}", v[0]);

    let images: Vec<_> = v[1]
        .must_list()
        .into_iter()
        .map(|e| e.must_image())
        .collect();
    (next_state, images)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::iter::FromIterator;

    #[test]
    fn test_galaxy() {
        let env = default_env();
        for tc in vec![("nil", (0, 0))] {
            let (next_state, image) = galaxy(tc.0.to_string(), tc.1);
            assert_eq!(
                next_state,
                "ap ap cons 0 ap ap cons ap ap cons 0 nil ap ap cons 0 ap ap cons nil nil"
            );
            assert_eq!(
                image,
                vec![
                    vec![
                        (-3, 0),
                        (-3, 1),
                        (-2, -1),
                        (-2, 2),
                        (-1, -3),
                        (-1, -1),
                        (-1, 0),
                        (-1, 3),
                        (0, -3),
                        (0, -1),
                        (0, 1),
                        (0, 3),
                        (1, -3),
                        (1, 0),
                        (1, 1),
                        (1, 3),
                        (2, -2),
                        (2, 1),
                        (3, -1),
                        (3, 0)
                    ],
                    vec![(-8, -2), (-7, -3)],
                    vec![]
                ]
            );
        }
    }

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
