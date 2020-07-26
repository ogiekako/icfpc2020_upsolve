#![allow(unused)]

extern crate anyhow;
extern crate im_rc as im;
extern crate itertools;
#[macro_use]
extern crate lazy_static;
extern crate console_error_panic_hook;
extern crate serde;
extern crate wasm_bindgen;

#[cfg(target_os = "linux")]
extern crate reqwest;

use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::Formatter;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
struct ScopeCall<F: FnMut()> {
    c: F,
}
impl<F: FnMut()> Drop for ScopeCall<F> {
    fn drop(&mut self) {
        (self.c)();
    }
}

macro_rules! defer {
    ($e:expr) => {
        let _scope_call = ScopeCall {
            c: || -> () {
                $e;
            },
        };
    };
}
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
    static ref INDENT: std::sync::Mutex<usize> = std::sync::Mutex::new(0);
    static ref API_KEY: std::sync::Mutex<String> =
        std::sync::Mutex::new(std::env::var("API_KEY").unwrap_or(String::new()));
    static ref DEJA: std::sync::Mutex<HashSet<String>> = std::sync::Mutex::new(HashSet::new());
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
    fn must_list_rev(&self, env: &Env) -> Vec<Expr> {
        // eprintln!("must_list_rev: {}", self);
        let e = self.eval(env).expr;
        // eprintln!("must_list_rev evaluated: {}", e);
        match e.must_op() {
            ("nil", []) => vec![],
            ("cons", [x0, x1]) => {
                let mut res = x1.expr.borrow().clone().must_list_rev(env);
                res.push(x0.expr.borrow().clone());
                res
            }
            _ => panic!("not list"),
        }
    }
    fn must_list(&self, env: &Env) -> Vec<Expr> {
        self.must_list_rev(env).into_iter().rev().collect()
    }
    fn must_point(&self, env: &Env) -> (i64, i64) {
        let e = self.eval(env).expr;
        match e.must_op() {
            (s, [x, y]) if s == "cons" || s == "vec" => {
                let x = x.eval(env).expr;
                let y = y.eval(env).expr;
                (x.num(), y.num())
            }
            _ => panic!("not vec: {}", self),
        }
    }
    fn must_image(&self, env: &Env) -> Vec<(i64, i64)> {
        let e = self.eval(env).expr;
        match e {
            Image(img) => {
                let mut v = vec![];
                for p in img.iter() {
                    v.push(*p);
                }
                v
            }
            _ => panic!("not image: {}", e),
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

    fn modulate(&self, env: &Env) -> String {
        // eprintln!("modulate: {}", self);
        let e = self.eval(env).expr;
        // eprintln!("modulate e: {}", e);

        match e {
            Num(n) => {
                let mut res = String::new();
                let n = if n >= 0 {
                    res.push_str("01");
                    n
                } else {
                    res.push_str("10");
                    n.abs()
                };

                let keta = 64 - n.leading_zeros();
                let t = (keta + 3) / 4;

                for _ in 0..t {
                    res.push('1');
                }
                res.push('0');

                for i in (0..4 * t).rev() {
                    res.push(if (n >> i & 1) == 1 { '1' } else { '0' });
                }
                res
            }
            Op(s, v) => match (s.as_str(), v.as_slice()) {
                ("nil", []) => "00".into(),
                (s, [hd, tl]) if s == "cons" || s == "vec" => {
                    let hd = hd.expr.borrow().clone();
                    let tl = tl.expr.borrow().clone();
                    "11".to_string() + &hd.modulate(env) + &tl.modulate(env)
                }
                _ => panic!("unexpected op {}", s),
            },
            _ => panic!("unexpected type to modulate: {:?}", self),
        }
    }
    fn demodulate(s: &str) -> Expr {
        Expr::demodulate_iter(&mut s.chars().map(|c| c == '1'))
    }
    fn demodulate_iter(it: &mut impl Iterator<Item = bool>) -> Expr {
        let t0 = it.next().unwrap();
        let t1 = it.next().unwrap();

        match (t0, t1) {
            (false, false) => Expr::nil(),
            (true, true) => {
                let x = Expr::demodulate_iter(it);
                let y = Expr::demodulate_iter(it);
                Expr::cons(x.into(), y.into())
            }
            (_, y) => {
                let mut t = 0;
                while it.next().unwrap() {
                    t += 1;
                }
                let mut v = 0;
                for i in (0..4 * t).rev() {
                    v |= (if it.next().unwrap() { 1 } else { 0 }) << i;
                }
                Num(if y { v } else { -v })
            }
        }
    }

    pub fn reduce(&self, env: &Env) -> Expr {
        let x = self.eval(env).expr;
        // eprintln!("result before reduce: {}", x);
        match x {
            Op(s, v) => Op(
                s,
                v.iter()
                    .map(|e| e.expr.borrow().reduce(env).into())
                    .collect(),
            ),
            Mod(_) | Num(_) | Image(_) => x,
            _ => panic!("unexpected expr after eval: {}", x),
        }
    }

    pub fn eval(&self, env: &Env) -> EvalResult {
        self.eval2(env, false)
    }

    fn eval2(&self, env: &Env, dump: bool) -> EvalResult {
        if dump {
            *INDENT.lock().unwrap() += 1;
            defer!({
                *INDENT.lock().unwrap() -= 1;
            });
            let i = *INDENT.lock().unwrap();
            // eprintln!(
            //     "{}eval: {}",
            //     std::iter::repeat(" ").take((i - 1)).collect::<String>(),
            //     self
            // );
        }
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
                e.eval2(env, dump)
            }),
            Var(name) => env.get(name).unwrap().eval2(env, dump),
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
                ("eq", [x, y]) => x.eval(env).then(|x| {
                    y.eval(env)
                        .then(|y| EvalResult::new(bb(x.num() == y.num()), true))
                }),
                ("lt", [x, y]) => x.eval(env).then(|x| {
                    y.eval(env)
                        .then(|y| EvalResult::new(bb(x.num() < y.num()), true))
                }),
                ("neg", [x]) => x.eval(env).then(|x| EvalResult::new(Num(-x.num()), true)),

                ("s", [x0, x1, x2]) => {
                    //
                    ap(&ap(x0, x2).into(), &ap(x1, x2).into()).eval2(env, dump)
                }
                ("c", [x0, x1, x2]) => {
                    //
                    ap(&ap(x0, x2).into(), x1).eval2(env, dump)
                }
                ("b", [x0, x1, x2]) => {
                    //
                    ap(x0, &ap(x1, x2).into()).eval2(env, dump)
                }

                ("i", [x0]) => x0.eval2(env, dump),

                ("f", [x0, x1]) => x1.eval2(env, dump),
                ("t", [x0, x1]) => x0.eval2(env, dump),

                (s, [x0, x1, x2]) if s == "cons" || s == "vec" => {
                    ap(&ap(x2, x0).into(), x1).eval2(env, dump)
                }

                ("car", [x2]) => ap(x2, &t.into()).eval2(env, dump),
                ("cdr", [x2]) => ap(x2, &f.into()).eval2(env, dump),

                ("nil", [x0]) => t.eval2(env, dump),
                ("isnil", [x0]) => x0.eval(env).then(|x0| match x0 {
                    Op(s, v) if s == "nil" && v.len() == 0 => t.eval2(env, dump),
                    Op(s, v) if s == "cons" && v.len() == 2 => f.eval2(env, dump),
                    _ => panic!("unexpected x0: {:?}", x0),
                }),
                ("if0", &[x0, x1, x2]) => x0.eval(env).then(|x0| match x0 {
                    Num(0) => x1.eval2(env, dump),
                    Num(1) => x2.eval2(env, dump),
                    _ => {
                        panic!("invalid if0 arg {:?}", x0);
                    }
                }),
                ("send", &[x0]) => {
                    let req = x0.expr.borrow().clone().modulate(env);

                    let url = format!(
                        "https://icfpc2020-api.testkontur.ru/aliens/send?apiKey={}",
                        API_KEY.lock().unwrap()
                    );
                    // eprintln!("running send for expr: {}", self);
                    let res = request(&url, req);
                    // eprintln!("got body: {}", res);
                    EvalResult {
                        expr: Expr::demodulate(&res).eval(env).expr,
                        cacheable: true, // FIXME
                    }
                }
                ("mod", &[x0]) => panic!(),
                ("dem", &[x0]) => panic!(),
                ("modem", &[x0]) => {
                    // EvalResult::new(x0.expr.borrow().clone(), true)
                    Expr::demodulate(&x0.expr.borrow().clone().modulate(env)).eval(env)
                }
                ("draw", &[x0]) => {
                    // eprintln!("drawing: {}", x0.expr.borrow().clone());
                    x0.eval(env).then(|lst| {
                        let mut dummy = EvalResult::new(Num(0), true);
                        let mut img = std::collections::BTreeSet::new();

                        for x in lst.must_list(env) {
                            // eprintln!("drawing point: {}", x);
                            let (i, j) = x.must_point(env);
                            img.insert((i, j));
                        }

                        EvalResult::new(Image(img), dummy.cacheable)
                    })
                }
                (s, v) => {
                    if v.len() >= 3 {
                        panic!();
                    }
                    EvalResult::new(self.clone(), true)
                }
            },
            Func(args, body, v) if args.len() == v.len() => {
                let mut env2 = im::hashmap! {};
                for (a, x) in args.iter().zip(v.iter()) {
                    env2.insert(a.clone(), x.expr.borrow().clone());
                }
                replace_vars_once(&env2, body.as_ref().clone()).eval(&env)
            }
            // Func(_, _, _) => panic!("hoge: FIXME: revert"),
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
    fn then(self, op: impl FnOnce(Expr) -> Self) -> Self {
        let EvalResult { expr, cacheable } = op(self.expr);
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
        ("multipledraw", "x0", "ap ap ap isnil x0 nil ap ap cons ap draw ap car x0 ap multipledraw ap cdr x0"),
        ("f38" ,"x2 x0","ap ap ap if0 ap car x0 ( ap modem ap car ap cdr x0 , ap multipledraw ap car ap cdr ap cdr x0 ) ap ap ap interact x2 ap modem ap car ap cdr x0 ap send ap car ap cdr ap cdr x0"),
        ("interact", "x2 x4 x3", "ap ap f38 x2 ap ap x2 x4 x3"),
        ("inc", "", "ap add 1"),
        ("dec", "", "ap add -1"),
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
        self.eval2(env, false)
    }
    fn eval2(&self, env: &Env, dump: bool) -> EvalResult {
        let EvalResult { expr, cacheable } = self.expr.borrow().clone().eval2(env, dump);
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
                    res = format!("ap {} {}", res, e.expr.borrow());
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
        | "mod" | "dem" | "send" | "modem" | "vec" | "draw" | "if0" | "mul" | "neg" | "nil"
        | "s" | "t" => {
            if s == "vec" {
                Op("cons".into(), vec![]).into()
            } else {
                Op(s.into(), vec![]).into()
            }
        }
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
pub fn galaxy_wasm(state: String, x: i32, y: i32, api_key: String) -> GalaxyResult {
    console_error_panic_hook::set_once();
    *API_KEY.lock().unwrap() = api_key.clone();

    let (state, images) = galaxy(state, (x as _, y as _));
    GalaxyResult { state, images }
}

pub fn galaxy(state: String, vec: (i64, i64)) -> (String, Vec<Vec<(i64, i64)>>) {
    let env = &default_env();

    let input = format!(
        "ap ap ap interact galaxy {} ap ap vec {} {}",
        state, vec.0, vec.1
    );
    let expr = parse_string(env, &input);

    // eprintln!("evaluating: {}", expr);
    let expr = expr.eval(env).expr;
    // eprintln!("expr after eval: {}", expr);

    let v = expr.must_list(env);
    assert_eq!(v.len(), 2);
    // eprintln!("v[0] after eval: {}", v[0]);
    // eprintln!("v[1] after eval: {}", v[1]);

    let next_state = format!("{}", Expr::demodulate(&v[0].modulate(env)));
    // eprintln!("next_state: {}", next_state);

    let images = v[1].must_list(env);
    // eprintln!("images computed: {}", images.len());

    let mut img_cnt = 0;
    let images: Vec<_> = images
        .into_iter()
        .map(|e| {
            img_cnt += 1;
            let img = e.must_image(env);
            // eprintln!("image computed: {:?}", img);
            img
        })
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
            ("ap send ( 0 )", "( 1 , 0 )"),
            ("ap ap ap s add inc 1", "3"),
            ("ap ap ap s mul ap add 1 6", "42"),
            ("ap ap ap c add 1 2", "3"),
            ("ap ap ap b inc dec 2", "2"),
            // ("f", "ap s t"),
            ("ap i ap add 1", "ap add 1"),
            ("ap ap div 4 2", "2"),
            ("ap ap div 4 3", "1"),
            ("ap ap div 4 4", "1"),
            ("ap ap div 4 5", "0"),
            ("ap ap div 5 2", "2"),
            ("ap ap div 6 -2", "-3"),
            ("ap ap div 5 -3", "-1"),
            ("ap ap div -5 3", "-1"),
            ("ap ap div -5 -3", "1"),
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

    #[test]
    fn test_to_image() {
        let expr = "ap ap :1131 ap ap ap c :1126 ap ap c :1172 ap ap :1162 0 0 ap ap ap c :1131 ap i ap ap ap ap ap ap ap ap ap c ap ap b b ap ap b b ap ap b b ap ap b b ap ap b b ap ap b c ap ap c ap ap b c ap ap s ap ap b s ap ap b ap b ap ap s i i lt eq nil ap ap b ap b ap c ap ap b b ap ap b c ap ap b ap b :1131 ap ap c ap ap b b ap :1196 1 ap add -1 ap ap c ap ap b b ap ap b b ap ap b :1183 :1214 ap ap b :1162 ap add -3 1 ap ap ap c :1141 7 ap ap cons 2 ap ap ap ap ap b c ap ap b ap c ap ap c ap ap b b b ap ap s ap ap b s ap ap b ap b b ap ap b ap b s ap ap c ap ap b b ap ap b b ap eq 0 :1115 ap ap b ap b ap c ap ap b b cons ap ap b c ap ap b ap c ap ap c ap ap b b b ap ap s ap ap b s ap ap b ap b b ap ap b ap b s ap ap c ap ap b b ap ap b b ap eq 0 :1115 ap ap b ap b ap c ap ap b b cons ap ap b c ap ap b ap c ap ap c ap ap b b b ap ap s ap ap b s ap ap b ap b b ap ap b ap b s ap ap c ap ap b b ap ap b b ap eq 0 :1115 ap ap b ap b ap c ap ap b b cons ap ap b c ap ap b ap c ap ap c ap ap b b b ap ap s ap ap b s ap ap b ap b b ap ap b ap b s ap ap c ap ap b b ap ap b b ap eq 0 :1115 ap ap b ap b ap c ap ap b b cons ap ap b c ap ap b ap c ap ap c ap ap b b b ap ap s ap ap b s ap ap b ap b b ap ap b ap b s ap ap c ap ap b b ap ap b b ap eq 0 :1115 ap ap b ap b ap c ap ap b b cons ap ap b c ap ap b ap c ap ap c ap ap b b b ap ap s ap ap b s ap ap b ap b b ap ap b ap b s ap ap c ap ap b b ap ap b b ap eq 0 :1115 ap ap b ap b ap c ap ap b b cons ap ap b c ap ap b ap c ap ap c ap ap b b b ap ap s ap ap b s ap ap b ap b b ap ap b ap b s ap ap c ap ap b b ap ap b b ap eq 0 cons ap ap b ap b ap c ap ap b b :1115 ap ap b c ap ap b ap c :1144 ap add -1 ap add -1 ap add -1 ap add -1 ap add -1 ap add -1 ap add -1 ap add -1 7 ap ap ap c ap ap c b ap ap s ap ap b c ap ap b ap b b ap eq 0 ap ap b ap c ap ap c b ap ap s ap ap b c ap ap b ap b b ap eq 0 ap ap b ap c :1141 ap add -1 ap add -1 1 ap send ap car ap ap cons ap ap cons 0 nil nil ap ap cons 0 ap ap cons nil ap ap cons nil ap ap cons nil ap ap cons nil ap ap cons nil ap ap cons 0 nil ap ap ap b ap mul ap neg 18 ap ap c add 1 ap neg 5 48 51 ap ap ap c add 3 ap ap ap b ap mul ap neg 18 ap ap c add 1 ap neg 5 ap :1128 :1247 ap ap ap c :1131 ap i ap ap ap ap ap ap b ap b ap b ap b :1134 ap ap c ap ap b b ap ap b b ap ap b c ap ap b ap b :1126 ap ap b ap b ap :1135 :1247 ap ap b ap b ap c ap ap b ap c ap ap s ap ap b s ap ap b ap b ap ap s i i lt eq ap ap c :1141 0 ap ap b add neg ap ap c ap ap b b ap ap b s ap ap b ap b :1183 ap ap b ap b :1214 ap ap c ap ap b b ap ap b add neg ap ap c :1141 5 ap c ap ap b :1162 ap ap b ap add -3 ap ap b :1175 ap ap b :1266 ap ap c :1141 0 1 ap ap ap c :1141 7 ap ap cons 2 ap ap ap ap ap b c ap ap b ap c ap ap c ap ap b b b ap ap s ap ap b s ap ap b ap b b ap ap b ap b s ap ap c ap ap b b ap ap b b ap eq 0 :1115 ap ap b ap b ap c ap ap b b cons ap ap b c ap ap b ap c ap ap c ap ap b b b ap ap s ap ap b s ap ap b ap b b ap ap b ap b s ap ap c ap ap b b ap ap b b ap eq 0 :1115 ap ap b ap b ap c ap ap b b cons ap ap b c ap ap b ap c ap ap c ap ap b b b ap ap s ap ap b s ap ap b ap b b ap ap b ap b s ap ap c ap ap b b ap ap b b ap eq 0 :1115 ap ap b ap b ap c ap ap b b cons ap ap b c ap ap b ap c ap ap c ap ap b b b ap ap s ap ap b s ap ap b ap b b ap ap b ap b s ap ap c ap ap b b ap ap b b ap eq 0 :1115 ap ap b ap b ap c ap ap b b cons ap ap b c ap ap b ap c ap ap c ap ap b b b ap ap s ap ap b s ap ap b ap b b ap ap b ap b s ap ap c ap ap b b ap ap b b ap eq 0 :1115 ap ap b ap b ap c ap ap b b cons ap ap b c ap ap b ap c ap ap c ap ap b b b ap ap s ap ap b s ap ap b ap b b ap ap b ap b s ap ap c ap ap b b ap ap b b ap eq 0 :1115 ap ap b ap b ap c ap ap b b cons ap ap b c ap ap b ap c ap ap c ap ap b b b ap ap s ap ap b s ap ap b ap b b ap ap b ap b s ap ap c ap ap b b ap ap b b ap eq 0 cons ap ap b ap b ap c ap ap b b :1115 ap ap b c ap ap b ap c :1144 ap add -1 ap add -1 ap add -1 ap add -1 ap add -1 ap add -1 ap add -1 ap add -1 7 ap ap ap c ap ap c b ap ap s ap ap b c ap ap b ap b b ap eq 0 ap ap b ap c ap ap c b ap ap s ap ap b c ap ap b ap b b ap eq 0 ap ap b ap c :1141 ap add -1 ap add -1 1 ap send ap car ap ap cons ap ap cons 0 nil nil ap ap cons 0 ap ap cons nil ap ap cons nil ap ap cons nil ap ap cons nil ap ap cons nil ap ap cons 0 nil 51 ap :1128 :1247 ap ap ap c :1126 ap ap c :1172 ap ap :1162 0 51 ap ap ap c :1131 ap i ap ap ap ap c ap ap b c ap ap c ap ap b b ap ap c lt 0 ap ap c ap ap b cons ap ap b ap :1162 0 ap ap c add 1 nil nil 0 1 ap ap ap c :1126 ap ap b ap :1162 0 ap add 1 ap ap ap b :1138 ap add -1 1 ap ap :1131 ap ap :1131 ap ap ap c :1183 ap ap ap c ap ap c :1173 ap neg 3 ap neg 12 ap ap cons -108 0 ap :1264 ap ap ap c :1141 2 ap car ap ap cons ap ap cons 5 ap ap cons 270608505102339400 ap ap cons 5 ap ap cons 8 ap ap cons 0 ap ap cons ap neg 71253615015 ap ap cons ap s t nil ap ap ap ap ap b c ap c :1133 nil ap ap s ap ap b c ap ap b ap s ap ap b ap eq 270608505102339400 ap ap c ap ap c b ap ap s ap ap b c ap ap b ap b b ap eq 0 ap ap b ap c ap ap c b ap ap s ap ap b c ap ap b ap b b ap eq 0 ap ap b ap c :1141 ap add -1 ap add -1 1 ap c cons i nil ap ap ap c :1183 ap ap ap c ap ap c :1173 ap neg 3 6 ap ap cons -108 0 ap :1264 ap ap ap c :1141 3 ap car ap ap cons ap ap cons 5 ap ap cons 270608505102339400 ap ap cons 5 ap ap cons 8 ap ap cons 0 ap ap cons ap neg 71253615015 ap ap cons ap s t nil ap ap ap ap ap b c ap c :1133 nil ap ap s ap ap b c ap ap b ap s ap ap b ap eq 270608505102339400 ap ap c ap ap c b ap ap s ap ap b c ap ap b ap b b ap eq 0 ap ap b ap c ap ap c b ap ap s ap ap b c ap ap b ap b b ap eq 0 ap ap b ap c :1141 ap add -1 ap add -1 1 ap c cons i nil ap ap ap c :1183 ap ap cons -108 0 ap ap ap c ap :1195 ap ap add -7 ap neg 5 7 ap ap ap c ap ap c i 9 ap neg 10 ap ap ap b ap eq 0 ap ap c :1141 4 ap ap cons 5 ap ap cons 270608505102339400 ap ap cons 5 ap ap cons 8 ap ap cons 0 ap ap cons ap neg 71253615015 ap ap cons ap s t nil";
        let env = default_env();
        let expr = parse_string(&env, &expr);
        let post_eval = expr.eval2(&env, true);
        eprintln!("post_eval: {}", post_eval.expr);
    }

    #[test]
    fn test_demod() {
        let env = default_env();

        for tc in [
            (
                "110110000111011111100001001111110100110000",
                "( 1 , 81740 )",
            ),
            ("010", "0"),
            ("00", "nil"),
            ("1101000", "( 0 )"),
            ("01100001", "1"),
            ("10100001", "-1"),
        ]
        .iter()
        {
            let e1 = parse_string(&env, tc.1);
            eprintln!("e1: {}", e1);
            let e1 = e1.reduce(&env);
            let bin = e1.modulate(&env);

            assert_eq!(tc.0, bin);
        }
    }

    #[test]
    fn test_modulate() {
        let env = default_env();

        for tc in [
            (
                "110110000111011111100001001111110100110000",
                "( 1 , 81740 )",
            ),
            ("010", "0"),
            ("00", "nil"),
            ("1101000", "( 0 )"),
            ("01100001", "1"),
            ("10100001", "-1"),
        ]
        .iter()
        {
            let e1 = Expr::demodulate(tc.0);
            let e2 = parse_string(&env, tc.1);

            eprintln!("e1: {}", e1);
            eprintln!("e2: {}", e2);

            assert_eq!(e1, e2);
        }
    }
}

#[cfg(target_os = "linux")]
fn request(url: &str, req: String) -> String {
    *CNT.lock().unwrap() += 1;

    if *CNT.lock().unwrap() > 10 {
        panic!("too much request")
    }

    let client = reqwest::blocking::Client::new();
    eprintln!("sending: {} {}", url, req);

    if req == "1101000" {
        // (   1       ,  ( 0 , nil ) )
        // 11 01100001   11 010  00
        return "11011000011101000".into();
    }
    panic!("unsupported request: {}", req);
    // client.post(url).body(req).send().unwrap().text().unwrap()
}

#[wasm_bindgen(module = "/define.js")]
#[cfg(target_arch = "wasm32")]
extern "C" {
    fn name() -> String;

    fn request(url: &str, req: String) -> String;
}
