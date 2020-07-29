#![allow(unused)]

extern crate anyhow;
extern crate console_error_panic_hook;

extern crate itertools;
extern crate lazy_static;
extern crate serde;
extern crate typed_arena;
extern crate wasm_bindgen;

#[cfg(target_os = "linux")]
extern crate reqwest;

use lazy_static::lazy_static;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt::{Display, Formatter},
    rc::Rc,
    sync::Mutex,
};
use typed_arena::Arena;
use wasm_bindgen::prelude::*;

lazy_static! {
    pub static ref API_KEY: Mutex<String> = Mutex::new(std::env::var("API_KEY").unwrap_or(String::new()));
    static ref STR_PRIMITIVE: HashMap<&'static str, Primitive> = {
        use Primitive::*;
        let mut m = HashMap::new();
        m.insert("add", Add);
        m.insert("mul", Mul);
        m.insert("div", Div);
        m.insert("eq", Eq);
        m.insert("lt", Lt);
        m.insert("neg", Neg);
        m.insert("s", S);
        m.insert("c", C);
        m.insert("b", B);
        m.insert("i", I);
        m.insert("f", F);
        m.insert("t", T);
        m.insert("cons", Cons);
        m.insert("car", Car);
        m.insert("cdr", Cdr);
        m.insert("nil", Nil);
        m.insert("isnil", Isnil);
        m
    };
    static ref PRIMITIVE_STR: HashMap<Primitive, &'static str> = {
        let mut res = HashMap::new();
        STR_PRIMITIVE.iter().for_each(|(k, v)| {
            res.insert(*v, *k);
        });
        res
    };
}

#[derive(Clone, Eq, PartialEq, Debug)]
enum Expr {
    Ap(CachedExpr, CachedExpr),
    Op(Primitive, Option<CachedExpr>, Option<CachedExpr>, Option<CachedExpr>),
    Num(i64),
    Var(String),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Primitive {
    Add,
    Mul,
    Div,
    Eq,
    Lt,
    Neg,
    S,
    C,
    B,
    I,
    F,
    T,
    Cons,
    Car,
    Cdr,
    Nil,
    Isnil,
}

impl Display for Primitive {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", PRIMITIVE_STR.get(self).unwrap())
    }
}

use Expr::*;

impl Into<CachedExpr> for Expr {
    fn into(self) -> CachedExpr {
        CachedExpr {
            cache: Rc::new(RefCell::new(Cache { expr: self, state: 0 })),
        }
    }
}

impl Expr {
    fn boolean(b: bool) -> Expr {
        if b {
            Expr::op(Primitive::T)
        } else {
            Expr::op(Primitive::F)
        }
    }
    fn op(p: Primitive) -> Expr {
        Op(p, None, None, None)
    }

    fn reduce(self, env: &Env) -> Expr {
        match self {
            Op(p, x, y, z) => Op(
                p,
                x.map(|e| e.reduce(env).into()),
                y.map(|e| e.reduce(env).into()),
                z.map(|e| e.reduce(env).into()),
            ),
            x @ Num(_) => x,
            x => x.eval(env).reduce(env),
        }
    }
    fn eval(self, env: &Env) -> Expr {
        use Primitive::*;

        match self {
            Ap(l, r) => match l.eval(env) {
                Op(name, None, _, _) => Op(name, Some(r), None, None),
                Op(name, x, None, _) => Op(name, x.clone(), Some(r), None),
                Op(name, x, y, None) => Op(name, x.clone(), y.clone(), Some(r)),
                _ => panic!("unexpected lhs: {:?}", l),
            }
            .eval(env),
            Op(B, Some(x), Some(y), Some(z)) => Ap(x, Ap(y, z).into()).eval(env),
            Op(C, Some(x), Some(y), Some(z)) => Ap(Ap(x, z).into(), y).eval(env),
            Op(S, Some(x), Some(y), Some(z)) => Ap(Ap(x, z.clone()).into(), Ap(y, z).into()).eval(env),
            Op(Cons, Some(x), Some(y), Some(z)) => Ap(Ap(z, x).into(), y).eval(env),

            Op(I, Some(x), _, _) => x.eval(env),
            Op(Car, Some(x), _, _) => Ap(x, Expr::boolean(true).into()).eval(env),
            Op(Cdr, Some(x), _, _) => Ap(x, Expr::boolean(false).into()).eval(env),
            Op(Neg, Some(x), _, _) => Num(-x.eval(env).must_num()),
            Op(Nil, Some(_), _, _) => Expr::boolean(true),
            Op(Isnil, Some(x), _, _) => match x.eval(env) {
                Op(Nil, None, _, _) => Expr::boolean(true),
                Op(Cons, Some(_), Some(_), None) => Expr::boolean(false),
                _ => panic!("unexpected x: {:?}", x),
            },
            Op(T, Some(x), Some(_), _) => x.eval(env),
            Op(F, Some(_), Some(y), _) => y.eval(env),

            Op(Add, Some(x), Some(y), _) => Num(x.eval(env).must_num() + y.eval(env).must_num()),
            Op(Mul, Some(x), Some(y), _) => Num(x.eval(env).must_num() * y.eval(env).must_num()),
            Op(Div, Some(x), Some(y), _) => Num(x.eval(env).must_num() / y.eval(env).must_num()),
            Op(Eq, Some(x), Some(y), _) => Expr::boolean(x.eval(env).must_num() == y.eval(env).must_num()),
            Op(Lt, Some(x), Some(y), _) => Expr::boolean(x.eval(env).must_num() < y.eval(env).must_num()),

            Var(name) => env.get(&name).unwrap().clone().eval(env),
            _ => self,
        }
    }

    fn must_num(&self) -> i64 {
        match self {
            Expr::Num(x) => *x,
            _ => panic!("not a num: {}", self),
        }
    }
    fn must_list_rev(&self) -> Vec<Expr> {
        match self {
            Op(Primitive::Nil, None, _, _) => vec![],
            Op(Primitive::Cons, Some(x0), Some(x1), None) => {
                let mut res = x1.expr().must_list_rev();
                res.push(x0.expr());
                res
            }
            _ => panic!("not list"),
        }
    }
    fn must_list(&self) -> Vec<Expr> {
        self.must_list_rev().into_iter().rev().collect()
    }
    fn must_point(&self) -> (i64, i64) {
        match self {
            Op(Primitive::Cons, Some(x), Some(y), None) => (x.expr().must_num(), y.expr().must_num()),
            _ => panic!("not vec"),
        }
    }

    fn cons(hd: CachedExpr, tl: CachedExpr) -> Expr {
        Op(Primitive::Cons, Some(hd), Some(tl), None)
    }
    fn nil() -> Expr {
        Expr::op(Primitive::Nil)
    }

    fn modulate(&self) -> String {
        match self {
            Num(n) => {
                let mut res = String::new();
                let n = if *n >= 0 {
                    res.push_str("01");
                    *n
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
            _ => match self {
                Op(Primitive::Nil, None, _, _) => "00".into(),
                Op(Primitive::Cons, Some(hd), Some(tl), None) => "11".to_owned() + &hd.expr().modulate() + &tl.expr().modulate(),
                _ => panic!("unexpected op {}", self),
            },
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
            (_, pos) => {
                let mut t = 0;
                while it.next().unwrap() {
                    t += 1;
                }
                let mut v = 0;
                for i in (0..4 * t).rev() {
                    v |= (if it.next().unwrap() { 1 } else { 0 }) << i;
                }
                Num(if pos { v } else { -v })
            }
        }
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
struct CachedExpr {
    cache: Rc<RefCell<Cache>>,
}
#[derive(Clone, Eq, PartialEq, Debug)]

struct Cache {
    expr: Expr,
    state: u8, // 1: cached, 2: reduced
}

impl std::ops::Deref for Cache {
    type Target = Expr;
    fn deref(&self) -> &Self::Target {
        &self.expr
    }
}

type Env = HashMap<String, Expr>;

fn default_env() -> Env {
    let mut env = Env::new();

    for line in include_str!("../galaxy.txt").split("\n") {
        let ss = line.split(" = ").collect::<Vec<_>>();
        let (name, expr) = (ss[0], ss[1]);
        let e = parse_string(&env, expr);
        env.insert(name.to_string(), e);
    }

    env
}

impl CachedExpr {
    fn eval(&self, env: &Env) -> Expr {
        let mut state = self.cache.borrow().state;
        if state == 0 {
            let expr = { self.cache.borrow().expr.clone().eval(env) };
            {
                self.cache.borrow_mut().expr = expr
            };
        }
        self.cache.borrow().expr.clone()
    }
    fn reduce(&self, env: &Env) -> Expr {
        let mut state = self.cache.borrow().state;
        if state < 2 {
            let expr = { self.cache.borrow().expr.clone().reduce(env) };
            {
                self.cache.borrow_mut().expr = expr
            };
        }
        self.cache.borrow().expr.clone()
    }
    fn expr(&self) -> Expr {
        self.cache.borrow().expr.clone()
    }
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Ap(l, r) => write!(f, "ap {} {}", l.expr().to_string(), r.expr().to_string()),
            Expr::Op(s, x, y, z) => {
                let mut res = format!("{}", s);
                for e in [x, y, z].iter() {
                    if let Some(e) = e {
                        res = format!("ap {} {}", res, e.expr().to_string());
                    }
                }
                write!(f, "{}", res)
            }
            Expr::Num(i) => write!(f, "{}", i),
            Expr::Var(s) => write!(f, "{}", s),
        }
    }
}

fn parse_string(env: &Env, expr: &str) -> Expr {
    parse(env, &mut expr.split(" ").map(String::from).into_iter().peekable())
}

fn parse(env: &Env, mut it: &mut std::iter::Peekable<impl std::iter::Iterator<Item = String>>) -> Expr {
    use Expr::*;

    let mut s: &str = &it.next().expect("iterator exhausted");
    if s == "vec" {
        s = "cons";
    }
    match s {
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
        "ap" => Ap(parse(env, &mut it).into(), parse(env, &mut it).into()),
        s => {
            if let Some(p) = STR_PRIMITIVE.get(s) {
                Expr::op(*p).into()
            } else if let Ok(i) = s.parse::<i64>() {
                Num(i)
            } else if env.contains_key(s) || s.chars().next().unwrap() == ':' || s.chars().next().unwrap() == 'x' {
                Var(s.to_string())
            } else {
                panic!("unknown var {}", s);
            }
        }
    }
}

#[wasm_bindgen]
pub struct InteractResult {
    state: String,
    images: Vec<Vec<(i64, i64)>>,
}

#[wasm_bindgen]
impl InteractResult {
    pub fn state(&self) -> String {
        self.state.clone()
    }
    pub fn image_count(&self) -> usize {
        self.images.len()
    }
    pub fn image(&self, i: usize) -> Image {
        Image {
            img: self.images[i].iter().map(|p| Point { x: p.0 as _, y: p.1 as _ }).collect(),
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
pub struct G {
    env: Env,
}

#[wasm_bindgen]
impl G {
    pub fn new() -> G {
        G { env: default_env() }
    }
    pub fn galaxy(&self, mut state: String, x: i32, y: i32, api_key: &str) -> InteractResult {
        self.interact("galaxy", state, x, y, api_key)
    }
    fn interact(&self, protocol: &str, mut state: String, x: i32, y: i32, api_key: &str) -> InteractResult {
        let env = &self.env;
        let mut vector = format!("ap ap vec {} {}", x, y);
        loop {
            let input = format!("ap ap {} {} {}", protocol, state, vector);
            let expr = parse_string(&self.env, &input);
            let (flag, new_state, data) = {
                let e = expr.reduce(&self.env);
                let mut v = e.must_list();
                (v.remove(0), v.remove(0), v.remove(0))
            };

            state = format!("{}", new_state);
            match flag.must_num() {
                0 => {
                    return InteractResult {
                        state,
                        images: data
                            .must_list()
                            .into_iter()
                            .map(|l| l.must_list().into_iter().map(|v| v.must_point()).collect::<Vec<_>>())
                            .map(|mut l| {
                                l.sort();
                                l
                            })
                            .collect(),
                    }
                }
                1 => {
                    let next_data = send(&data, env, api_key);
                    vector = format!("{}", next_data);
                }
                _ => panic!("unexpected flag: {}", flag),
            }
        }
    }
}

fn send_url(api_key: &str) -> String {
    format!("https://icfpc2020-api.testkontur.ru/aliens/send?apiKey={}", api_key)
}

fn send(req: &Expr, env: &Env, api_key: &str) -> Expr {
    let req = req.clone().modulate();
    Expr::demodulate(&request(dbg!(&send_url(api_key)), req))
}

#[cfg(target_os = "linux")]
fn request(url: &str, req: String) -> String {
    let client = reqwest::blocking::Client::new();
    dbg!(client.post(url).body(dbg!(req)).send().unwrap().text().unwrap())
}

#[wasm_bindgen(module = "/define.js")]
#[cfg(target_arch = "wasm32")]
extern "C" {
    fn name() -> String;

    fn request(url: &str, req: String) -> String;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::iter::FromIterator;

    #[test]
    fn test_statelessdraw() {
        let mut g = G::new();
        g.env.insert(
            "statelessdraw".into(),
            parse_string(
                &g.env,
                "ap ap c ap ap b b ap ap b ap b ap cons 0 ap ap c ap ap b b cons ap ap c cons nil ap ap c ap ap b cons ap ap c cons nil nil",
            ),
        );

        let res = g.interact("statelessdraw", "nil".into(), 1, 0, "");
        assert_eq!(res.state, "nil");
        assert_eq!(res.images, vec![vec![(1, 0)]]);
    }

    #[test]
    fn test_galaxy() {
        let g = G::new();
        for tc in vec![
            (
                "nil",
                (0, 0),
                "ap ap cons 0 ap ap cons ap ap cons 0 nil ap ap cons 0 ap ap cons nil nil",
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
                        (3, 0),
                    ],
                    vec![(-8, -2), (-7, -3)],
                    vec![],
                ],
            ),
            (
                "ap ap cons 0 ap ap cons ap ap cons 0 nil ap ap cons 0 ap ap cons nil nil",
                (0, 0),
                "ap ap cons 0 ap ap cons ap ap cons 1 nil ap ap cons 0 ap ap cons nil nil",
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
                        (3, 0),
                    ],
                    vec![(-8, -2), (-7, -3), (-7, -2)],
                    vec![],
                ],
            ),
        ] {
            let res = g.galaxy(tc.0.to_string(), (tc.1).0, (tc.1).1, API_KEY.lock().unwrap().as_str());
            assert_eq!(res.state, tc.2);
            assert_eq!(res.images, tc.3);
        }
    }

    #[test]
    fn test() {
        for tc in vec![
            ("ap ap add 1 2", "3"),
            ("ap ap add 1 ap ap add 2 3", "6"),
            ("ap ap t 1 :1", "1"),
            ("ap ap f :1 1", "1"),
            ("( )", "nil"),
            ("( 1 , 2 )", "ap ap cons 1 ap ap cons 2 nil"),
            ("ap car ( 1 , 2 )", "1"),
            ("ap ap ap s mul ap add 1 6", "42"),
            ("ap ap ap c add 1 2", "3"),
            ("ap ap ap c i 1 ap i ap add 1", "2"),
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
    fn test_demod() {
        let env = default_env();

        for tc in [
            ("110110000111011111100001001111110100110000", "( 1 , 81740 )"),
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
            let bin = e1.modulate();

            assert_eq!(tc.0, bin);
        }
    }

    #[test]
    fn test_demodulate() {
        let env = default_env();

        for tc in [
            ("110110000111011111100001001111110100110000", "( 1 , 81740 )"),
            ("010", "0"),
            ("00", "nil"),
            ("1101000", "( 0 )"),
            ("01100001", "1"),
            ("10100001", "-1"),
            (
                "1101100001111111011000011101111111111111111100100100010011000101110101000111101110101110100100110100101001001000000",
                "ap ap cons 1 ap ap cons ap ap cons ap ap cons 1 ap ap cons 5231136092510644553 nil nil nil",
            ),
        ]
        .iter()
        {
            let e1 = Expr::demodulate(tc.0);
            let e2 = parse_string(&env, tc.1).reduce(&env);

            eprintln!("e1: {}", e1);
            eprintln!("e2: {}", e2);
            assert_eq!(e1, e2);
        }
    }
}
