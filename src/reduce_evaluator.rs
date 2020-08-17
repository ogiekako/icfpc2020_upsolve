extern crate anyhow;

extern crate itertools;
extern crate lazy_static;

#[cfg(target_os = "linux")]
extern crate reqwest;

use crate::common::{self, Node};
use lazy_static::lazy_static;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt::{Display, Formatter},
    rc::Rc,
    sync::Mutex,
};

lazy_static! {
    pub static ref API_KEY: Mutex<String> =
        Mutex::new(std::env::var("API_KEY").unwrap_or(String::new()));
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
pub enum Expr {
    Ap(CachedExpr, CachedExpr),
    Op(
        Primitive,
        Option<CachedExpr>,
        Option<CachedExpr>,
        Option<CachedExpr>,
    ),
    Num(i64),
    Var(String),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Primitive {
    Add,   // x y   => x + y
    Mul,   // x y   => x * y
    Div,   // x y   => x / y
    Eq,    // x y   => x == y
    Lt,    // x y   => x <= y
    Neg,   // x     => -x
    S,     // x y z => (x z) (y z)  !
    C,     // x y z => (x z) y
    B,     // x y z => x (y z)
    I,     // x     => x
    F,     // x y   => y  !
    T,     // x y   => x  !
    Cons,  // x y z => (z x) y
    Car,   // x     => x T
    Cdr,   // x     => x F
    Nil,   // x     => T
    Isnil, // x     => x == Nil ? T : F
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
            cache: Rc::new(RefCell::new(Cache {
                expr: self,
                state: 0,
            })),
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
            Op(S, Some(x), Some(y), Some(z)) => {
                Ap(Ap(x, z.clone()).into(), Ap(y, z).into()).eval(env)
            }
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
            Op(Eq, Some(x), Some(y), _) => {
                Expr::boolean(x.eval(env).must_num() == y.eval(env).must_num())
            }
            Op(Lt, Some(x), Some(y), _) => {
                Expr::boolean(x.eval(env).must_num() < y.eval(env).must_num())
            }

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
    fn cons(hd: CachedExpr, tl: CachedExpr) -> Expr {
        Op(Primitive::Cons, Some(hd), Some(tl), None)
    }
    fn nil() -> Expr {
        Expr::op(Primitive::Nil)
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct CachedExpr {
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
        let state = self.cache.borrow().state;
        if state == 0 {
            let expr = { self.cache.borrow().expr.clone().eval(env) };
            {
                self.cache.borrow_mut().expr = expr
            };
        }
        self.cache.borrow().expr.clone()
    }
    fn reduce(&self, env: &Env) -> Expr {
        let state = self.cache.borrow().state;
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

pub(crate) fn parse_string(env: &Env, expr: &str) -> Expr {
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

pub struct Eval {
    pub(crate) env: Env,
}

impl Eval {
    pub fn new() -> Self {
        Eval { env: default_env() }
    }
}

impl crate::common::Evaluator for Eval {
    fn evaluate(&self, expr: &str) -> Node {
        let expr = parse_string(&self.env, expr);
        expr_to_node(expr.reduce(&self.env))
    }
    fn add_def(&mut self, line: &str) {
        let ss = line.split(" = ").collect::<Vec<_>>();
        let (name, expr) = (ss[0], ss[1]);
        let e = parse_string(&self.env, expr);
        self.env.insert(name.to_string(), e);
    }
}

fn expr_to_node(e: Expr) -> Node {
    match e {
        Op(Primitive::Nil, None, _, _) => Node::Nil,
        Op(Primitive::Cons, Some(x0), Some(x1), None) => Node::Cons(
            expr_to_node(x0.expr()).into(),
            expr_to_node(x1.expr()).into(),
        ),
        Num(x) => Node::Num(x),
        _ => panic!("unconvertible to node: {}", e),
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
}
