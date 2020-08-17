// common provides common type fo galaxy.
// TODO: rename this file to galaxy.rs.

extern crate wasm_bindgen;
use anyhow::*;
use std::{fmt::Formatter, str::FromStr};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

#[wasm_bindgen]
pub struct InteractResult {
    pub(crate) state: String,
    pub(crate) images: Vec<Vec<(i64, i64)>>,
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

pub trait Evaluator {
    fn evaluate(&self, expr: &str) -> Node;
    // add definition in the form of "f = ap ap ...".
    fn add_def(&mut self, s: &str);
}

#[derive(Eq, PartialEq, Debug)]
pub enum Node {
    Cons(Box<Node>, Box<Node>),
    Nil,
    Num(i64),
}

impl FromStr for Node {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Node::parse(&mut s.split(" "))
    }
}

impl Node {
    fn parse<'a>(i: &mut impl Iterator<Item = &'a str>) -> Result<Self> {
        let mut nxt = || i.next().ok_or(anyhow::anyhow!("iterator exhausted"));
        Ok(match nxt()? {
            "nil" => Node::Nil,
            "ap" => {
                nxt()?; // ap
                nxt()?; // cons
                Node::Cons(Node::parse(i)?.into(), Node::parse(i)?.into())
            }
            s => Node::Num(s.parse()?),
        })
    }

    fn must_list_rev(self) -> Vec<Node> {
        match self {
            Node::Nil => vec![],
            Node::Cons(x, y) => {
                let mut res = y.must_list_rev();
                res.push(*x);
                res
            }
            _ => panic!("not list"),
        }
    }
    fn must_list(self) -> Vec<Node> {
        self.must_list_rev().into_iter().rev().collect()
    }
    fn must_num(&self) -> i64 {
        match self {
            Node::Num(i) => *i,
            _ => panic!("not num"),
        }
    }
    fn must_point(&self) -> (i64, i64) {
        match self {
            Node::Cons(x, y) => (x.must_num(), y.must_num()),
            _ => panic!("not cons"),
        }
    }

    fn modulate(&self) -> String {
        match self {
            Node::Nil => "00".into(),
            Node::Cons(x, y) => "11".to_string() + &x.modulate() + &y.modulate(),
            Node::Num(n) => {
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
        }
    }
    fn demodulate(s: &str) -> Node {
        Node::demodulate_iter(&mut s.chars().map(|c| c == '1'))
    }
    fn demodulate_iter(it: &mut impl Iterator<Item = bool>) -> Node {
        let t0 = it.next().unwrap();
        let t1 = it.next().unwrap();

        match (t0, t1) {
            (false, false) => Node::Nil,
            (true, true) => Node::Cons(
                Node::demodulate_iter(it).into(),
                Node::demodulate_iter(it).into(),
            ),
            (_, pos) => {
                let mut t = 0;
                while it.next().unwrap() {
                    t += 1;
                }
                let mut v = 0;
                for i in (0..4 * t).rev() {
                    v |= (if it.next().unwrap() { 1 } else { 0 }) << i;
                }
                Node::Num(if pos { v } else { -v })
            }
        }
    }
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Node::Nil => write!(f, "nil"),
            Node::Cons(x, y) => write!(f, "ap ap cons {} {}", x, y),
            Node::Num(i) => write!(f, "{}", i),
        }
    }
}

pub struct G {
    evaluator: Box<dyn Evaluator>,
}

impl G {
    pub fn new(evaluator: Box<dyn Evaluator>) -> G {
        G { evaluator }
    }
    pub fn galaxy(&self, state: String, x: i32, y: i32, api_key: &str) -> InteractResult {
        self.interact("galaxy", state, x, y, api_key)
    }

    pub fn interact(
        &self,
        protocol: &str,
        mut state: String,
        x: i32,
        y: i32,
        api_key: &str,
    ) -> InteractResult {
        let mut vector = format!("ap ap vec {} {}", x, y);
        loop {
            let input = format!("ap ap {} {} {}", protocol, state, vector);
            let (flag, new_state, data) = {
                let n = self.evaluator.evaluate(&input);
                let mut v = n.must_list();
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
                            .map(|l| {
                                l.must_list()
                                    .into_iter()
                                    .map(|v| v.must_point())
                                    .collect::<Vec<_>>()
                            })
                            .map(|mut l| {
                                l.sort();
                                l
                            })
                            .collect(),
                    }
                }
                1 => {
                    let next_data = send(&data, api_key);
                    vector = format!("{}", next_data);
                }
                _ => panic!("unexpected flag: {}", flag),
            }
        }
    }
}

pub fn send_url(api_key: &str) -> String {
    format!("https://api.pegovka.space/aliens/send?apiKey={}", api_key)
}

fn send(req: &Node, api_key: &str) -> Node {
    let req = req.modulate();
    Node::demodulate(&request(dbg!(&send_url(api_key)), req))
}

#[cfg(target_os = "linux")]
pub fn request(url: &str, req: String) -> String {
    let client = reqwest::blocking::Client::new();
    dbg!(client
        .post(url)
        .body(dbg!(req))
        .send()
        .unwrap()
        .text()
        .unwrap())
}

#[wasm_bindgen(module = "/js/wasm_define.js")]
#[cfg(target_arch = "wasm32")]
extern "C" {
    fn name() -> String;

    pub fn request(url: &str, req: String) -> String;
}
