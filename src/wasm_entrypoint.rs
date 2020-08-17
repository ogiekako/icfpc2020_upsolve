use crate::*;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct GalaxyEvaluator(common::G);

#[wasm_bindgen]
impl GalaxyEvaluator {
    // pub fn new_gen_js_evaluator() -> Self {
    //     Self(common::G::new(Box::new(gen_js::GalaxyEvaluator::new())))
    // }
    pub fn new_reduce_evaluator() -> Self {
        Self(common::G::new(Box::new(reduce_evaluator::Eval::new())))
    }
    pub fn galaxy(&self, state: String, x: i32, y: i32, api_key: &str) -> common::InteractResult {
        self.0.interact("galaxy", state, x, y, api_key)
    }
}
