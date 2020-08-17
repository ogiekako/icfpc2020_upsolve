/// test galaxy.rs and gen_js.rs.
use crate::*;

use crate::common::G;
use crate::galaxy::Eval;
use common::Evaluator;

fn evaluators() -> Vec<Box<dyn Evaluator>> {
    vec![
        Box::new(Eval::new()),
        Box::new(gen_js::GalaxyEvaluator::new()),
    ]
}

#[test]
fn test_statelessdraw() {
    for mut eval in evaluators() {
        eval.add_def(
        "statelessdraw = ap ap c ap ap b b ap ap b ap b ap cons 0 ap ap c ap ap b b cons ap ap c cons nil ap ap c ap ap b cons ap ap c cons nil nil"
    );
        let g = common::G::new(eval);

        let res = g.interact("statelessdraw", "nil".into(), 1, 0, "");
        assert_eq!(res.state, "nil");
        assert_eq!(res.images, vec![vec![(1, 0)]]);
    }
}

#[test]
fn test_statefulldraw() {
    for mut eval in evaluators() {
        eval.add_def(
        ":67108929 = ap ap b ap b ap ap s ap ap b ap b ap cons 0 ap ap c ap ap b b cons ap ap c cons nil ap ap c cons nil ap c cons"
    );
        let g = crate::common::G::new(eval);

        let res = g.interact(":67108929", "nil".into(), 0, 0, "");
        assert_eq!(res.state, "ap ap cons ap ap cons 0 0 nil");
        assert_eq!(res.images, vec![vec![(0, 0)]]);
    }
}

#[test]
fn test_galaxy() {
    for (name, eval) in vec!["pattern match", "gen_js"]
        .into_iter()
        .zip(evaluators())
    {
        let g = common::G::new(eval);
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
            let res = g.galaxy(tc.0.to_string(), (tc.1).0, (tc.1).1, "".into());
            eprintln!("test {}: {}", name, tc.0);
            assert_eq!(res.state, tc.2);
            assert_eq!(res.images, tc.3);
        }
    }
}
