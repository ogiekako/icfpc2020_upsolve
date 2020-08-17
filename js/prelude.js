class Lazy {
    // f is a function that returns a value. Lazy objects are created by ap.
    constructor(f) {
        this.f = f;
        this.memo = undefined;
    }
    val() {
        if (this.memo === undefined) {
            this.memo = this.f();
        }
        return this.memo;
    }
}

// TODO: Can we avoid recursive eval? Currently without recursive eval, pwr2 doesn't work.
const eval = (x) => (x instanceof Lazy) ? eval(x.val()) : x

// ap() is the only way to create a Lazy value.
// When a result of this function is eval()ed, it always returns a non-Lazy value.
const ap = (f, x) => new Lazy(() => eval(f)(x));

// Values passed to functions can be Lazy, and must be eval()ed before actual use.
const add = (x) => (y) => eval(x) + eval(y);
const inc = ap(add, 1n);

const mul = (x) => (y) => eval(x) * eval(y);
const div = (x) => (y) => eval(x) / eval(y);
const eq = (x) => (y) => eval(x) == eval(y) ? t : f;
const lt = (x) => (y) => eval(x) <= eval(y) ? t : f;
const neg = (x) => -eval(x);

const s = (x) => (y) => (z) => ap(ap(x, z), ap(y, z));
const c = (x) => (y) => (z) => ap(ap(x, z), y);
const b = (x) => (y) => (z) => ap(x, ap(y, z));
const i = (x) => x;
const f = (x) => (y) => y;
const t = (x) => (y) => x;

const cons = (x) => (y) => (z) => ap(ap(z, x), y);
const vec = cons;

const car = (x) => ap(x, t);
const cdr = (x) => ap(x, f);

const nil = (x) => t;
const isnil = (x) => eval(x) == nil ? t : f;

const to_string = (x) =>
    x == nil ? "nil" :
        typeof x == "bigint" ? "" + x :
            "ap ap cons " + to_string(eval(car(x))) + " " + to_string(eval(cdr(x)));

function test() {

    const pwr2 = new Lazy(() => ap(
        ap(
            s,
            ap(
                ap(
                    c,
                    ap(eq, 0n)
                ),
                1n
            )
        ),
        ap(
            ap(
                b,
                ap(
                    mul,
                    2n
                )
            ),
            ap(
                ap(
                    b,
                    pwr2,
                ),
                ap(
                    add,
                    -1n
                )
            )
        )
    ));

    // S (C =0 1) (B *2 (B pwr2 -1))

    //   S (C =0 1) (B *2 (B pwr2 -1)) 0
    // = (C =0 1 0) (B *2 (B pwr2 -1) 0)
    // = (t 1) (...)
    // = 1
    //
    //   S (C =0 1) (B *2 (B pwr2 -1)) 1
    // = (C =0 1 1) (B *2 (B pwr2 -1) 1)
    // = B *2 (B pwr2 -1) 1
    // = *2 (B pwr2 -1 1)
    // = *2 (pwr2 0)

    console.log(eval(ap(pwr2, 0n))); // 1
    console.log(eval(ap(pwr2, 3n))); // 8

    const myif = (x) => (y) => (z) => eval(x) ? y : z;
    let n = ap(inc, 1n);
    console.log(eval(n));

    n = ap(ap(ap(myif, true), 1), undefined)
    console.log(eval(n));

    console.log(eval(ap(i, 42n)));

    n = ap(ap(ap(myif, true), 1n), ap(ap(ap(s, i), i), ap(ap(s, i), i)))
    console.log(eval(n));

    n = ap(inc, ap(inc, 1n));
    console.log(eval(n));

    n = ap(ap(ap(s, add), inc), 1n)
    console.log(eval(n)); // (add 1) (inc 1) = 3


    n = ap(ap(ap(s, add), inc), 1n)
    console.log(to_string(eval(n))); // (add 1) (inc 1) = 3

    n = ap(ap(cons, 1), 2)
    console.log(to_string(eval(n))); // (1.2)
}

if (process.argv[2] == "-test") {
    test()
}
