class Lazy {
    // f is a function the returns non-lazy value.
    constructor(f) {
        this.f = f;
        this.memo = undefined;
    }
    val() {
        if (this.memo === undefined) {
            this.memo = this.f();
        }
        if (this.memo instanceof Lazy) {
            throw 'BUG: result was lazy';
        }
        return this.memo;
    }
}

// invariants:
// - 値とは，数字か，1つの lazy object を引数にとり，lazy object を返す関数である．
// - Lazy object は，それに対して，val() を呼ぶと値を返すような object である．
// - primitive は，すべて値である．

const lazy = (f) => new Lazy(f);

const lazy_num = (x) => lazy(() => x);

const add = (x) => lazy(
                (y) => lazy(x.val() + y.val()));
const mul = (x) => lazy((y) => lazy(() => x.val() * y.val()));
const div = (x) => lazy((y) => lazy(() => x.val() / y.val()));
const eq = (x) => lazy((y) => lazy(() => x.val() == y.val()));
const lt = (x) => lazy((y) => lazy(() => x.val() < y.val()));
const neg = (x) => lazy(() => -x.val());

const s = (x) => (y) => (z) => lazy(() => x.val()(y).val()(y.val()(z)).val());

const t = lazy(() => "t");
const nil = lazy(() => "nil");

console.log(add(num(1)).val()(num(2)).val());

// 木構造を直接 JS で表現して，評価可能にしたもの をつくりたい．
// まず，galaxy では，関数はカリー化され，その引数は遅延評価される．たとえば，if 関数で，false であっ
// たときには，then にあたる部分は評価されない．これによって，無限リストを表現することなどが可能になっている．
// 単純にやると，JS では，部分適用であっても，引数を評価してしまうので，まずい．
// というわけで，遅延評価を実現せねばならず，ここでは，関数にわたすものはすべて Lazy object であるという仮定をおくことによって，それを実現している．

// (\x x x)(\x x x) という式を考える．これは，評価されてしまうと，無限ループに陥る．
// 