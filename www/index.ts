const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;

const keyElem = document.getElementById("api") as HTMLInputElement;
const stateInput = document.getElementById("state") as HTMLInputElement;

const STORAGE_NAME = 'API_KEY';
let apiKey = localStorage.getItem(STORAGE_NAME) || '';

function onApiKeyChanged(ev: Event): void {
    apiKey = keyElem.value;
    localStorage.setItem(STORAGE_NAME, apiKey);
}

export function getApiKey(): string {
    return apiKey;
}

keyElem.value = apiKey;
keyElem.addEventListener('change', onApiKeyChanged);

canvas.width = 800;
canvas.height = 800;

import('../pkg').then(mod => {
    let nextState = "ap ap cons 1 ap ap cons ap ap cons 11 nil ap ap cons 0 ap ap cons nil nil"
    let min = { x: 1000, y: 1000 };
    let max = { x: -1000, y: -1000 };
    let cellSize = 1;

    // let g = mod.GalaxyEvaluator.new_gen_js_evaluator();
    let g = mod.GalaxyEvaluator.new_reduce_evaluator();

    function toCanvas(p: Point): Point {
        return {
            x: (p.x - min.x) * cellSize,
            y: (p.y - min.y) * cellSize
        };
    }
    function fromCanvas(p: Point): Point {
        return {
            x: Math.floor(p.x / cellSize) + min.x,
            y: Math.floor(p.y / cellSize) + min.y,
        };
    }

    function step(input: string, vec: Point) {
        ctx.clearRect(0, 0, 800, 800);

        let api_key = keyElem.value;
        let res = g.galaxy(input, vec.x, vec.y, api_key);
        nextState = res.state();

        stateInput.value = nextState;

        min = { x: 1000, y: 1000 };
        max = { x: -1000, y: -1000 };

        let images = [];
        for (let i = 0; i < res.image_count(); i++) {
            let image = [];
            for (let j = 0; j < res.image(i).count(); j++) {
                const p = res.image(i).point(j);
                image.push({ x: p.x, y: p.y });
                min.x = Math.min(min.x, p.x);
                max.x = Math.max(max.x, p.x + 1);
                min.y = Math.min(min.y, p.y);
                max.y = Math.max(max.y, p.y + 1);
            }
            images.push(image);
        }
        cellSize = Math.floor(800.0 / Math.max(max.x - min.x, max.y - min.y));

        ctx.globalAlpha = 0.5;
        for (let i = 0; i < images.length; i++) {
            let image = images[i];
            let a = i * 360 / images.length;
            ctx.fillStyle = "hsl(" + a + ",100%,50%)";
            for (let p of image) {
                p = toCanvas(p);
                ctx.fillRect(p.x, p.y, cellSize, cellSize);
            }
        }
    }
    step(nextState, { x: -1000, y: -1000 });

    canvas.addEventListener("click", (e) => {
        let p = fromCanvas({ x: e.offsetX, y: e.offsetY });
        console.log("step", nextState, p);
        step(nextState, p)
    })

    stateInput.addEventListener("keyup", (e) => {
        if (e.keyCode != 13) {
            return
        }
        step(stateInput.value.trim(), { x: -1000, y: -1000 });
    })
});

interface Point {
    x: number;
    y: number;
}
