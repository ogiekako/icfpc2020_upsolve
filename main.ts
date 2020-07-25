import fs = require('fs');

const input = fs.readFileSync("/dev/stdin", "utf8").split(" ");
const a = +input[0]; // string を number にする hack.
const b = +input[1];

console.log(a + b);
