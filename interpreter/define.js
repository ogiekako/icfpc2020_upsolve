export function name() {
    return 'Rust';
}

export function request(url, req) {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url, false);
    xhr.setRequestHeader('Content-Type', 'text/plain');
    xhr.setRequestHeader('accept', '*/*');
    console.log('sending', url, req);
    xhr.send(req);
    console.log("response", xhr.responseText);
    return xhr.responseText;
}
