build:
	wasm-pack build --release; \
	( cd www; npx webpack )

run: build
	cd www; \
	npm start

init:
	npm init wasm-app www
	npm install
