cd "$(dirname "$0")"/www

rm -r dist
npm run build -- --mode production
cp -v dist/* ../../docs/
