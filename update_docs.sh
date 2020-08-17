cd "$(dirname "$0")"/www

rm -r dist &&
npm run build -- --mode production &&
rm ../docs/* &&
cp -v dist/* ../docs/
