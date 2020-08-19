const path = require('path');
const CopyPlugin = require("copy-webpack-plugin");
const WasmPackPlugin = require("@wasm-tool/wasm-pack-plugin");

distDir = path.resolve(__dirname, "dist")

module.exports = {
  resolve: {
    extensions: ['.ts', '.js', '.wasm']
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        loader: 'ts-loader',
        // TODO: check the meaning of this flag.
        options: {
          transpileOnly: true,
        }
      }
    ]
  },
  mode: "development",
  entry: "./index.ts",
  output: {
    path: distDir,
    filename: "index.js",
  },
  plugins: [
    new CopyPlugin(
      [
        { from: 'index.html', to: 'index.html' }
      ]
    ),
    // TODO: this plugin creates www/pkg meaniglessly. Check why.
    new WasmPackPlugin({
      crateDirectory: path.resolve(__dirname, ".."),
      outName: "app",
    })
  ],
  devServer: {
    contentBase: distDir,
    inline: false,
    disableHostCheck: true
  }
};
