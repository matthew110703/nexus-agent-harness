# Nexus

**Nexus** is an all-plugin AI agent harness built on top of the [Cordis](https://github.com/cordiverse/cordis) architecture (originally forked from DeepSeek Harness).

## Run from source

```sh
pnpm install
pnpm run build
pnpm dsh web
```

`pnpm run build` prepares the repository artifacts. `pnpm dsh web` starts the Web UI at `http://127.0.0.1:3080` and opens it in your browser. Pass `--no-open` to run without opening a browser.

## Development

- Start with [docs/development.md](docs/development.md) and [docs/architecture.md](docs/architecture.md).
- Run `pnpm run dev:web` to build, serve, and rebuild client bundles on source edits.
- Run `pnpm run dev:desktop` to launch the Electron desktop app in dev mode.

## License

[MIT](LICENSE)

Third-party dependencies and their licenses are disclosed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
