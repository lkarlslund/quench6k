# Vendored dashboard dependencies

The dashboard loads these files from the same quench6k binary. It makes no CDN requests and needs no
Node.js build step. Versions are pinned; update the files and this record together.

| File | Package | Version | License |
| --- | --- | --- | --- |
| `alpine.min.js` | [alpinejs](https://github.com/alpinejs/alpine) (`dist/cdn.min.js`) | 3.17.5 | MIT (`alpinejs-LICENSE.md`) |
| `echarts.min.js` | [echarts](https://github.com/apache/echarts) (`dist/echarts.min.js`) | 6.1.0 | Apache-2.0 (`echarts-LICENSE`, `echarts-NOTICE`) |

Downloaded from the corresponding npm release tarballs, verified against the registry's SHA-512
integrity values. Alpine's license is from `alpinejs/alpine` tag `v3.17.5` (not included at the root of
its npm tarball). ECharts' license and notice are copied from its npm tarball.
