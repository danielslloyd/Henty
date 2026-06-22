# Vendored third-party assets

Bundled locally so the GPU server has **no CDN / network dependency** at runtime.

| File             | Library | Version  | Source                                                        | License |
|------------------|---------|----------|---------------------------------------------------------------|---------|
| `marked.min.js`  | marked  | 12.0.2   | https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js      | MIT     |

`marked` renders chunk markdown in the reader (`reader_tab.js`). To upgrade, replace the
file with a new pinned version from the same source and update the version above.
