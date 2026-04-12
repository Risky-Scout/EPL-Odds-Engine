# Portability and Local Overlay

The public repo stays portable.

Keep these **out** of the public repository:

- absolute local machine paths
- `.env.local`
- private API keys
- local caches
- generated research outputs
- personal runtime scripts

Use the separate local overlay package for your Mac-specific path:

```bash
/Users/josephshackelford/woo_models/CyberDuck/EPL Page
```

The local overlay copies machine-specific files into your local clone without polluting GitHub.
