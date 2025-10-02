# Math Rendering Solutions for Jekyll GitHub Pages

## Current Setup
Your website uses **MathJax 2.7.7** with the following configuration:
- Math engine: `mathjax` (in `_config.yml`)
- CDN: `https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_CHTML-full,Safe`
- AMS packages: Now included for `\begin{cases}` support

## Problem: `\begin{cases}` Not Working

The `cases` environment requires the AMS (American Mathematical Society) packages. The issue was that your MathJax configuration wasn't explicitly loading the necessary extensions.

## Solution 1: Updated MathJax Configuration (IMPLEMENTED)

I've updated your `_layouts/default.html` to include:
```javascript
extensions: ["AMSmath.js", "AMSsymbols.js", "noErrors.js", "noUndefined.js"]
```

This should now support:
- `\begin{cases}...\end{cases}`
- `\begin{align}...\end{align}`
- `\begin{matrix}...\end{matrix}`
- `\begin{pmatrix}...\end{pmatrix}`
- And other AMS environments

## Solution 2: Alternative MathJax 3.x (Recommended for New Projects)

If you want to upgrade to MathJax 3.x (faster, more modern):

```html
<!-- Replace the MathJax 2.7.7 script with: -->
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
  }
};
</script>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
```

## Solution 3: KaTeX Alternative

If you prefer KaTeX (faster rendering, smaller bundle):

```html
<!-- Add to <head> -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>

<script>
document.addEventListener("DOMContentLoaded", function() {
    renderMathInElement(document.body, {
        delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false},
            {left: '\\(', right: '\\)', display: false},
            {left: '\\[', right: '\\]', display: true}
        ],
        throwOnError: false
    });
});
</script>
```

## Testing Your Math

Test these examples to verify everything works:

### Cases Environment
```latex
$$
f(x) = 
\begin{cases}
    1, & x \geq 0\\
    0, & x < 0
\end{cases}
$$
```

### Matrix
```latex
$$
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
$$
```

### Align Environment
```latex
$$
\begin{align}
y &= mx + b \\
f(x) &= ax^2 + bx + c
\end{align}
$$
```

## Common Issues and Solutions

### Issue 1: Math not rendering at all
- Check that `mathjax: true` is in `_config.yml`
- Verify the MathJax script is loading (check browser console)
- Ensure math is wrapped in `$...$` or `$$...$$`

### Issue 2: Specific environments not working
- The AMS extensions are now included in your configuration
- For custom environments, add them to the `Macros` section

### Issue 3: Math rendering slowly
- Consider upgrading to MathJax 3.x or switching to KaTeX
- Use `processEnvironments: true` for better performance

### Issue 4: GitHub Pages compatibility
- All solutions work with GitHub Pages
- No server-side processing required
- Math rendering happens client-side

## Performance Comparison

| Solution | Bundle Size | Rendering Speed | Compatibility |
|----------|-------------|-----------------|---------------|
| MathJax 2.7.7 | ~200KB | Medium | Excellent |
| MathJax 3.x | ~100KB | Fast | Excellent |
| KaTeX | ~50KB | Very Fast | Good |

## Recommendation

For your current setup, the updated MathJax 2.7.7 configuration should resolve the `\begin{cases}` issue. If you want better performance, consider upgrading to MathJax 3.x or KaTeX in the future.

## Troubleshooting

1. **Clear browser cache** after making changes
2. **Check browser console** for JavaScript errors
3. **Test locally** with `bundle exec jekyll serve`
4. **Verify on GitHub Pages** after deployment

The updated configuration should now properly render your `\begin{cases}` environment and other AMS math environments.
