# LaTeX Cases Environment Fix

## Problem
The `\begin{cases}` environment in LaTeX was not rendering correctly on the Jekyll website. The function definition was showing as `f(x) = {1, x >= 00, x < 0` instead of the proper piecewise function format.

## Root Cause
Kramdown (the markdown processor) was HTML-encoding the LaTeX before MathJax could process it:
- `&` became `&amp;`
- `<` became `&lt;`
- This prevented MathJax from properly parsing the `\begin{cases}` environment

## Solution Implemented

### 1. Enhanced MathJax Configuration
Updated the MathJax configuration in `_layouts/default.html` to include:
- `autoload-all.js` extension for automatic package loading
- Explicit package configuration: `packages: { '[+]': ['base', 'ams', 'noerrors', 'noundefined'] }`
- Better error handling and processing options

### 2. Script Tag Approach
Used `<script type="math/tex; mode=display">` to bypass Kramdown processing:

**Before (problematic):**
```markdown
$$
f(x) = 
\begin{cases}
    1, & x\geq 0\\
    0,  & x < 0
\end{cases}
$$
```

**After (working):**
```html
<script type="math/tex; mode=display">
f(x) = 
\begin{cases}
    1, & x\geq 0\\
    0,  & x < 0
\end{cases}
</script>
```

## Technical Details

### Why Script Tags Work
- `<script type="math/tex">` is a MathJax-specific format
- Kramdown doesn't process content inside script tags
- MathJax directly processes the LaTeX without HTML encoding interference
- `mode=display` ensures proper display math formatting

### MathJax Configuration Updates
```javascript
extensions: [
    "AMSmath.js", 
    "AMSsymbols.js", 
    "noErrors.js", 
    "noUndefined.js",
    "autoload-all.js"  // Added for automatic package loading
],
packages: {
    '[+]': ['base', 'ams', 'noerrors', 'noundefined']  // Explicit package loading
}
```

## Results
 **Cases environment renders correctly**: The piecewise function now displays properly
 **No HTML encoding**: LaTeX symbols are preserved as intended
 **Proper formatting**: Display math is centered and well-formatted
 **MathJax compatibility**: All other math expressions continue to work

## Usage
For complex LaTeX environments like `cases`, `align`, `matrix`, etc., use:

```html
<script type="math/tex; mode=display">
\begin{cases}
    condition1, & \text{if } condition1\\
    condition2, & \text{if } condition2
\end{cases}
</script>
```

For inline math, continue using standard markdown: `$x \geq 0$`

## Alternative Approaches Tried
1. **Display math delimiters** (`$$...$$`): Still got HTML encoded
2. **HTML div wrapper**: Still got HTML encoded  
3. **Kramdown configuration**: Didn't resolve the encoding issue
4. **Script tags**: ✅ **Success!**

The script tag approach is the most reliable solution for complex LaTeX environments in Jekyll/Kramdown setups.
