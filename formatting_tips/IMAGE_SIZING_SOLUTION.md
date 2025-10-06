# Image Sizing Solution for Jekyll Website

## Problem
The Kramdown `{width="50%"}` syntax for image sizing was not working correctly on the Jekyll website. Images were displaying at full width instead of the specified percentage.

## Root Cause
Kramdown's attribute syntax `{width="50%"}` was not being processed correctly by the Jekyll/Kramdown configuration, appearing as literal text instead of HTML attributes.

## Solution Implemented

### 1. Enhanced Image Processing in Sync Script
Updated `sync_materials.py` to detect and convert Kramdown attribute syntax to proper HTML:

**Before:**
```markdown
![Scatter plot of the dataset](p1-data.png){width="50%"}
```

**After:**
```html
<img src="/materials/assignments/p1-data.png" alt="Scatter plot of the dataset" width="50%" />
```

### 2. Regex Pattern Enhancement
Modified the image processing regex to capture attributes:
```python
image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)(?:\{([^}]+)\})?'
```

### 3. Attribute Parsing and HTML Generation
Added logic to:
- Parse Kramdown attributes like `width="50%"`
- Convert them to proper HTML attributes
- Generate clean HTML `<img>` tags with all attributes

## Supported Syntax
The solution now supports:

### Basic Images
```markdown
![Alt text](image.png)
```
→ `<img src="/path/image.png" alt="Alt text" />`

### Images with Width
```markdown
![Alt text](image.png){width="50%"}
```
→ `<img src="/path/image.png" alt="Alt text" width="50%" />`

### Images with Multiple Attributes
```markdown
![Alt text](image.png){width="75%", height="auto", class="centered"}
```
→ `<img src="/path/image.png" alt="Alt text" width="75%" height="auto" class="centered" />`

## Testing Results
✅ **hw4.md images**: Both `p1-data.png` and `nn-image.png` now display at 50% width
✅ **HTML output**: Proper `<img>` tags with `width="50%"` attributes
✅ **Website rendering**: Images display correctly at specified sizes
✅ **MathJax compatibility**: Mathematical expressions still render properly

## Usage
The solution is automatic - no changes needed to your workflow:

1. **Write markdown** with Kramdown syntax:
   ```markdown
   ![Figure 1](figure.png){width="50%"}
   ```

2. **Run sync script** as usual:
   ```bash
   python sync_materials.py
   ```

3. **Images display correctly** on the website at the specified size

## Benefits
- **Consistent sizing**: Images display at intended sizes across all devices
- **Better layout**: Smaller images improve readability and page flow
- **Automatic processing**: No manual HTML editing required
- **Backward compatible**: Standard markdown images still work
- **Flexible**: Supports any HTML attribute, not just width

## Technical Details
- **Regex pattern**: Captures alt text, image path, and optional attributes
- **Attribute parsing**: Handles quoted values and multiple attributes
- **HTML generation**: Creates clean, valid HTML `<img>` tags
- **Path resolution**: Maintains correct website-relative paths
- **Error handling**: Gracefully handles missing images or malformed syntax

The solution ensures that your homework assignments display with properly sized images, improving the overall presentation and readability of your course materials.
