# Introduction to Machine Learning - Course Website

A simple, clean Jekyll website for the Introduction to Machine Learning course at American University of Beirut, inspired by Stanford CS231n's design with integrated MathJax support for mathematical content.

## Features

- **Clean Design**: Minimalist layout inspired by CS231n.stanford.edu
- **Math Support**: Full MathJax integration for LaTeX mathematical notation
- **Responsive**: Mobile-friendly design
- **Course Schedule**: Clean table layout for lectures, assignments, and deadlines
- **Resource Collection**: Comprehensive learning materials and references
- **Assignment Management**: Detailed project guidelines and submission instructions

## Local Development

### Prerequisites
- Ruby 2.7+ with bundler
- Jekyll 4.0+

### Setup
1. Clone or navigate to the website directory
2. Install dependencies:
   ```bash
   bundle install
   ```
3. Start the development server:
   ```bash
   bundle exec jekyll serve
   ```
4. Visit `http://localhost:4000` in your browser

### Alternative Port
If port 4000 is in use:
```bash
bundle exec jekyll serve --port 4001
```

## Site Structure

```
website/
├── _config.yml          # Site configuration
├── _layouts/
│   └── default.html     # Main layout template
├── assets/
│   └── css/
│       └── main.css     # Stylesheet
├── index.md             # Homepage
├── schedule.md          # Course schedule
├── assignments.md       # Assignments and projects
├── resources.md         # Learning resources
├── logistics.md         # Course policies
├── Gemfile             # Ruby dependencies
└── README.md           # This file
```

## Content Management

### Adding New Pages
1. Create a new `.md` file in the root directory
2. Add YAML front matter:
   ```yaml
   ---
   layout: default
   title: Page Title
   ---
   ```
3. Add the page to navigation in `_layouts/default.html`

### Math Notation
Use LaTeX syntax with MathJax:
- Inline math: `$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$`
- Display math: `$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$`

### Schedule Updates
Edit `schedule.md` to update:
- Lecture dates and topics
- Assignment due dates
- Reading assignments
- Exam schedules

## Customization

### Colors and Styling
- Edit `assets/css/main.css` for visual customization
- Color scheme follows professional academic standards
- Responsive breakpoints included for mobile devices

### Configuration
- Update `_config.yml` for site metadata
- Modify course information, instructor details, and URLs
- Configure MathJax settings if needed

## Deployment

### GitHub Pages
1. Create a new GitHub repository
2. Push this code to the repository
3. Enable GitHub Pages in repository settings
4. Set source to main branch
5. Site will be available at `https://username.github.io/repository-name`

### Custom Domain
1. Add `CNAME` file with your domain
2. Configure DNS settings with your domain provider
3. Enable HTTPS in GitHub Pages settings

## Course Integration

This website is designed to complement the Introduction to Machine Learning course structure:

- **28 Lectures** over 14 weeks
- **Progressive Complexity**: Linear Methods → Neural Networks → Advanced Topics
- **Hands-on Focus**: Python implementations and real datasets
- **Assessment Integration**: Clear assignment schedules and project guidelines

## Technical Notes

- **Jekyll Version**: 4.3+
- **MathJax**: Version 2.7.7 (reliable, widely compatible)
- **Markdown**: Kramdown with GitHub-flavored markdown support
- **CSS**: Custom responsive design, no external frameworks
- **Performance**: Optimized for fast loading and accessibility

## Contributing

To suggest improvements or report issues:
1. Contact the course instructor
2. Use the course discussion forum
3. Submit pull requests for technical improvements

## License

This website template is available for educational use. Course content remains property of the instructor and American University of Beirut.

---

**Course**: Introduction to Machine Learning  
**Institution**: American University of Beirut  
**Instructor**: Joseph Bakarji  
**Semester**: Spring 2025