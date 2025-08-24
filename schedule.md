---
layout: default
title: Schedule
---

# Course Schedule
**Introduction to Machine Learning • Fall 2025**

<style>
.schedule-search {
    margin: 10px 0 20px;
    display: flex;
    gap: 10px;
}

.schedule-search input {
    flex: 1;
    padding: 10px 12px;
    border: 1px solid #e5e5e5;
    border-radius: 6px;
}

.schedule-search small { color: #666; }

.schedule-search .btn {
    padding: 10px 12px;
    border: 1px solid #e5e5e5;
    border-radius: 6px;
    background: #f8f9fa;
    cursor: pointer;
    color: #2c3e50;
}

.schedule-search .btn:hover { background: #eef1f4; }
.schedule-section {
    margin: 20px 0;
    border: 1px solid #e5e5e5;
    border-radius: 8px;
    overflow: hidden;
}

.section-header {
    background: #f8f9fa;
    padding: 15px 20px;
    cursor: pointer;
    border-bottom: 1px solid #e5e5e5;
    user-select: none;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.section-header:hover {
    background: #e9ecef;
}

.section-header h3 {
    margin: 0;
    color: #2c3e50;
}

.toggle-icon {
    font-size: 18px;
    transition: transform 0.3s;
}

.section-content {
    display: none;
}

.section-content.active {
    display: block;
}

.lecture-item {
    border-bottom: 1px solid #f0f0f0;
}

.lecture-item:last-child {
    border-bottom: none;
}

.lecture-header {
    padding: 15px 20px;
    display: flex;
    gap: 20px;
    align-items: flex-start;
    cursor: pointer;
    transition: background-color 0.2s;
}

.lecture-header:hover {
    background: #f8f9fa;
}

.lecture-date {
    min-width: 80px;
    font-weight: 600;
    color: #2c3e50;
}

.lecture-number {
    min-width: 40px;
    background: #3498db;
    color: white;
    padding: 4px 8px;
    border-radius: 4px;
    text-align: center;
    font-size: 12px;
    font-weight: 600;
}

.assignment-number {
    background: #e67e22;
}

.exam-number {
    background: #e74c3c;
}

.hackathon-number {
    background: #27ae60;
}

.lecture-content {
    flex: 1;
}

.lecture-title {
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 5px;
}

.lecture-subtitle {
    color: #666;
    font-size: 14px;
    margin-bottom: 8px;
}

.lecture-description {
    color: #666;
    font-size: 14px;
    margin-bottom: 10px;
}

.lecture-meta {
    display: flex;
    gap: 15px;
    font-size: 12px;
    color: #888;
}

.assignment-info {
    margin-top: 8px;
    margin-bottom: 8px;
}

.assignment-due, .assignment-released {
    display: inline-block;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 8px;
    border-radius: 4px;
    margin-right: 8px;
    margin-bottom: 4px;
}

.assignment-due {
    background: #fff3cd;
    color: #856404;
    border: 1px solid #ffeaa7;
}

.assignment-released {
    background: #d1ecf1;
    color: #0c5460;
    border: 1px solid #bee5eb;
}

.lecture-details {
    padding: 20px;
    background: #fafbfc;
    border-top: 1px solid #e5e5e5;
    display: none;
}

.lecture-details.active {
    display: block;
}

.materials-section {
    margin-bottom: 15px;
}

.lecture-description-expanded {
    color: #666;
    font-size: 14px;
    margin-bottom: 15px;
    padding: 10px 0;
    border-bottom: 1px solid #f0f0f0;
}

.materials-title {
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 10px;
}

.material-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.material-item {
    padding: 3px 0;
    border-bottom: 1px solid #f0f0f0;
}

.material-item:last-child {
    border-bottom: none;
}

.material-link {
    color: #3498db;
    text-decoration: none;
    font-size: 14px;
}

.material-link:hover {
    text-decoration: underline;
}

.material-type {
    background: #f8f9fa;
    color: #666;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 3px;
    margin-left: 8px;
}

.learning-objectives {
    margin-top: 15px;
}

.objectives-title {
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 8px;
}

.objectives-list {
    margin: 0;
    padding-left: 20px;
    color: #555;
}

@media (max-width: 768px) {
    .lecture-header {
        flex-direction: column;
        gap: 10px;
    }
    
    .lecture-date, .lecture-number {
        min-width: auto;
    }
}
</style>

<!-- Group lectures by sections -->
<div class="schedule-search">
  <input type="text" id="scheduleFilter" placeholder="Search by title, subtitle, or material..." aria-label="Filter schedule" />
  <button class="btn" id="expandAllBtn" type="button">Expand All</button>
  <button class="btn" id="collapseAllBtn" type="button">Collapse All</button>
  <small id="scheduleCount"></small>
</div>

<script>
  (function(){
    const input = document.getElementById('scheduleFilter');
    const count = document.getElementById('scheduleCount');
    const expandBtn = document.getElementById('expandAllBtn');
    const collapseBtn = document.getElementById('collapseAllBtn');

    function norm(s){ return (s||'').toLowerCase(); }

    function expandAll(){
      document.querySelectorAll('.section-content').forEach(sc => sc.classList.add('active'));
      document.querySelectorAll('.lecture-details').forEach(ld => ld.classList.add('active'));
      document.querySelectorAll('.section-header .toggle-icon').forEach(icn => icn.textContent = '▼');
    }

    function collapseAll(){
      document.querySelectorAll('.lecture-details').forEach(ld => ld.classList.remove('active'));
      document.querySelectorAll('.section-content').forEach(sc => sc.classList.remove('active'));
      document.querySelectorAll('.section-header .toggle-icon').forEach(icn => icn.textContent = '▶');
    }

    function ensureSectionsOpen(){
      document.querySelectorAll('.section-content').forEach(sc => sc.classList.add('active'));
    }

    function filter(){
      const q = norm(input.value);
      let visible = 0;
      if (q) ensureSectionsOpen();

      document.querySelectorAll('.lecture-item').forEach(item => {
        // Include hidden text using textContent (covers collapsed details)
        const details = item.querySelector('.lecture-details');
        const header = item.querySelector('.lecture-content');
        const hay = norm((header ? header.textContent : '') + ' ' + (details ? details.textContent : ''));
        const show = !q || hay.indexOf(q) !== -1;
        item.style.display = show ? '' : 'none';
        if (details) {
          if (q && show) { details.classList.add('active'); }
          else if (!q) { details.classList.remove('active'); }
        }
        if (show) visible++;
      });
      count.textContent = visible + ' results';
    }

    input.addEventListener('input', filter);
    expandBtn.addEventListener('click', function(){ expandAll(); });
    collapseBtn.addEventListener('click', function(){ collapseAll(); });
    document.addEventListener('DOMContentLoaded', filter);
  })();
</script>
{% if site.data.lectures.size > 0 %}
{% assign sorted_lectures = site.data.lectures | sort: 'lecture_number' %}
{% assign sections = sorted_lectures | group_by: 'section' %}

{% for section in sections %}
<div class="schedule-section">
    <div class="section-header" onclick="toggleSection(this)">
        <h3>{{ section.name | default: "Other Lectures" }}</h3>
        <span class="toggle-icon">▼</span>
    </div>
    <div class="section-content active">
        {% for lecture in section.items %}
        <div class="lecture-item">
            <div class="lecture-header" onclick="toggleLectureDetails(this)">
                <div class="lecture-date">
                    {% if lecture.date %}
                        {% assign d = lecture.date | date: "%Y-%m-%d" %}
                        {% if d == lecture.date %}
                            {{ lecture.date | date: "%b %e" }}
                        {% else %}
                            {{ lecture.date }}
                        {% endif %}
                    {% else %}
                        TBD
                    {% endif %}
                </div>
                <div class="lecture-number">{{ lecture.lecture_number }}</div>
                <div class="lecture-content">
                    <div class="lecture-title">{{ lecture.title }}</div>
                    {% if lecture.subtitle %}
                    <div class="lecture-subtitle">{{ lecture.subtitle }}</div>
                    {% endif %}
                    
                    <!-- Assignment info -->
                    <div class="assignment-info">
                        {% if lecture.assignments and lecture.assignments.due_today %}
                            {% for assignment in lecture.assignments.due_today %}
                                <div class="assignment-due">{{ assignment.name }} Due</div>
                            {% endfor %}
                        {% endif %}
                        {% if lecture.assignments and lecture.assignments.released_today %}
                            {% for assignment in lecture.assignments.released_today %}
                                <div class="assignment-released">{{ assignment.name }} Released</div>
                            {% endfor %}
                        {% endif %}
                    </div>
                    
                    <div class="lecture-meta">
                        {% if lecture.duration %}
                        <span>{{ lecture.duration }} min</span>
                        {% endif %}
                        {% if lecture.type %}
                        <span>{{ lecture.type | capitalize }}</span>
                        {% endif %}
                    </div>
                </div>
            </div>
            
        <div class="lecture-details">
                <!-- Description (moved here from header) -->
                {% if lecture.description %}
                <div class="lecture-description-expanded">{{ lecture.description }}</div>
                {% endif %}
                
                <!-- Materials Section -->
                {% if lecture.materials %}
                    {% for material_category in lecture.materials %}
                        {% assign category_name = material_category[0] %}
                        {% assign materials = material_category[1] %}
                        {% if materials.size > 0 %}
                        <div class="materials-section">
                            <div class="materials-title">{{ category_name | replace: '_', ' ' | capitalize }}</div>
                            <ul class="material-list">
                                {% for material in materials %}
                                <li class="material-item">
                                    {% if material.url %}
                                        {% if material.url contains 'http' %}
                                        <a href="{{ material.url }}" class="material-link">{{ material.name }}</a>
                                        {% elsif material.url contains '.md' %}
                                        {% assign clean_url = material.url | replace: '.md', '/' %}
                                        <a href="{{ clean_url | relative_url }}" class="material-link">{{ material.name }}</a>
                                        {% elsif material.url contains '.ipynb' %}
                                        {% assign clean_url = material.url | replace: '.ipynb', '/' %}
                                        <a href="{{ clean_url | relative_url }}" class="material-link">{{ material.name }}</a>
                                        {% else %}
                                        <a href="{{ material.url | relative_url }}" class="material-link">{{ material.name }}</a>
                                        {% endif %}
                                    {% else %}
                                    <span class="material-link">{{ material.name }}</span>
                                    {% endif %}
                                    {% if material.type %}
                                    <span class="material-type">{{ material.type }}</span>
                                    {% endif %}
                                    {% if material.description %}
                                    <div style="font-size: 12px; color: #666; margin-top: 2px;">{{ material.description }}</div>
                                    {% endif %}
                                </li>
                                {% endfor %}
                            </ul>
                        </div>
                        {% endif %}
                    {% endfor %}
                {% endif %}
                
                <!-- References Section -->
                {% if lecture.references %}
                <div class="materials-section">
                    <div class="materials-title">References</div>
                    <ul class="material-list">
                        <!-- Readings (merged textbooks and articles) -->
                        {% if lecture.references.textbooks %}
                            {% for book in lecture.references.textbooks %}
                            <li class="material-item">
                                 {% if book.url %}
                                <a href="{{ book.url }}" class="material-link">{{ book.name }}</a>
                                {% else %}
                                <span class="material-link">{{ book.name }}</span>
                                {% endif %}
                                {% if book.chapter %}<span class="material-type">{{ book.chapter }}</span>{% endif %}
                                {% if book.pages %}<span class="material-type">{{ book.pages }}</span>{% endif %}
                                {% if book.required %}<span class="material-type" style="background: #e74c3c; color: white;">Required</span>{% endif %}
                                {% if book.optional %}<span class="material-type" style="background: #3498db; color: white;">Optional</span>{% endif %}
                            </li>
                            {% endfor %}
                        {% endif %}
                        
                        {% if lecture.references.articles %}
                            {% for article in lecture.references.articles %}
                            <li class="material-item">
                                 {% if article.url %}
                                <a href="{{ article.url }}" class="material-link">{{ article.name }}</a>
                                {% else %}
                                <span class="material-link">{{ article.name }}</span>
                                {% endif %}
                                {% if article.type %}<span class="material-type">{{ article.type }}</span>{% endif %}
                                {% if article.required %}<span class="material-type" style="background: #e74c3c; color: white;">Required</span>{% endif %}
                                {% if article.optional %}<span class="material-type" style="background: #3498db; color: white;">Optional</span>{% endif %}
                            </li>
                            {% endfor %}
                        {% endif %}
                        
                        <!-- Videos -->
                        {% if lecture.references.videos %}
                            {% for video in lecture.references.videos %}
                            <li class="material-item">
                                 {% if video.url %}
                                <a href="{{ video.url }}" class="material-link">{{ video.name }}</a>
                                {% else %}
                                <span class="material-link">{{ video.name }}</span>
                                {% endif %}
                                {% if video.required %}<span class="material-type" style="background: #e74c3c; color: white;">Required</span>{% endif %}
                                {% if video.recommended %}<span class="material-type" style="background: #27ae60; color: white;">Recommended</span>{% endif %}
                                {% if video.optional %}<span class="material-type" style="background: #3498db; color: white;">Optional</span>{% endif %}
                            </li>
                            {% endfor %}
                        {% endif %}
                        
                        <!-- Tools -->
                        {% if lecture.references.tools %}
                            {% for tool in lecture.references.tools %}
                            <li class="material-item">
                                 {% if tool.url %}
                                <a href="{{ tool.url }}" class="material-link">{{ tool.name }}</a>
                                {% else %}
                                <span class="material-link">{{ tool.name }}</span>
                                {% endif %}
                                {% if tool.required %}<span class="material-type" style="background: #e74c3c; color: white;">Required</span>{% endif %}
                                {% if tool.optional %}<span class="material-type" style="background: #3498db; color: white;">Optional</span>{% endif %}
                            </li>
                            {% endfor %}
                        {% endif %}
                    </ul>
                </div>
                {% endif %}
                
                <!-- Key Concepts -->
                {% if lecture.key_concepts %}
                <div class="learning-objectives">
                    <div class="objectives-title">Key Concepts</div>
                    <div style="color: #555;">
                        {% for concept in lecture.key_concepts %}
                            <span style="background: #f0f0f0; padding: 2px 6px; margin: 2px; border-radius: 3px; font-size: 12px;">{{ concept | replace: '_', ' ' }}</span>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
        </div>
        </div>
        {% endfor %}
    </div>
</div>
{% endfor %}
{% else %}
<div class="schedule-section">
    <div class="section-header">
        <h3>No Lectures Found</h3>
    </div>
    <div class="section-content active">
        <p>No lecture metadata files were found. Please ensure meta.yml files exist in lecture folders.</p>
    </div>
</div>
{% endif %}

<script>
function toggleSection(header) {
    const content = header.nextElementSibling;
    const icon = header.querySelector('.toggle-icon');
    
    if (content.classList.contains('active')) {
        content.classList.remove('active');
        icon.textContent = '▶';
    } else {
        content.classList.add('active');
        icon.textContent = '▼';
    }
}

function toggleLectureDetails(header) {
    const details = header.nextElementSibling;
    
    if (details.classList.contains('active')) {
        details.classList.remove('active');
    } else {
        // Close other open lecture details in the same section
        const section = header.closest('.section-content');
        section.querySelectorAll('.lecture-details.active').forEach(detail => {
            detail.classList.remove('active');
        });
        
        details.classList.add('active');
    }
    
    // Prevent event bubbling to section toggle
    event.stopPropagation();
}
</script>

---

*This schedule is dynamically generated from lecture metadata. Materials and links are updated as they become available.*