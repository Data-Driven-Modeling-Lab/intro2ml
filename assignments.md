---
layout: default
title: Assignments
---

<style>
.assignment-item { margin: 1.25rem 0; padding: 0.75rem 1rem; border: 1px solid #e5e7eb; border-radius: 8px; }
.assignment-header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.5rem 1rem; }
.assignment-title { font-weight: 600; font-size: 1.05rem; }
.assignment-meta { color: #6b7280; font-size: 0.95rem; }
.assignment-links { margin-top: 0.35rem; font-size: 0.95rem; }
.assignment-links a { display: inline-block; margin-right: 0.75rem; text-decoration: none; color: #2563eb; }
.assignment-links a:hover { text-decoration: underline; }
.assignment-desc { margin-top: 0.35rem; }
</style>

## Assignments

{% assign today_ts = site.time | date: "%s" | plus: 0 %}
{% assign items = site.data.assignments | sort: 'release_date' | reverse %}
{% if items and items.size > 0 %}
  {% for a in items %}
  {% assign release_ts = a.release_date | date: "%s" | plus: 0 %}
  {% if release_ts <= today_ts %}
  <div class="assignment-item">
    <div class="assignment-header">
      <div class="assignment-title">{{ a.title }}</div>
      <div class="assignment-meta">Released: {{ a.release_date }} • Due: {{ a.due_date }}</div>
    </div>
    {% if a.description %}
    <div class="assignment-desc">{{ a.description }}</div>
    {% endif %}
    {% if a.links %}
    <div class="assignment-links">
      {% for l in a.links %}
        <a href="{{ l.url }}">{{ l.name }}</a>
      {% endfor %}
    </div>
    {% endif %}
  </div>
  {% endif %}
  {% endfor %}
{% else %}
No assignments published yet.
{% endif %}
