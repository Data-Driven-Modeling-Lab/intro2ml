---
layout: default
title: Weekly Notes
---

<style>
.wn-intro { color: #4b5563; max-width: 46em; line-height: 1.65; }
.wn-item {
  margin: 1.1rem 0;
  padding: 0.9rem 1.1rem;
  border: 1px solid #e5e7eb;
  border-left: 3px solid #7aa2f7;
  border-radius: 8px;
}
.wn-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.4rem 0.9rem; }
.wn-week {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.72rem; letter-spacing: 0.09em; text-transform: uppercase;
  color: #3b63c4;
}
.wn-title { font-weight: 600; font-size: 1.06rem; }
.wn-title a { color: inherit; text-decoration: none; }
.wn-title a:hover { color: #2563eb; }
.wn-date { color: #6b7280; font-size: 0.9rem; }
.wn-summary { margin-top: 0.35rem; color: #374151; line-height: 1.6; }
</style>

## Weekly Notes

<p class="wn-intro">
A note goes out at the start of each week: the plan, a recap of the week before,
and whatever I happen to be thinking about that is relevant. They are kept here
so they do not only live in your inbox, and so you can go back to them when a
question in class turns out to have been answered in one.
</p>

{% assign notes = site.notes | sort: 'week' | reverse %}
{% if notes and notes.size > 0 %}
  {% for n in notes %}
  <div class="wn-item">
    <div class="wn-head">
      <span class="wn-week">Week {{ n.week }}</span>
      <span class="wn-title"><a href="{{ n.url | relative_url }}">{{ n.title }}</a></span>
      <span class="wn-date">{{ n.date | date: "%b %-d, %Y" }}</span>
    </div>
    {% if n.summary %}<div class="wn-summary">{{ n.summary }}</div>{% endif %}
  </div>
  {% endfor %}
{% else %}
No notes yet.
{% endif %}
