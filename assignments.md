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

{% assign items = site.data.assignments | sort: 'release_date' | reverse %}
{% if items and items.size > 0 %}
  {% for a in items %}
  {% if a.visible == false %}{% continue %}{% endif %}
  <div class="assignment-item is-scheduled" data-release="{{ a.release_date }}" hidden>
    <div class="assignment-header">
      <div class="assignment-title">{{ a.title }}</div>
      <div class="assignment-meta">Released: {{ a.release_date }} &bull; Due: {{ a.due_date }}</div>
    </div>
    {% if a.description %}
    <div class="assignment-desc">{{ a.description }}</div>
    {% endif %}
    {% if a.links %}
    <div class="assignment-links">
      {% for l in a.links %}
        {% unless l.visible == false %}<a href="{{ l.url }}">{{ l.name }}</a>{% endunless %}
      {% endfor %}
    </div>
    {% endif %}
  </div>
  {% endfor %}
  <p id="no-assignments-yet" hidden>No assignments published yet.</p>
{% else %}
No assignments published yet.
{% endif %}

<script>
// Release dates are compared against the reader's date, not the build date.
// Jekyll's site.time is the moment the site was last built, so a set whose
// release date fell after the last push stayed invisible until someone
// happened to push again. PS0 released 2026-09-01 against a site last built
// 2026-08-31, and simply did not appear. Deciding in the browser means a set
// shows up on its date with no rebuild.
(function () {
  var today = new Date();
  today.setHours(0, 0, 0, 0);
  var shown = 0;
  document.querySelectorAll('.assignment-item.is-scheduled').forEach(function (el) {
    var parts = (el.dataset.release || '').split('-');
    if (parts.length !== 3) { el.hidden = false; shown++; return; }
    var rel = new Date(+parts[0], +parts[1] - 1, +parts[2]);
    if (rel <= today) { el.hidden = false; shown++; }
  });
  if (!shown) {
    var none = document.getElementById('no-assignments-yet');
    if (none) none.hidden = false;
  }
})();
</script>
