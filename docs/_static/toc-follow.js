// Mark the page-TOC entries of the section being read as active, so that the
// theme shows the subsections of that section only. The theme's own ScrollSpy
// targets ".bd-toc-nav", which sphinx-book-theme does not render.
(function () {
  function setup() {
    var nav = document.getElementById("pst-page-toc-nav");
    if (!nav) return;

    var entries = [];
    nav.querySelectorAll("li.toc-entry > a.nav-link").forEach(function (link) {
      var id = decodeURIComponent(link.hash.slice(1));
      var target = id && document.getElementById(id);
      if (target) entries.push({ item: link.parentElement, target: target });
    });
    if (!entries.length) return;

    var scheduled = false;
    function update() {
      scheduled = false;
      var offset = 120;
      var current = null;
      entries.forEach(function (entry) {
        if (entry.target.getBoundingClientRect().top <= offset) current = entry;
      });
      nav.querySelectorAll("li.toc-entry.active").forEach(function (item) {
        item.classList.remove("active");
      });
      for (var item = current && current.item; item && nav.contains(item); item = item.parentElement) {
        if (item.matches("li.toc-entry")) item.classList.add("active");
      }
    }

    function schedule() {
      if (!scheduled) {
        scheduled = true;
        setTimeout(update, 50);
      }
    }

    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("hashchange", schedule);
    update();
  }

  window.addEventListener("DOMContentLoaded", setup);
}());
