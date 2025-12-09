// longpage.js (updated for nav-group submenu handling)
document.addEventListener('DOMContentLoaded', () => {

  const menuToggle = document.getElementById('menu-toggle');
  const nav = document.getElementById('primary-nav');

  if (!menuToggle || !nav) return;

  function setMenuOpen(open) {
    nav.classList.toggle('open', open);
    menuToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    // optionally prevent body scroll when menu open
    document.body.style.overflow = open ? 'hidden' : '';
  }

  menuToggle.addEventListener('click', (e) => {
    e.preventDefault();
    setMenuOpen(!nav.classList.contains('open'));
  });

  // Close the menu when a nav link is clicked (mobile UX)
  nav.addEventListener('click', (e) => {
    const a = e.target.closest('a[href^="#"]');
    if (!a) return;
    // only close on small screens
    if (window.innerWidth <= 900) {
      setMenuOpen(false);
    }
  });

  // Close menu on Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') setMenuOpen(false);
  });

  const navParents = document.querySelectorAll('.sidenav .parent');            // parent links
  const navGroups = document.querySelectorAll('.sidenav .nav-group');         // containers
  const submenuItems = document.querySelectorAll('.sidenav .submenu a');      // submenu links
  const allHashLinks = Array.from(document.querySelectorAll('a[href^="#"]'));
  const sections = Array.from(document.querySelectorAll('.page-section'));
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // helper: normalize id
  const normalizeId = s => s ? s.replace(/^#/, '') : '';

  // Remove all active/expanded states
  const clearNavState = () => {
    navParents.forEach(p => p.classList.remove('active'));
    navGroups.forEach(g => g.classList.remove('expanded'));
    submenuItems.forEach(si => si.classList.remove('active'));
  };

  // Expand a nav-group for sectionId (e.g., 'tsp')
  function expandGroupFor(sectionId) {
    // clear first
    navParents.forEach(p => p.classList.remove('active'));
    navGroups.forEach(g => g.classList.remove('expanded'));
    submenuItems.forEach(si => si.classList.remove('active'));

    // find the group
    const group = document.querySelector(`.sidenav .nav-group[data-section="${sectionId}"]`);
    if (group) {
      group.classList.add('expanded');
      // mark the parent link active
      const parent = group.querySelector('.parent');
      if (parent) parent.classList.add('active');
    }
  }

  // Mark a submenu item active (and ensure its parent group is expanded)
  function setActiveDemo(fullId) {
    // fullId = 'tsp-12' or 'clustering-5'
    const parts = fullId.split('-');
    const parentSection = parts[0];
    expandGroupFor(parentSection);

    // mark submenu item
    const submenuLink = document.querySelector(`.sidenav .submenu a[data-target="${fullId}"]`);
    if (submenuLink) submenuLink.classList.add('active');
  }

  // Smooth scroll helper
  function smoothScrollToId(id) {
    const el = document.getElementById(id);
    if (!el) return;
    if (prefersReducedMotion) {
      el.scrollIntoView({ behavior: 'auto', block: 'start' });
      history.replaceState(null, '', `#${id}`);
      return;
    }
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    // update URL after small delay to avoid browser jump
    setTimeout(() => history.pushState(null, '', `#${id}`), 200);
  }

  // Parent click: toggle or activate & scroll
  navParents.forEach(parent => {
    parent.addEventListener('click', (e) => {
      e.preventDefault();
      const sectionId = normalizeId(parent.dataset.target || parent.getAttribute('href'));

      const parentGroup = parent.closest('.nav-group');
      if (!parentGroup) {
        // if no group, just scroll and set active
        expandGroupFor(sectionId);
        smoothScrollToId(sectionId);
        return;
      }

      // otherwise expand and scroll
      expandGroupFor(sectionId);
      smoothScrollToId(sectionId);
    });
  });

  // Submenu link clicks (cards / demo list links)
  document.addEventListener('click', (e) => {
    const a = e.target.closest('a[href^="#"]');
    if (!a) return;
    const href = a.getAttribute('href');
    if (!href || href === '#') return;
    const id = normalizeId(href);
    if (!document.getElementById(id)) return; // outside-page anchor -> ignore

    e.preventDefault();
    // set active demo (expands group + mark submenu item)
    setActiveDemo(id);
    smoothScrollToId(id);
  }, { passive: false });


// SECTION OBSERVER: detects top-level sections (home, tsp, clustering, other)
const sectionObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const id = entry.target.id;
      if (!id) return;
      // Activate / expand the parent group immediately when section top reaches activation zone
      expandGroupFor(id);
      history.replaceState(null, '', `#${id}`);
    }
  });
}, {
  root: null,
  threshold: 0, // trigger as soon as any part intersects the rootMargin box
  // rootMargin pushes the activation point down (40% from top)
  rootMargin: '-40% 0px -60% 0px'
});

// DEMO OBSERVER: detects demo blocks inside sections and marks submenu items active
const demoObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const id = entry.target.id;
      if (!id) return;
      // If demo block (contains a '-'), mark demo active
      if (id.indexOf('-') !== -1) {
        setActiveDemo(id); // expands parent and marks submenu item
        history.replaceState(null, '', `#${id}`);
      }
    }
  });
}, {
  root: null,
  threshold: [0.05, 0.25], // small fraction visible
  rootMargin: '0px 0px -60% 0px'
});

// Observe everything
document.querySelectorAll('.page-section').forEach(s => sectionObserver.observe(s));

// Observe demo blocks explicitly (class names you used earlier)
document.querySelectorAll('.tsp-demo-block, .clustering-demo-block, .other-demo-block')
  .forEach(el => demoObserver.observe(el));

  // On load, if there's a hash, scroll to it
  if (window.location.hash) {
    const id = normalizeId(window.location.hash);
    // if id is a demo (has '-') set active demo; if top-level, expand that group
    if (id.includes('-')) setActiveDemo(id);
    else expandGroupFor(id);
    setTimeout(() => smoothScrollToId(id), 100);
  }

  // handle hashchange
  window.addEventListener('hashchange', () => {
    const id = normalizeId(location.hash);
    if (!id) return;
    if (id.includes('-')) setActiveDemo(id);
    else expandGroupFor(id);
  });
});
