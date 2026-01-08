
document.addEventListener('DOMContentLoaded', () => {
    const mainSections = document.querySelectorAll(".page-section[id]");
    console.log("mainSections:", mainSections);
    const subSections = document.querySelectorAll("[class$=-demo-block]");
    console.log("subSections: ", subSections);
    const listLinks = document.querySelectorAll("[class$=-project-link]");
    console.log("projectLink: ", listLinks)

    const allSections = [
    ...mainSections,
    ...subSections
    ];

    allSections.sort((a, b) => {
        if (a === b) return 0;

        const position = a.compareDocumentPosition(b);

        if (position & Node.DOCUMENT_POSITION_FOLLOWING) {
            return -1;
        } else {
            return 1;
        }
    });

    const ACTIVATION_THRESHOLD = 120; //pixels from the top of the screen

    function activateOnScroll() {
        //Reset active and expanded
        navGroups.forEach(group => {
            group.classList.remove('expanded');
            group.querySelector('.parent')?.classList.remove('active');
        });

        subLinks.forEach(link => {
            link.classList.remove('active');
        });

        let active = null;

        allSections.forEach(section => {
            const rect = section.getBoundingClientRect();

            if (rect.top <= ACTIVATION_THRESHOLD && rect.bottom > ACTIVATION_THRESHOLD) {
                active = section;
            }
        })

        if (!active) return

        let activeMain = null;
        let activeSub = null;

        if (active.dataset.type === 'main') {
          activeMain = active;
          const group = document.querySelector(`.nav-group[data-section="${activeMain.id}"]`);
          group.classList.add("expanded")
          const parentLink = group.querySelector('.parent');
          parentLink.classList.add('active');
        }

        if (active.dataset.type === 'sub') {
          activeMain = document.getElementById(active.dataset.parent);
          const group = document.querySelector(`.nav-group[data-section="${activeMain.id}"]`);
          group.classList.add("expanded")
          const parentLink = group.querySelector('.parent');
          parentLink.classList.add('active');

          activeSub = active;
          const subLink = document.querySelector(`.submenu a[data-target="${activeSub.id}"]`);
          subLink.classList.add('active');
        }

        if (activeMain) {
            console.log('MAIN:', activeMain.id);
        }
        if (activeSub) {
            console.log('SUB:', activeSub.id);
        }
    }
    //Scroll listener activates sections based on the what is in view
    window.addEventListener('scroll', activateOnScroll);

    const navGroups = document.querySelectorAll('.nav-group');
    console.log("Navgroups", navGroups)
    const parentLinks = document.querySelectorAll('.nav-group .parent');
    const subLinks = document.querySelectorAll('.submenu a');

    //Reset active and expanded
    navGroups.forEach(group => {
        group.classList.remove('expanded');
        group.querySelector('.parent')?.classList.remove('active');
    });

    subLinks.forEach(link => {
        link.classList.remove('active');
    });

    //Click handler for main section links
    parentLinks.forEach(link => {
        link.addEventListener('click', e => {
            e.preventDefault();
            const id = link.dataset.target;
            const target = document.getElementById(id);

            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });

            history.replaceState(null, '', `#${id}`)
        });
    });

    //Click handler for subsection links
    subLinks.forEach(link => {
        link.addEventListener('click', e => {
            e.preventDefault();
            const id = link.dataset.target;
            const target = document.getElementById(id);

            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });

            history.replaceState(null, '', `#${id}`)
        });
    });

    //Click handler for list links
    listLinks.forEach(link => {
        link.addEventListener('click', e => {
            e.preventDefault();
            const id = link.dataset.target;
            const target = document.getElementById(id);

            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });

            history.replaceState(null, '', `#${id}`)
        });
    });

    const sidenav = document.querySelector('.sidenav');
    const menuToggle = document.getElementById('menu-toggle');

    const MOBILE_BREAKPOINT = 900;

    function isMobile() {
      return window.innerWidth <= MOBILE_BREAKPOINT;
    }

    menuToggle?.addEventListener('click', () => {
      sidenav.classList.toggle('open');
    });

    function closeMenuIfMobile() {
      if (isMobile()) {
        sidenav.classList.remove('open');
      }
    }

    //Close menu after click for small screens
    parentLinks.forEach(link => {
      link.addEventListener('click', () => {
        closeMenuIfMobile();
      });
    });

    subLinks.forEach(link => {
      link.addEventListener('click', () => {
        closeMenuIfMobile();
      });
    });

    // Close menu when resizing from mobile → desktop
    window.addEventListener('resize', () => {
      if (!isMobile()) {
        sidenav.classList.remove('open');
      }
    });







});



