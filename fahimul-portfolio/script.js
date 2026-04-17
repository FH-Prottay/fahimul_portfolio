/* ============================================================
   FAHIMUL HAQUE — PORTFOLIO JS
   ============================================================ */

/* ── NAV SCROLL ──────────────────────────────────────────── */
const nav = document.getElementById('nav');
window.addEventListener('scroll', () => {
  nav.classList.toggle('scrolled', window.scrollY > 60);
}, { passive: true });

/* ── MOBILE NAV ──────────────────────────────────────────── */
const navToggle = document.getElementById('navToggle');
const navLinks  = document.getElementById('navLinks');

navToggle.addEventListener('click', () => {
  navLinks.classList.toggle('open');
  const spans = navToggle.querySelectorAll('span');
  const isOpen = navLinks.classList.contains('open');
  if (isOpen) {
    spans[0].style.transform = 'rotate(45deg) translate(5px, 5px)';
    spans[1].style.opacity = '0';
    spans[2].style.transform = 'rotate(-45deg) translate(5px, -5px)';
  } else {
    spans.forEach(s => { s.style.transform = ''; s.style.opacity = ''; });
  }
});

navLinks.querySelectorAll('a').forEach(a => {
  a.addEventListener('click', () => {
    navLinks.classList.remove('open');
    navToggle.querySelectorAll('span').forEach(s => { s.style.transform = ''; s.style.opacity = ''; });
  });
});

/* ── SCROLL REVEAL ───────────────────────────────────────── */
(function () {
  const observer = new IntersectionObserver(
    entries => entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        observer.unobserve(e.target);
      }
    }),
    { threshold: 0.1, rootMargin: '0px 0px -36px 0px' }
  );
  document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
})();

/* ── PUBLICATION TABS ────────────────────────────────────── */
document.querySelectorAll('.pub-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.pub-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.pub-list').forEach(l => l.classList.remove('active'));
    tab.classList.add('active');
    const list = document.getElementById('tab-' + tab.dataset.tab);
    if (list) {
      list.classList.add('active');
      list.querySelectorAll('.reveal:not(.visible)').forEach(el => {
        setTimeout(() => el.classList.add('visible'), 50);
      });
    }
  });
});

/* ── SMOOTH SCROLL ───────────────────────────────────────── */
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const target = document.querySelector(a.getAttribute('href'));
    if (!target) return;
    e.preventDefault();
    const offset = 70;
    window.scrollTo({ top: target.getBoundingClientRect().top + window.scrollY - offset, behavior: 'smooth' });
  });
});

/* ── CHATBOT ─────────────────────────────────────────────── */
(function () {
  const toggle   = document.getElementById('chatbot-toggle');
  const panel    = document.getElementById('chatbot-panel');
  const closeBtn = document.getElementById('chatbot-close-btn');
  const input    = document.getElementById('chatbot-input');
  const sendBtn  = document.getElementById('chatbot-send');
  const messages = document.getElementById('chatbot-messages');
  const iconOpen = document.getElementById('chat-open');
  const iconClose= document.getElementById('chat-close');
  if (!toggle) return;

  let isOpen = false;
  let history = [];

  const open  = () => { isOpen = true;  panel.classList.add('open');    iconOpen.style.display='none';  iconClose.style.display='block'; input.focus(); };
  const close = () => { isOpen = false; panel.classList.remove('open'); iconOpen.style.display='block'; iconClose.style.display='none'; };

  toggle.addEventListener('click', () => isOpen ? close() : open());
  closeBtn.addEventListener('click', close);

  function addMsg(text, role) {
    const d = document.createElement('div');
    d.className = 'msg msg-' + (role === 'user' ? 'user' : 'bot');
    d.textContent = text;
    messages.appendChild(d);
    messages.scrollTop = messages.scrollHeight;
  }

  function addTyping() {
    const d = document.createElement('div');
    d.className = 'msg msg-bot msg-typing'; d.id = 'typing';
    d.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
    messages.appendChild(d);
    messages.scrollTop = messages.scrollHeight;
  }

  function removeTyping() { const t = document.getElementById('typing'); if (t) t.remove(); }

  async function send() {
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    input.style.height = 'auto';
    sendBtn.disabled = true;
    addMsg(text, 'user');
    history.push({ role: 'user', content: text });
    addTyping();
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: history.slice(-10) }),
      });
      removeTyping();
      if (!res.ok) throw new Error();
      const data = await res.json();
      const reply = data.response || 'Sorry, please try again.';
      addMsg(reply, 'bot');
      history.push({ role: 'assistant', content: reply });
    } catch {
      removeTyping();
      addMsg('Sorry, something went wrong. Please email fahimulhaq2001@gmail.com directly.', 'bot');
    }
    sendBtn.disabled = false;
    input.focus();
  }

  sendBtn.addEventListener('click', send);
  input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } });
  input.addEventListener('input', () => { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 90) + 'px'; });
})();
