// Motion & Animation utilities for FairScore
// Inspired by Motion / Framer Motion principles: restrained, subtle, spring physics

(function (root, factory) {
  if (typeof define === 'function' && define.amd) {
    define([], factory);
  } else if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.FairMotion = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // Check user preference for reduced motion
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /**
   * Smoothly animates a numeric value from start to end on an element
   */
  function animateValue(element, start, end, duration = 800, suffix = '', prefix = '') {
    if (!element) return;
    if (prefersReducedMotion) {
      element.textContent = `${prefix}${Math.round(end).toLocaleString()}${suffix}`;
      return;
    }

    const startTime = performance.now();
    const range = end - start;

    function update(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const ease = 1 - Math.pow(1 - progress, 3);
      const current = Math.round(start + range * ease);

      element.textContent = `${prefix}${current.toLocaleString()}${suffix}`;

      if (progress < 1) {
        requestAnimationFrame(update);
      }
    }

    requestAnimationFrame(update);
  }

  /**
   * Animates SVG circle stroke-dashoffset for score gauge
   */
  function animateGauge(circleEl, score, minScore = 300, maxScore = 900, duration = 1000) {
    if (!circleEl) return;
    const radius = circleEl.r.baseVal.value;
    const circumference = 2 * Math.PI * radius;

    circleEl.style.strokeDasharray = `${circumference} ${circumference}`;

    const normalized = Math.max(0, Math.min(1, (score - minScore) / (maxScore - minScore)));
    const targetOffset = circumference - normalized * circumference;

    if (prefersReducedMotion) {
      circleEl.style.strokeDashoffset = targetOffset;
      return;
    }

    // Set initial full offset
    circleEl.style.strokeDashoffset = circumference;
    circleEl.style.transition = `stroke-dashoffset ${duration}ms cubic-bezier(0.16, 1, 0.3, 1)`;

    requestAnimationFrame(() => {
      circleEl.style.strokeDashoffset = targetOffset;
    });
  }

  /**
   * Fade and slide in a container or view
   */
  function transitionIn(element, direction = 'up', duration = 240) {
    if (!element) return;
    if (prefersReducedMotion) {
      element.style.opacity = '1';
      element.style.transform = 'none';
      return;
    }

    const startY = direction === 'up' ? '8px' : direction === 'down' ? '-8px' : '0';
    element.style.opacity = '0';
    element.style.transform = `translateY(${startY})`;
    element.style.transition = `opacity ${duration}ms cubic-bezier(0.16, 1, 0.3, 1), transform ${duration}ms cubic-bezier(0.16, 1, 0.3, 1)`;

    requestAnimationFrame(() => {
      element.style.opacity = '1';
      element.style.transform = 'translateY(0)';
    });
  }

  /**
   * Stagger animate child items (e.g. queue rows or factor cards)
   */
  function staggerChildren(container, childSelector = '.motion-item', staggerMs = 40) {
    if (!container) return;
    const items = container.querySelectorAll(childSelector);
    if (!items.length) return;

    if (prefersReducedMotion) {
      items.forEach(el => {
        el.style.opacity = '1';
        el.style.transform = 'none';
      });
      return;
    }

    items.forEach((el, index) => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(6px)';
      el.style.transition = `opacity 200ms ease, transform 200ms cubic-bezier(0.16, 1, 0.3, 1)`;

      setTimeout(() => {
        el.style.opacity = '1';
        el.style.transform = 'translateY(0)';
      }, index * staggerMs);
    });
  }

  /**
   * Show a smooth status toast notification
   */
  function showToast(message, type = 'success', duration = 3500) {
    let container = document.getElementById('toast-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'toast-container';
      container.className = 'fixed bottom-5 right-5 z-50 flex flex-col gap-2 pointer-events-none';
      document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    const bg = type === 'success' ? 'bg-[#173D30] text-white border-[#245B45]' :
               type === 'error' ? 'bg-[#8B2525] text-white border-[#A83232]' :
               'bg-[#242B27] text-white border-[#77817A]';

    toast.className = `pointer-events-auto px-4 py-3 rounded-lg border shadow-lg flex items-center gap-3 text-sm font-medium ${bg} transform transition-all duration-300 opacity-0 translate-y-3`;
    
    const icon = type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ';
    toast.innerHTML = `<span class="font-bold">${icon}</span><span>${message}</span>`;
    container.appendChild(toast);

    requestAnimationFrame(() => {
      toast.classList.remove('opacity-0', 'translate-y-3');
      toast.classList.add('opacity-100', 'translate-y-0');
    });

    setTimeout(() => {
      toast.classList.remove('opacity-100', 'translate-y-0');
      toast.classList.add('opacity-0', 'translate-y-2');
      setTimeout(() => {
        if (toast.parentNode) toast.parentNode.removeChild(toast);
      }, 300);
    }, duration);
  }

  return {
    prefersReducedMotion,
    animateValue,
    animateGauge,
    transitionIn,
    staggerChildren,
    showToast
  };
});
