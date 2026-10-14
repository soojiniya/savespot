/*
 * Smooth wheel scrolling for the landing page only.
 *
 * A mouse wheel moves the page in ~100px notches, which makes the scroll-driven
 * map feel steppy. While the landing is mounted, wheel input is turned into a
 * scroll *target* and the window glides toward it each frame. Everything else
 * stays native:
 *  - trackpads (small pixel deltas) already scroll smoothly → left alone
 *  - ctrl/⌘ + wheel (browser zoom), horizontal wheels → left alone
 *  - a scrollable box under the cursor that can still scroll (e.g. the AI answer
 *    card) keeps the wheel
 *  - keyboard, scrollbar dragging and in-page jumps re-sync the target
 * Returns a cleanup function.
 */
const EASE = 9; // 1/s — how fast the page catches up with the wheel

function canScroll(el: Element | null, dy: number): boolean {
  for (let node = el; node && node !== document.body && node !== document.documentElement; node = node.parentElement) {
    const style = getComputedStyle(node);
    if (!/(auto|scroll)/.test(style.overflowY)) continue;
    const max = node.scrollHeight - node.clientHeight;
    if (max <= 1) continue;
    if ((dy > 0 && node.scrollTop < max - 1) || (dy < 0 && node.scrollTop > 0)) return true;
  }
  return false;
}

export function smoothWheel(): () => void {
  const root = document.scrollingElement ?? document.documentElement;
  let target = window.scrollY;
  let current = window.scrollY;
  let raf = 0;
  let last = 0;
  let animating = false;

  const maxY = () => root.scrollHeight - window.innerHeight;

  const step = (now: number) => {
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 1 / 60;
    last = now;
    current += (target - current) * (1 - Math.exp(-dt * EASE));
    if (Math.abs(target - current) < 0.5) current = target;
    window.scrollTo(0, current);
    if (current !== target) raf = requestAnimationFrame(step);
    else { animating = false; last = 0; }
  };

  const onWheel = (e: WheelEvent) => {
    if (e.defaultPrevented || e.ctrlKey || e.metaKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
    let dy = e.deltaY;
    if (e.deltaMode === 1) dy *= 40; // lines
    else if (e.deltaMode === 2) dy *= window.innerHeight; // pages
    // trackpads and precision wheels send many small deltas and are smooth already
    if (e.deltaMode === 0 && Math.abs(dy) < 50 && !animating) return;
    if (canScroll(e.target as Element, dy)) return;
    e.preventDefault();
    if (!animating) { current = window.scrollY; target = current; }
    target = Math.max(0, Math.min(maxY(), target + dy));
    if (!animating) { animating = true; last = 0; raf = requestAnimationFrame(step); }
  };

  // any scroll we didn't cause (keys, scrollbar, anchor jumps) becomes the new resting point
  const onScroll = () => {
    // our own glide lands within a pixel of `current`; anything else came from elsewhere
    if (animating && Math.abs(window.scrollY - Math.round(current)) <= 2) return;
    if (animating) { cancelAnimationFrame(raf); animating = false; last = 0; }
    current = target = window.scrollY;
  };

  window.addEventListener("wheel", onWheel, { passive: false });
  window.addEventListener("scroll", onScroll, { passive: true });
  return () => {
    cancelAnimationFrame(raf);
    window.removeEventListener("wheel", onWheel);
    window.removeEventListener("scroll", onScroll);
  };
}
