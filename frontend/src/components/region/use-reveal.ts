import { useEffect, useRef } from "react";

/*
 * Scroll reveal for the 지역분석 sections: children marked `.rv` rise in once
 * as they enter the viewport. The hidden state only applies after the root gets
 * `.fx-on`, so without JS (or with reduced motion) everything is simply shown.
 * A MutationObserver picks up sections that mount later (e.g. after a region
 * is selected).
 */
export function useReveal<T extends HTMLElement>(ready: boolean) {
  const ref = useRef<T | null>(null);
  useEffect(() => {
    const root = ref.current;
    if (!root || !ready) return;
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches || !("IntersectionObserver" in window)) return;
    root.classList.add("fx-on");
    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          entry.target.classList.add("is-in");
          io.unobserve(entry.target);
        }
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.08 },
    );
    const scan = () => root.querySelectorAll(".rv:not(.is-in)").forEach((el) => io.observe(el));
    scan();
    const mo = new MutationObserver(scan);
    mo.observe(root, { childList: true, subtree: true });
    return () => {
      io.disconnect();
      mo.disconnect();
      root.classList.remove("fx-on");
    };
  }, [ready]);
  return ref;
}
