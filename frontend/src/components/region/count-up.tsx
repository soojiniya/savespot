import { useEffect, useRef, useState } from "react";

/*
 * Animates the first number inside an already-formatted string ("6,852명",
 * "27.9점", "144위") from the previously shown value to the new one. The final
 * frame is always the exact string passed in, so formatting and values are
 * never altered — only the in-between frames are interpolated.
 */
const NUM = /-?[\d,]*\.?\d+/;
const reduce = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

function parse(text: string) {
  const m = text.match(NUM);
  if (!m || m.index == null) return null;
  const raw = m[0];
  const value = Number(raw.replace(/,/g, ""));
  if (!Number.isFinite(value)) return null;
  const decimals = raw.includes(".") ? raw.split(".")[1].length : 0;
  return { value, decimals, grouped: raw.includes(","), before: text.slice(0, m.index), after: text.slice(m.index + raw.length) };
}

export function CountUp({ text, duration = 900 }: { text: string; duration?: number }) {
  const [shown, setShown] = useState(text);
  const from = useRef<number | null>(null);
  useEffect(() => {
    const target = parse(text);
    if (!target || reduce()) {
      setShown(text);
      from.current = target?.value ?? null;
      return;
    }
    const start = from.current ?? 0;
    from.current = target.value;
    if (start === target.value) {
      setShown(text);
      return;
    }
    let frame = 0;
    const t0 = performance.now();
    const fmt = (v: number) =>
      target.before +
      (target.grouped
        ? v.toLocaleString("ko-KR", { minimumFractionDigits: target.decimals, maximumFractionDigits: target.decimals })
        : v.toFixed(target.decimals)) +
      target.after;
    const tick = (now: number) => {
      const t = Math.min(1, (now - t0) / duration);
      const e = 1 - Math.pow(1 - t, 4); // ease-out quart
      if (t < 1) {
        setShown(fmt(start + (target.value - start) * e));
        frame = requestAnimationFrame(tick);
      } else setShown(text);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [text, duration]);
  return <>{shown}</>;
}
