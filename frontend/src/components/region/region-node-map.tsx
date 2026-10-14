import { useEffect, useMemo, useState } from "react";

/*
 * 지역분석 hero map: the 229 시군구 analysed by the AI drawn as nodes on the
 * same teal point grid the landing uses. The selected region glows, regions in
 * the comparison list get numbered coral rings. Clicking a node selects that
 * region (the same thing the search/select controls do). Node positions are
 * simplified centroids for orientation only — not survey-accurate.
 */
type Pt = [number, number];
type Geo = { koreaDots: Pt[]; nodes: Record<string, Pt> };

const W = 560;
const H = 600;
const LON0 = 124.55, LON1 = 131.0, LAT0 = 33.05, LAT1 = 38.7;
// equirectangular with a cos(lat) squeeze so the peninsula keeps its shape
const K = Math.cos((36 * Math.PI) / 180);
const SCALE = Math.min(W / ((LON1 - LON0) * K), H / (LAT1 - LAT0));
const OX = (W - (LON1 - LON0) * K * SCALE) / 2;
const OY = (H - (LAT1 - LAT0) * SCALE) / 2;
const project = ([lon, lat]: Pt): Pt => [OX + (lon - LON0) * K * SCALE, OY + (LAT1 - lat) * SCALE];
const shortName = (name: string) => name.replace(/^(\S+?)(특별자치도|특별자치시|특별시|광역시|도)\s/, (_, a: string) => `${a.slice(0, 2)} `);

export function RegionNodeMap({
  regions,
  selected,
  compare,
  onSelect,
}: {
  regions: string[];
  selected: string;
  compare: string[];
  onSelect: (name: string) => void;
}) {
  const [geo, setGeo] = useState<Geo | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${import.meta.env.BASE_URL}landing/korea-scroll.json`, { signal: controller.signal })
      .then((r) => r.json())
      .then((g: Geo) => setGeo(g))
      .catch(() => {});
    return () => controller.abort();
  }, []);

  const dots = useMemo(
    () => (geo ? geo.koreaDots.filter(([lon]) => lon <= LON1).map(project) : []),
    [geo],
  );
  const nodes = useMemo(() => {
    if (!geo) return [];
    const known = new Set(regions);
    return Object.entries(geo.nodes)
      .filter(([name, p]) => known.has(name) && p[0] <= LON1)
      .map(([name, p]) => ({ name, xy: project(p) }))
      // intro order: nodes switch on from north to south like a scan
      .map((n) => ({ ...n, delay: Math.round(Math.max(0, n.xy[1] / H) * 900) }));
  }, [geo, regions]);

  const sel = nodes.find((n) => n.name === selected) ?? null;
  const hov = hover && hover !== selected ? nodes.find((n) => n.name === hover) ?? null : null;
  const label = (n: { name: string; xy: Pt }, tone: "sel" | "hov") => {
    const text = shortName(n.name);
    const w = text.length * 12.5 + 22;
    const right = n.xy[0] < W - w - 40;
    const x = right ? n.xy[0] + 18 : n.xy[0] - 18 - w;
    const y = n.xy[1] - 34;
    return (
      <g className={`rnm-label ${tone}`} pointerEvents="none">
        <path d={`M${n.xy[0]} ${n.xy[1]} L${right ? x : x + w} ${y + 13}`} />
        <rect x={x} y={y} width={w} height={26} rx={13} />
        <text x={x + w / 2} y={y + 17.5} textAnchor="middle">
          {text}
        </text>
      </g>
    );
  };

  return (
    <figure className="region-node-map">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`AI 분석 ${regions.length}개 시군구 위치${selected ? ` · 선택: ${selected}` : ""}`}
        onMouseLeave={() => setHover(null)}
      >
        <g className="rnm-dots">
          {dots.map(([x, y], i) => (
            <rect key={i} x={x - 0.9} y={y - 0.9} width={1.8} height={1.8} />
          ))}
        </g>
        <g className="rnm-nodes">
          {nodes.map((n) => {
            const ci = compare.indexOf(n.name);
            return (
              <g
                key={n.name}
                className={`rnm-node${n.name === selected ? " is-selected" : ""}${ci >= 0 ? " is-compare" : ""}${n.name === hover ? " is-hover" : ""}`}
                transform={`translate(${n.xy[0]} ${n.xy[1]})`}
                onMouseEnter={() => setHover(n.name)}
                onClick={() => onSelect(n.name)}
              >
                <circle className="hit" r={8} />
                <ellipse className="shadow" rx={4} ry={1.6} />
                <g className="lift">
                  <line className="stem" y1={0} y2={7} />
                  <circle className="core" r={2.6} style={{ animationDelay: `${n.delay}ms` }} />
                  {ci >= 0 ? (
                    <>
                      <circle className="ring" r={8} />
                      <text className="idx" y={-12} textAnchor="middle">
                        {ci + 1}
                      </text>
                    </>
                  ) : null}
                </g>
              </g>
            );
          })}
        </g>
        {sel ? (
          <g className="rnm-focus" style={{ transform: `translate(${sel.xy[0]}px, ${sel.xy[1]}px)` }} pointerEvents="none">
            <circle className="pulse" r={10} />
            <circle className="pulse late" r={10} />
            <circle className="halo" r={16} />
            <circle className="dot" r={5} />
          </g>
        ) : null}
        {sel ? <g key={sel.name} className="rnm-label-in">{label(sel, "sel")}</g> : null}
        {hov ? label(hov, "hov") : null}
      </svg>
      <figcaption>
        <span>
          <i className="k-node" /> AI 분석 {regions.length}개 시군구
        </span>
        <span>
          <i className="k-sel" /> 선택 지역
        </span>
        <span>
          <i className="k-cmp" /> 비교 지역
        </span>
        <em>점 위치는 시군구 중심 근사값 · 클릭하면 선택</em>
      </figcaption>
    </figure>
  );
}
