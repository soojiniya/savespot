import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  GeoJSON as LeafletGeoJSON,
  MapContainer,
  Marker,
  TileLayer,
  Tooltip,
  useMap,
  useMapEvents,
} from "react-leaflet";
import L from "leaflet";
import { BarChart } from "./components/charts/bar-chart";
import { Bar } from "./components/charts/bar";
import { BarXAxis } from "./components/charts/bar-x-axis";
import { RingChart } from "./components/charts/ring-chart";
import { Ring } from "./components/charts/ring";
import { RingCenter } from "./components/charts/ring-center";
import { LineChart } from "./components/charts/line-chart";
import { Line } from "./components/charts/line";
import { ScatterChart } from "./components/charts/scatter-chart";
import { Scatter } from "./components/charts/scatter";
import { Grid } from "./components/charts/grid";
import NumberFlow from "@number-flow/react";
import { DotLand } from "./components/map/dot-land";
import { RegionNodeMap } from "./components/region/region-node-map";
import { CountUp as CountUpText } from "./components/region/count-up";
import { useReveal } from "./components/region/use-reveal";
import { topoToGeoJSON } from "./components/map/topo";
import { ScrollLanding, type LandingHotspot } from "./components/landing/scroll-landing";

const API_BASE = (
  import.meta.env.VITE_API_BASE ??
  import.meta.env.VITE_API_URL ??
  "http://127.0.0.1:8000"
).replace(/\/$/, "");
// Public demo builds set VITE_AI_ASSISTANT_ENABLED=false: the AI question box
// stays on screen but never sends a request (so no OpenAI cost can occur).
const ASSISTANT_BUILD_ENABLED =
  String(import.meta.env.VITE_AI_ASSISTANT_ENABLED ?? "true").trim().toLowerCase() !== "false";
const ASSISTANT_DISABLED_MESSAGE =
  "데모 버전에서는 AI 자연어 분석 기능이 비활성화되어 있습니다.";
// Design palette (UI only). Ink for observed data, blue for AI-derived values,
// orange/purple for the two hotspot source types (also told apart by shape).
const INK = "#D6D8DB";
const INK_MUTED = "rgba(214,216,219,.35)";
const AI_BLUE = "#5FB39D";
const SCHOOLZONE_COLOR = "#FF7A5C";
const PEDESTRIAN_COLOR = "#96FFE2";
const AMBER = INK;
// 지역분석 palette (landing language): teal = AI, mint = rate, ink/slate = counts
const REGION_TEAL = "#2FE0C0";
const REGION_MINT = "#96FFE2";
const REGION_INK = "#E9F4F2";
const REGION_SLATE = "#5D6F73";
const MUTED_AMBER = INK_MUTED;
const DEFAULT_CENTER: [number, number] = [36.35, 127.8];

type View = "home" | "map" | "region" | "ai" | "guide";
type SourceType = "schoolzone_hotspot" | "child_pedestrian_hotspot";
type FilterOption = { value: string; label: string };
type Hotspot = {
  hotspot_id: number;
  source_type: SourceType;
  location_name?: string;
  accident_year?: number | null;
  year_verification_status?: string;
  sido?: string;
  sigungu?: string;
  latitude: number;
  longitude: number;
  accident_count?: number;
  casualty_count?: number;
  death_count?: number;
  serious_injury_count?: number;
  minor_injury_count?: number;
  injury_report_count?: number;
};
type HotspotResponse = {
  items: Hotspot[];
  pagination: { limit: number; offset: number; count: number; total: number };
};
type NearbyItem = Hotspot & { distance_m?: number };
type NearbyResponse = { items: NearbyItem[] };
type Region = {
  sido_sgg_nm: string;
  SIDO_NM?: string;
  SIGUNGU_NM?: string;
  child_accident_count?: number;
  child_death_count?: number;
  child_injury_count?: number;
  child_population_0_12?: number;
  accident_per_10k_child?: number;
  crosswalk_count?: number;
  traffic_light_count?: number;
  schoolzone_count?: number;
  cctv_install_rate?: number;
  infrastructure_score?: number;
  speed_bump_count?: number | null;
};
type RegionResponse = {
  items: Region[];
  pagination: { total: number; count: number; limit?: number; offset?: number };
};
type AiRiskRegion = {
  risk_rank_2026: number;
  sido_sgg_nm: string;
  predicted_high_risk_probability_2026: number;
  prev_child_population?: number;
  prev_vehicle_count?: number;
  prev_accident_per_10k_child?: number;
  prev_accident_count?: number;
  prev_injury_count?: number;
  prev_death_count?: number;
  accident_count_change_1y?: number;
  accident_rate_change_1y?: number;
  accident_count_mean_2y?: number;
  [key: string]: string | number | undefined;
};
type AiRiskResponse = {
  items: AiRiskRegion[];
  pagination?: { total: number; count: number };
  model?: ModelInfo["model"];
};
type ImprovementCandidate = {
  improvement_priority_rank: number;
  sido_sgg_nm: string;
  predicted_high_risk_probability_2026: number;
  infrastructure_score: number;
  infrastructure_deficiency_score: number;
  improvement_priority_score: number;
  crosswalk_per_1000_child?: number;
  traffic_light_per_1000_child?: number;
  schoolzone_per_1000_child?: number;
  cctv_install_rate?: number;
};
type ImprovementResponse = {
  items: ImprovementCandidate[];
  pagination: { total: number; count: number };
};
type ModelInfo = {
  model: {
    model_available: boolean;
    model_name?: string;
    feature_cols: string[];
    target: string;
    prediction_unit?: string;
    positive_label?: string;
    not_supported?: string[];
  };
  metrics?: Array<Record<string, string | number>>;
};
type ImportanceItem = {
  feature: string;
  importance_mean: number;
  importance_std?: number;
};
type ImportanceResponse = {
  items: ImportanceItem[];
  explanation_method: string;
  shap_availbale?: boolean;
  shap_available?: boolean;
  model?: ModelInfo["model"];
};

const sourceMeta: Record<
  SourceType,
  { label: string; short: string; color: string }
> = {
  schoolzone_hotspot: {
    label: "어린이보호구역 사고다발지역",
    short: "보호구역",
    color: SCHOOLZONE_COLOR,
  },
  child_pedestrian_hotspot: {
    label: "보행어린이 사고다발지역",
    short: "보행어린이",
    color: PEDESTRIAN_COLOR,
  },
};

const featureLabels: Record<string, string> = {
  prev_child_population: "전년도 어린이 인구",
  prev_vehicle_count: "전년도 자동차 등록대수",
  prev_accident_count: "전년도 어린이 사고건수",
  prev_injury_count: "전년도 부상자수",
  prev_death_count: "전년도 사망자수",
  prev_accident_per_10k_child: "전년도 어린이 1만명당 사고율",
  prev_vehicles_per_100_child: "전년도 어린이 100명당 자동차 등록대수",
  accident_count_change_1y: "최근 1년 사고건수 변화",
  accident_rate_change_1y: "최근 1년 사고율 변화",
  accident_count_mean_2y: "최근 2년 평균 사고건수",
};

const nf = new Intl.NumberFormat("ko-KR");
function formatNumber(value?: number | null) {
  return typeof value === "number" ? nf.format(value) : "—";
}
function formatDecimal(value?: number | null, digits = 1) {
  return typeof value === "number"
    ? value.toLocaleString("ko-KR", { maximumFractionDigits: digits })
    : "—";
}
// 은/는: pick the topic particle from the last syllable of a Korean name
function withTopic(name: string) {
  const code = name.charCodeAt(name.length - 1) - 0xac00;
  if (code < 0 || code > 11171) return `${name}은(는)`;
  return `${name}${code % 28 ? "은" : "는"}`;
}
function formatMetric(value?: string | number | null, digits = 3) {
  const numeric = typeof value === "number" ? value : typeof value === "string" ? Number(value) : NaN;
  return Number.isFinite(numeric)
    ? numeric.toLocaleString("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits })
    : "—";
}
function scoreNumber(value?: number | null) {
  return typeof value === "number"
    ? Math.round((value <= 1 ? value * 100 : value) * 10) / 10
    : null;
}
function formatScore(value?: number | null) {
  const score = scoreNumber(value);
  return score == null ? "—" : `${score}점`;
}
function formatPercent(value?: number | null) {
  if (typeof value !== "number") return "—";
  return `${Math.round((value <= 1 ? value * 100 : value) * 10) / 10}%`;
}
function formatSigned(value?: number | null, unit = "") {
  if (typeof value !== "number") return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toLocaleString("ko-KR", { maximumFractionDigits: 1 })}${unit}`;
}
function formatYear(item: Hotspot) {
  return item.accident_year == null
    ? "연도 정보 확인 불가"
    : `${Math.round(item.accident_year)}년`;
}
function regionNameFromHotspot(item: Hotspot | null) {
  return item?.sido && item?.sigungu
    ? `${item.sido} ${item.sigungu}`.trim()
    : "";
}
function ratio(value: number | null | undefined, max: number) {
  if (typeof value !== "number" || !Number.isFinite(value) || max <= 0)
    return 0;
  return Math.max(3, Math.min(100, (value / max) * 100));
}
function average(values: Array<number | undefined>) {
  const nums = values.filter(
    (value): value is number =>
      typeof value === "number" && Number.isFinite(value),
  );
  return nums.length
    ? nums.reduce((sum, value) => sum + value, 0) / nums.length
    : null;
}
function percentile(value: number | undefined, values: number[]) {
  if (typeof value !== "number" || values.length === 0) return 0;
  const lower = values.filter((item) => item <= value).length;
  return Math.round((lower / values.length) * 100);
}
async function fetchJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { signal });
  if (!response.ok) throw new Error(`API ${response.status}`);
  return response.json() as Promise<T>;
}

async function postJson<T>(
  path: string,
  body: Record<string, unknown>,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw new Error(`API ${response.status}`);
  return response.json() as Promise<T>;
}

function markerIcon(type: SourceType, selected: boolean) {
  return L.divIcon({
    className: `map-marker ${type === "schoolzone_hotspot" ? "marker-filled" : "marker-ring"} ${selected ? "selected" : ""}`,
    html: `<span style="--marker-color:${sourceMeta[type].color}"></span>`,
    iconSize: [22, 22],
    iconAnchor: [11, 11],
  });
}


function StateBoundaries() {
  const [data, setData] = useState<GeoJSON.FeatureCollection | null>(null);
  const [zoom, setZoom] = useState(7.5);
  const map = useMap();
  useEffect(() => {
    fetch("/korea-states-topo.json")
      .then((response) => response.json())
      .then((topology) => setData(topoToGeoJSON(topology)))
      .catch(() => setData(null));
  }, []);
  useMapEvents({ zoomend: () => setZoom(map.getZoom()) });
  if (!data) return null;
  return (
    <>
      <LeafletGeoJSON
        data={data}
        style={() => ({
          color: "#2FE0C0",
          opacity: 0.28,
          weight: 0.8,
          fillColor: "#03070B",
          fillOpacity: 0,
        })}
      />
      {zoom >= 8 &&
        data.features.map((feature, index) => {
          const geometry = feature.geometry as any;
          const points: number[][] = geometry.coordinates.flat(
            geometry.type === "Polygon" ? 1 : 2,
          );
          const lng =
            points.reduce((sum, point) => sum + point[0], 0) / points.length;
          const lat =
            points.reduce((sum, point) => sum + point[1], 0) / points.length;
          const name =
            feature.properties?.name_short ?? feature.properties?.name ?? "";
          return (
            <Marker
              key={`${feature.properties?.code ?? index}-label`}
              position={[lat, lng]}
              interactive={false}
              icon={L.divIcon({
                className: "state-label",
                html: `<span>${name}</span>`,
                iconSize: [64, 20],
                iconAnchor: [32, 10],
              })}
            />
          );
        })}
    </>
  );
}

function MapViewport({
  selected,
  onCenterChange,
}: {
  selected: Hotspot | null;
  onCenterChange: (center: [number, number]) => void;
}) {
  const map = useMap();
  const fitted = useRef(false);
  useEffect(() => {
    const frame = window.requestAnimationFrame(() => map.invalidateSize());
    return () => window.cancelAnimationFrame(frame);
  }, [map]);
  useEffect(() => {
    if (selected || fitted.current) return;
    map.fitBounds(
      L.latLngBounds([
        [33.05, 125.9],
        [38.6, 129.7],
      ]),
      {
        animate: false,
        // keep the peninsula clear of the floating list (left) and detail (right) panels
        paddingTopLeft: window.innerWidth > 1100 ? [390, 110] : [40, 40],
        paddingBottomRight: window.innerWidth > 1100 ? [410, 60] : [40, 40],
      },
    );
    fitted.current = true;
  }, [map, selected]);
  useEffect(() => {
    if (!selected) return;
    // The detail panel now sits beside the map instead of over it,
    // so the selected point is simply centered.
    map.flyTo(
      [selected.latitude, selected.longitude],
      Math.max(map.getZoom(), 12),
      { duration: 0.55 },
    );
  }, [map, selected]);
  useMapEvents({
    moveend: () => {
      const point = map.getCenter();
      onCenterChange([point.lat, point.lng]);
    },
  });
  return null;
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
function BigMetric({
  label,
  value,
  tone = "default",
}: {
  label: string;
  value: string;
  tone?: "default" | "ai";
}) {
  return (
    <div className={`big-metric ${tone === "ai" ? "ai-tone" : ""}`}>
      <span>{label}</span>
      <strong>
        <CountUpText text={value} />
      </strong>
    </div>
  );
}
function FilterSelect({
  label,
  value,
  options,
  disabled,
  displayValue,
  disabledReason,
  onChange,
}: {
  label: string;
  value: string;
  options: FilterOption[];
  disabled?: boolean;
  displayValue?: string;
  disabledReason?: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className={`filter-select ${disabled ? "is-disabled" : ""}`}>
      <span>{label}</span>
      <select
        value={value}
        disabled={disabled}
        title={disabled ? disabledReason : undefined}
        onChange={(event) => onChange(event.target.value)}
      >
        {displayValue ? <option value={value}>{displayValue}</option> : null}
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function FacilityMeter({
  label,
  value,
  max,
  format,
  nullable = false,
}: {
  label: string;
  value?: number | null;
  max: number;
  format?: "percent" | "score";
  nullable?: boolean;
}) {
  const unavailable = nullable && value == null;
  const display = unavailable
    ? "자료 없음"
    : format === "percent"
      ? formatPercent(value)
      : format === "score"
        ? formatScore(value)
        : formatNumber(value);
  return (
    <div className="facility-meter">
      <div>
        <span>{label}</span>
        <strong>{display}</strong>
      </div>
      <i>
        <b style={{ width: `${unavailable ? 0 : ratio(value, max)}%` }} />
      </i>
    </div>
  );
}

function BklitHorizontalBars({
  data,
  valueKey,
  labelKey,
  valueSuffix = "점",
  fill = AMBER,
  max,
  digits = 2,
}: {
  data: Record<string, unknown>[];
  valueKey: string;
  labelKey: string;
  valueSuffix?: string;
  aspect?: string;
  fill?: string;
  max?: number;
  digits?: number;
}) {
  // Rendered as labelled rows (label · bar · value) so every value stays
  // readable as text; bar length is relative to `max` or the largest value.
  const values = data.map((item) => Number(item[valueKey]) || 0);
  const scaleMax = max ?? Math.max(...values, 0);
  return (
    <ol className="hbar-list">
      {data.map((item, index) => {
        const value = values[index];
        const width = scaleMax > 0 ? Math.max(1, (value / scaleMax) * 100) : 0;
        return (
          <li key={`${String(item[labelKey])}-${index}`}>
            <span className="hbar-label" title={String(item[labelKey])}>{String(item[labelKey])}</span>
            <span className="hbar-track" aria-hidden="true">
              <i style={{ width: `${width}%`, background: fill }} />
            </span>
            <strong className="hbar-value">
              {formatDecimal(item[valueKey] as number, digits)}
              {valueSuffix}
            </strong>
          </li>
        );
      })}
    </ol>
  );
}

function FacilityRing({
  facility,
  candidate,
}: {
  facility: Region | null;
  candidate: ImprovementCandidate | null;
}) {
  if (!facility) return null;
  const data = [
    {
      label: "인프라 공급수준",
      value: scoreNumber(facility.infrastructure_score),
      maxValue: 100,
      color: REGION_MINT,
    },
    {
      label: "시설 부족도",
      value: scoreNumber(candidate?.infrastructure_deficiency_score),
      maxValue: 100,
      color: REGION_SLATE,
    },
  ].filter(
    (
      item,
    ): item is {
      label: string;
      value: number;
      maxValue: number;
      color: string;
    } => typeof item.value === "number",
  );
  if (data.length === 0) return null;
  return (
    <div className="facility-ring-wrap">
      <RingChart
        data={data}
        size={188}
        strokeWidth={11}
        ringGap={9}
        baseInnerRadius={50}
        enterStaggerScale={0.75}
      >
        {data.map((_, index) => (
          <Ring key={index} index={index} showGlow={false} />
        ))}
        <RingCenter
          suffix="점"
          showTotal={false}
          defaultLabel="시설 지표"
          valueClassName="ring-center-value"
          labelClassName="ring-center-label"
        />
      </RingChart>
      <div className="ring-legend">
        {data.map((item) => (
          <span key={item.label}>
            <i style={{ background: item.color }} />
            {item.label}
            <strong>{formatDecimal(item.value)}점</strong>
          </span>
        ))}
      </div>
    </div>
  );
}

function ScatterRiskInfrastructure({
  items,
  large = false,
}: {
  items: ImprovementCandidate[];
  large?: boolean;
}) {
  const data = items
    .filter(
      (item) =>
        typeof item.infrastructure_score === "number" &&
        typeof item.predicted_high_risk_probability_2026 === "number",
    )
    .map((item) => ({
      region: item.sido_sgg_nm,
      infrastructure: scoreNumber(item.infrastructure_score) ?? 0,
      aiScore: scoreNumber(item.predicted_high_risk_probability_2026) ?? 0,
    }));
  if (data.length === 0) return null;
  return (
    <div className={`bklit-scatter ${large ? "large" : ""}`}>
      <ScatterChart
        data={data}
        xDataKey="infrastructure"
        aspectRatio={large ? "2.5 / 1" : "2 / 1"}
        margin={{ top: 24, right: 28, bottom: 28, left: 34 }}
        animationDuration={700}
      >
        <Grid strokeOpacity={0.16} vertical />
        <Scatter dataKey="aiScore" fill={AI_BLUE} radius={large ? 6 : 5} />
      </ScatterChart>
      <div className="scatter-axis-note">
        <span>인프라 공급수준 →</span>
        <span>↑ AI 상대적 고위험 점수</span>
      </div>
    </div>
  );
}

function PageHeading({
  kicker,
  lead,
  title,
  copy,
  warning,
}: {
  kicker?: string;
  lead?: string;
  title: string;
  copy: string;
  warning?: string;
}) {
  return (
    <header className="page-heading">
      <div className="page-heading-title">
        {kicker ? <span className="kicker">{kicker}</span> : null}
        <h1>
          {lead ? (
            <>
              <span className="h-lead">{lead}</span>
              <br />
            </>
          ) : null}
          {title}
        </h1>
      </div>
      <div className="page-heading-copy">
        <p>{copy}</p>
        {warning ? (
          <p className="scope-warning" role="note">
            {warning}
          </p>
        ) : null}
      </div>
    </header>
  );
}

function ScopeTag({ scope }: { scope: "229" | "49" | "1803" }) {
  const label =
    scope === "229"
      ? "AI 분석 229개 지역"
      : scope === "49"
        ? "시설분석 49개 지역"
        : "선정지점 1,803개";
  return <span className={`scope-tag scope-${scope}`}>{label}</span>;
}

function App() {
  const [view, setView] = useState<View>("home");
  const [regionSeed, setRegionSeed] = useState("");
  const [mapSeed, setMapSeed] = useState("");
  const openRegion = useCallback((regionName: string) => {
    setRegionSeed(regionName);
    setView("region");
  }, []);
  const openMapRegion = useCallback((regionName: string) => {
    setMapSeed(regionName);
    setView("map");
  }, []);
  const navItems: Array<{ id: View; label: string; number: string; icon: string }> = [
    { id: "map", label: "현황지도", number: "01", icon: "map" },
    { id: "region", label: "지역분석", number: "02", icon: "grid" },
    { id: "ai", label: "AI 분석", number: "03", icon: "spark" },
    { id: "guide", label: "데이터안내", number: "04", icon: "info" },
  ];
  useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [view]);
  return (
    <div className={`app-shell view-${view}`}>
      <header className="site-header">
        <div className="site-header-inner">
          <button
            className="service-brand"
            onClick={() => setView("home")}
            aria-label="SaveSpot 첫 화면으로 이동"
          >
            <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
              <defs>
                <clipPath id="brand-mark-clip">
                  <circle cx="12" cy="12" r="10" />
                </clipPath>
              </defs>
              <g clipPath="url(#brand-mark-clip)" fill="currentColor">
                <rect x="2" y="2" width="20" height="3.2" />
                <rect x="2" y="7.6" width="20" height="3.2" />
                <rect x="2" y="13.2" width="20" height="3.2" />
                <rect x="2" y="18.8" width="20" height="3.2" />
              </g>
            </svg>
            <span className="brand-word">SaveSpot</span>
          </button>
          <nav className="service-nav" aria-label="주요 메뉴">
            {navItems.map((item) => (
              <button
                key={item.id}
                className={view === item.id ? "active" : ""}
                aria-current={view === item.id ? "page" : undefined}
                onClick={() => setView(item.id)}
              >
                {item.label}
              </button>
            ))}
          </nav>
          {view === "home" ? null : (
          <div className="header-coverage" aria-label="데이터 범위">
            <span>
              선정지점 <b>1,803</b>
            </span>
            <span>
              AI 분석 <b>229</b>개 지역
            </span>
            <span>
              시설분석 <b>49</b>개 지역
            </span>
            <em>읽기 전용</em>
          </div>
          )}
        </div>
      </header>
      <main className="service-main">
        {view === "home" ? (
          <Home
            onNavigate={setView}
            onOpenRegion={openRegion}
            onOpenMapRegion={openMapRegion}
          />
        ) : view === "map" ? (
          <MapView onOpenRegion={openRegion} initialRegion={mapSeed} />
        ) : view === "region" ? (
          <RegionAnalysis initialRegion={regionSeed} />
        ) : view === "ai" ? (
          <AiAnalysis onOpenRegion={openRegion} />
        ) : (
          <Guide />
        )}
      </main>
    </div>
  );
}

type AssistantAction = {
  type: "open_region" | "open_map" | "open_ai" | "open_guide" | string;
  label: string;
  region?: string | null;
};

type AskAnswer = {
  answer: string;
  intent: string;
  region?: string | null;
  sources?: Array<string | null>;
  actions?: AssistantAction[];
  metadata?: Record<string, unknown> | null;
  question?: string;
  error?: boolean;
};

function isSafeUrl(value: string) {
  try {
    const url = new URL(value, window.location.origin);
    return ["http:", "https:", "mailto:"].includes(url.protocol);
  } catch {
    return false;
  }
}

function renderInlineMarkdown(text: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const pattern = /(\*\*([^*]+)\*\*)|(\[([^\]]+)\]\(([^)]+)\))/g;
  let cursor = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) {
      nodes.push(text.slice(cursor, match.index));
    }

    if (match[2]) {
      nodes.push(<strong key={`strong-${match.index}`}>{match[2]}</strong>);
    } else if (match[4] && match[5]) {
      const href = match[5].trim();
      nodes.push(
        isSafeUrl(href) ? (
          <a key={`link-${match.index}`} href={href} target="_blank" rel="noreferrer">
            {match[4]}
          </a>
        ) : (
          <span key={`link-text-${match.index}`}>{match[4]}</span>
        ),
      );
    }

    cursor = pattern.lastIndex;
  }

  if (cursor < text.length) {
    nodes.push(text.slice(cursor));
  }
  return nodes;
}

function renderMarkdownLine(text: string, keyPrefix: string) {
  const lines = text.split("\n");
  return lines.flatMap((line, index) => [
    <span key={`${keyPrefix}-line-${index}`}>{renderInlineMarkdown(line)}</span>,
    index < lines.length - 1 ? <br key={`${keyPrefix}-br-${index}`} /> : null,
  ]);
}

function splitMarkdownTableRow(line: string) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function isMarkdownTableSeparator(line: string) {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
}

function MarkdownAnswer({ text }: { text: string }) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let index = 0;

  while (index < lines.length) {
    if (!lines[index].trim()) {
      index += 1;
      continue;
    }

    if (
      lines[index].trim().startsWith("|") &&
      index + 1 < lines.length &&
      isMarkdownTableSeparator(lines[index + 1])
    ) {
      const header = splitMarkdownTableRow(lines[index]);
      index += 2;
      const rows: string[][] = [];
      while (index < lines.length && lines[index].trim().startsWith("|")) {
        rows.push(splitMarkdownTableRow(lines[index]));
        index += 1;
      }
      blocks.push(
        <div className="ans-table-wrap" key={`table-${blocks.length}`}>
          <table>
            <thead>
              <tr>
                {header.map((cell, cellIndex) => (
                  <th key={`head-${cellIndex}`}>{renderInlineMarkdown(cell)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr key={`row-${rowIndex}`}>
                  {row.map((cell, cellIndex) => (
                    <td key={`cell-${rowIndex}-${cellIndex}`}>{renderInlineMarkdown(cell)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      continue;
    }

    if (/^\s*-\s+/.test(lines[index])) {
      const items: string[] = [];
      while (index < lines.length && /^\s*-\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*-\s+/, ""));
        index += 1;
      }
      blocks.push(
        <ul key={`list-${blocks.length}`}>
          {items.map((item, itemIndex) => (
            <li key={`item-${itemIndex}`}>{renderInlineMarkdown(item)}</li>
          ))}
        </ul>,
      );
      continue;
    }

    const paragraph: string[] = [];
    while (
      index < lines.length &&
      lines[index].trim() &&
      !/^\s*-\s+/.test(lines[index]) &&
      !(
        lines[index].trim().startsWith("|") &&
        index + 1 < lines.length &&
        isMarkdownTableSeparator(lines[index + 1])
      )
    ) {
      paragraph.push(lines[index]);
      index += 1;
    }
    blocks.push(<p key={`paragraph-${blocks.length}`}>{renderMarkdownLine(paragraph.join("\n"), `p-${blocks.length}`)}</p>);
  }

  return <div className="ans-text markdown-answer">{blocks}</div>;
}

function AnswerCard({
  answer,
  onClose,
  onAction,
}: {
  answer: AskAnswer;
  onClose: () => void;
  onAction: (action: AssistantAction) => void;
}) {
  const sources = (answer.sources ?? []).filter(Boolean);
  return (
    <section className="ask-card ask-answer" aria-live="polite">
      <div className="ans-q">
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="#F2B705"
          strokeWidth="1.8"
          aria-hidden="true"
        >
          <path d="M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8z" />
        </svg>
        <span>{answer.question ?? "AI 분석 답변"}</span>
        <button type="button" onClick={onClose} aria-label="답변 닫고 새 질문">
          ×
        </button>
      </div>
      {answer.region ? <h2 className="ans-title">{answer.region}</h2> : null}
      <MarkdownAnswer text={answer.answer} />
      <div className="ans-tags">
        <span className="scope-tag scope-229">LLM + 프로젝트 데이터</span>
        {answer.intent ? <span className="scope-tag scope-49">{answer.intent}</span> : null}
        {sources.length ? <em>출처 · {sources.join(" · ")}</em> : null}
      </div>
      <p className="ans-note">
        AI 점수는 실제 사고 발생 가능성 수치가 아니며, 시설 미제공은 시설이 0개라는 뜻이 아닙니다.
      </p>
      <div className="ans-actions">
        <button type="button" className="ask-shuffle" onClick={onClose}>
          새 질문
        </button>
        {(answer.actions ?? []).map((action) => (
          <button
            key={`${action.type}-${action.label}-${action.region ?? ""}`}
            type="button"
            className="ask-go"
            onClick={() => onAction(action)}
          >
            {action.label} →
          </button>
        ))}
      </div>
    </section>
  );
}
const ASK_SUGGESTIONS = [
  "동해시의 어린이 사고 현황과 안전시설은 어느 정도인지 알려줘",
  "AI 점수가 높은 지역 5곳은 어디야?",
  "원주시 주변 어린이 사고다발지점을 보여줘",
  "AI 점수는 무슨 뜻이야?",
];

// Counts up from 0 to the real API value once it arrives; shows "—" if the
// value could not be loaded. Screen readers get the final number directly.
function CountUp({ value }: { value: number | null }) {
  const [shown, setShown] = useState(0);
  useEffect(() => {
    if (value == null) return;
    const frame = window.requestAnimationFrame(() => setShown(value));
    return () => window.cancelAnimationFrame(frame);
  }, [value]);
  if (value == null) return <>—</>;
  return (
    <>
      <span className="sr-only">{formatNumber(value)}</span>
      <span aria-hidden="true">
        <NumberFlow
          value={shown}
          locales="ko-KR"
          transformTiming={{ duration: 800, easing: "cubic-bezier(.22,1,.36,1)" }}
          spinTiming={{ duration: 800, easing: "cubic-bezier(.22,1,.36,1)" }}
        />
      </span>
    </>
  );
}

// Landing view. Every number on this page comes from the existing read-only API;
// nothing is hard-coded or estimated here.
function Home({
  onNavigate,
  onOpenRegion,
  onOpenMapRegion,
}: {
  onNavigate: (view: View) => void;
  onOpenRegion: (region: string) => void;
  onOpenMapRegion: (region: string) => void;
}) {
  const [stats, setStats] = useState<{
    schoolzone: number | null;
    pedestrian: number | null;
    aiTotal: number | null;
    facilityTotal: number | null;
  }>({ schoolzone: null, pedestrian: null, aiTotal: null, facilityTotal: null });
  const [aiRegions, setAiRegions] = useState<AiRiskRegion[]>([]);
  const [regionInput, setRegionInput] = useState("");
  const [suggestion, setSuggestion] = useState(0);
  const [answer, setAnswer] = useState<AskAnswer | null>(null);
  const [asking, setAsking] = useState(false);
  // server switch (AI_ASSISTANT_ENABLED) — read once; the build switch wins
  const [assistantEnabled, setAssistantEnabled] = useState(ASSISTANT_BUILD_ENABLED);
  useEffect(() => {
    if (!ASSISTANT_BUILD_ENABLED) return;
    const controller = new AbortController();
    fetchJson<{ enabled?: boolean }>("/api/assistant/status", controller.signal)
      .then((status) => setAssistantEnabled(status.enabled !== false))
      .catch(() => undefined);
    return () => controller.abort();
  }, []);
  const ask = async (text: string) => {
    const question = text.trim();
    if (!question || !assistantEnabled) return;
    setAsking(true);
    setAnswer(null);
    try {
      const data = await postJson<AskAnswer>("/api/assistant/query", {
        question,
      });
      setAnswer({ ...data, question });
    } catch {
      setAnswer({
        question,
        answer:
          "AI 어시스턴트 API에 연결하지 못했습니다. FastAPI 서버 상태와 AI_ASSISTANT_ENABLED 설정을 확인해 주세요.",
        intent: "error",
        region: null,
        sources: [],
        actions: [],
        error: true,
      });
    } finally {
      setAsking(false);
    }
  };
  const handleAssistantAction = (action: AssistantAction) => {
    if (action.type === "open_region" && action.region) {
      onOpenRegion(action.region);
      return;
    }
    if (action.type === "open_map" && action.region) {
      onOpenMapRegion(action.region);
      return;
    }
    if (action.type === "open_ai") {
      onNavigate("ai");
      return;
    }
    if (action.type === "open_guide") {
      onNavigate("guide");
    }
  };
  const [loadError, setLoadError] = useState(false);
  // Real TAAS points for the region the landing camera flies into (Busan).
  const [focusHotspots, setFocusHotspots] = useState<{ items: LandingHotspot[]; total: number | null }>({ items: [], total: null });

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    // Each figure is shown as soon as its own request returns, so the fast
    // count queries are not held back by the larger 229-region request.
    // One retry so a single dropped request doesn't leave a figure as "—".
    const withRetry = <T,>(path: string) =>
      fetchJson<T>(path, signal).catch((error) => {
        if (signal.aborted) throw error;
        return new Promise<T>((resolve, reject) =>
          window.setTimeout(() => fetchJson<T>(path, signal).then(resolve, reject), 600),
        );
      });
    const fail = () => {
      if (!signal.aborted) setLoadError(true);
    };
    withRetry<HotspotResponse>("/api/hotspots?source_type=schoolzone_hotspot&limit=1")
      .then((data) => setStats((cur) => ({ ...cur, schoolzone: data.pagination.total })))
      .catch(fail);
    withRetry<HotspotResponse>("/api/hotspots?source_type=child_pedestrian_hotspot&limit=1")
      .then((data) => setStats((cur) => ({ ...cur, pedestrian: data.pagination.total })))
      .catch(fail);
    withRetry<RegionResponse>("/api/facilities?limit=1")
      .then((data) => setStats((cur) => ({ ...cur, facilityTotal: data.pagination.total })))
      .catch(fail);
    withRetry<AiRiskResponse>("/api/ai/high-risk?limit=300")
      .then((data) => {
        setStats((cur) => ({ ...cur, aiTotal: data.pagination?.total ?? data.items.length }));
        setAiRegions(data.items);
      })
      .catch(fail);
    withRetry<HotspotResponse>(`/api/hotspots?sido=${encodeURIComponent("부산광역시")}&limit=500`)
      .then((data) =>
        setFocusHotspots({
          total: data.pagination.total,
          items: data.items
            .filter((h) => typeof h.latitude === "number" && typeof h.longitude === "number")
            .map((h) => ({ lat: h.latitude as number, lon: h.longitude as number, type: h.source_type })),
        }),
      )
      .catch(() => {});
    return () => controller.abort();
  }, []);

  const hotspotTotal =
    stats.schoolzone != null && stats.pedestrian != null
      ? stats.schoolzone + stats.pedestrian
      : null;

  const landingRegions = useMemo(
    () =>
      aiRegions
        .filter((item) => typeof item.predicted_high_risk_probability_2026 === "number")
        .map((item) => ({
          name: item.sido_sgg_nm,
          rank: item.risk_rank_2026,
          score: scoreNumber(item.predicted_high_risk_probability_2026) ?? 0,
        })),
    [aiRegions],
  );

  const askBlock = answer ? (
    <AnswerCard answer={answer} onClose={() => setAnswer(null)} onAction={handleAssistantAction} />
  ) : (
    <form
      className={`ask-card${assistantEnabled ? "" : " is-disabled"}`}
      onSubmit={(event) => {
        event.preventDefault();
        ask(regionInput || ASK_SUGGESTIONS[suggestion]);
      }}
    >
      <label htmlFor="ask-input" className="sr-only">
        AI에게 질문하기
      </label>
      <textarea
        id="ask-input"
        rows={2}
        value={regionInput}
        placeholder={ASK_SUGGESTIONS[suggestion]}
        disabled={!assistantEnabled}
        onChange={(event) => setRegionInput(event.target.value)}
      />
      {assistantEnabled ? null : (
        <p className="ask-disabled-note" role="status">
          {ASSISTANT_DISABLED_MESSAGE}
        </p>
      )}
      <div className="ask-row">
        <button
          type="button"
          className="ask-shuffle"
          onClick={() => setSuggestion((i) => (i + 1) % ASK_SUGGESTIONS.length)}
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <path d="M16 3h5v5M4 20 21 3M21 16v5h-5M15 15l6 6M4 4l5 5" />
          </svg>
          다른 질문 추천
        </button>
        <button type="submit" className="ask-go" disabled={!assistantEnabled || asking}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" aria-hidden="true">
            <path d="M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8z" />
          </svg>
          {asking ? "찾는 중" : "물어보기"}
        </button>
      </div>
    </form>
  );

  return (
    <div className="home-view home-scroll">
      <ScrollLanding
        regions={landingRegions}
        hotspots={focusHotspots.items}
        hotspotTotal={hotspotTotal}
        focusHotspotTotal={focusHotspots.total}
        facilityTotal={stats.facilityTotal}
        ask={askBlock}
        onNavigate={onNavigate}
        onOpenRegion={onOpenRegion}
      />
      {loadError ? (
        <p className="sl-load-error" role="status">일부 데이터를 불러오지 못했습니다. 백엔드 API 상태를 확인해 주세요.</p>
      ) : null}
    </div>
  );
}

function MapView({
  onOpenRegion,
  initialRegion,
}: {
  onOpenRegion: (region: string) => void;
  initialRegion?: string;
}) {
  const [items, setItems] = useState<Hotspot[]>([]);
  const [total, setTotal] = useState(0);
  const [typeTotals, setTypeTotals] = useState({
    schoolzone_hotspot: 646,
    child_pedestrian_hotspot: 1157,
  });
  const [selected, setSelected] = useState<Hotspot | null>(null);
  const [nearby, setNearby] = useState<NearbyItem[]>([]);
  const [filters, setFilters] = useState({
    source_type: "all" as SourceType | "all",
    accident_year: "all",
    sido: "all",
    sigungu: "all",
  });
  const [radius, setRadius] = useState(1000);
  const [mapCenter, setMapCenter] = useState<[number, number]>(DEFAULT_CENTER);
  const [loading, setLoading] = useState(true);
  const [nearbyLoading, setNearbyLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadHotspots = useCallback(
    async (nextFilters = filters, signal?: AbortSignal) => {
      setLoading(true);
      setError(null);
      try {
        const baseParams = new URLSearchParams();
        if (nextFilters.sido !== "all")
          baseParams.set("sido", nextFilters.sido);
        if (nextFilters.sigungu !== "all")
          baseParams.set("sigungu", nextFilters.sigungu);
        let data: HotspotResponse;
        if (nextFilters.source_type === "all") {
          const makeTypeParams = (
            sourceType: SourceType,
            includeYear: boolean,
          ) => {
            const params = new URLSearchParams(baseParams);
            params.set("source_type", sourceType);
            params.set(
              "limit",
              nextFilters.accident_year !== "all" ? "500" : "250",
            );
            if (includeYear && nextFilters.accident_year !== "all")
              params.set("accident_year", nextFilters.accident_year);
            return params;
          };
          if (nextFilters.accident_year !== "all") {
            const pedestrian = await fetchJson<HotspotResponse>(
              `/api/hotspots?${makeTypeParams("child_pedestrian_hotspot", true)}`,
              signal,
            );
            data = {
              items: pedestrian.items,
              pagination: { ...pedestrian.pagination, limit: 500 },
            };
            setTypeTotals((current) => ({
              ...current,
              child_pedestrian_hotspot: pedestrian.pagination.total,
            }));
          } else {
            const [schoolzone, pedestrian] = await Promise.all([
              fetchJson<HotspotResponse>(
                `/api/hotspots?${makeTypeParams("schoolzone_hotspot", false)}`,
                signal,
              ),
              fetchJson<HotspotResponse>(
                `/api/hotspots?${makeTypeParams("child_pedestrian_hotspot", false)}`,
                signal,
              ),
            ]);
            setTypeTotals({
              schoolzone_hotspot: schoolzone.pagination.total,
              child_pedestrian_hotspot: pedestrian.pagination.total,
            });
            data = {
              items: [...schoolzone.items, ...pedestrian.items],
              pagination: {
                limit: 500,
                offset: 0,
                count: schoolzone.items.length + pedestrian.items.length,
                total:
                  schoolzone.pagination.total + pedestrian.pagination.total,
              },
            };
          }
        } else {
          const params = new URLSearchParams(baseParams);
          params.set("source_type", nextFilters.source_type);
          params.set("limit", "500");
          if (
            nextFilters.source_type === "child_pedestrian_hotspot" &&
            nextFilters.accident_year !== "all"
          )
            params.set("accident_year", nextFilters.accident_year);
          data = await fetchJson<HotspotResponse>(
            `/api/hotspots?${params}`,
            signal,
          );
          setTypeTotals((current) => ({
            ...current,
            [nextFilters.source_type]: data.pagination.total,
          }));
        }
        setItems(data.items);
        setTotal(data.pagination.total);
        setSelected((current) =>
          current &&
          data.items.some((item) => item.hotspot_id === current.hotspot_id)
            ? current
            : null,
        );
      } catch (cause) {
        if ((cause as Error).name !== "AbortError")
          setError(
            "API 연결에 실패했습니다. FastAPI 서버가 실행 중인지 확인해 주세요.",
          );
      } finally {
        if (!signal?.aborted) setLoading(false);
      }
    },
    [filters],
  );

  useEffect(() => {
    const controller = new AbortController();
    loadHotspots(filters, controller.signal);
    return () => controller.abort();
  }, [filters, loadHotspots]);
  useEffect(() => {
    if (!initialRegion) return;
    const [sido, ...sigunguParts] = initialRegion.split(" ");
    const sigungu = sigunguParts.join(" ");
    if (!sido || !sigungu) return;
    setFilters((current) => ({
      ...current,
      sido,
      sigungu,
    }));
    setSelected(null);
    setNearby([]);
  }, [initialRegion]);
  const options = useMemo(() => {
    const sidos = [
      ...new Set(items.map((item) => item.sido).filter(Boolean)),
    ].sort((a, b) => (a ?? "").localeCompare(b ?? "", "ko"));
    const sigungus =
      filters.sido === "all"
        ? []
        : [
            ...new Set(
              items
                .filter((item) => item.sido === filters.sido)
                .map((item) => item.sigungu)
                .filter(Boolean),
            ),
          ].sort((a, b) => (a ?? "").localeCompare(b ?? "", "ko"));
    return { sidos, sigungus };
  }, [filters.sido, items]);
  const loadNearby = useCallback(async () => {
    const target = selected
      ? [selected.latitude, selected.longitude]
      : mapCenter;
    setNearbyLoading(true);
    try {
      const params = new URLSearchParams({
        lat: String(target[0]),
        lng: String(target[1]),
        radius_m: String(radius),
        limit: "50",
      });
      if (filters.source_type !== "all")
        params.set("source_type", filters.source_type);
      const data = await fetchJson<NearbyResponse>(
        `/api/hotspots/nearby?${params}`,
      );
      setNearby(data.items);
    } finally {
      setNearbyLoading(false);
    }
  }, [filters.source_type, mapCenter, radius, selected]);
  const changeFilter = (key: keyof typeof filters, value: string) =>
    setFilters((current) => {
      const next = { ...current, [key]: value } as typeof filters;
      if (key === "source_type") next.accident_year = "all";
      if (key === "sido") next.sigungu = "all";
      return next;
    });
  const resetFilters = () => {
    setFilters({
      source_type: "all",
      accident_year: "all",
      sido: "all",
      sigungu: "all",
    });
    setSelected(null);
    setNearby([]);
  };
  const sourceOptions: FilterOption[] = [
    { value: "all", label: "전체 선정지점" },
    { value: "schoolzone_hotspot", label: "어린이보호구역" },
    { value: "child_pedestrian_hotspot", label: "보행어린이" },
  ];
  const yearDisabled = filters.source_type === "schoolzone_hotspot";
  const yearOptions: FilterOption[] = [
    { value: "all", label: "전체 연도" },
    ...Array.from({ length: 14 }, (_, index) => 2025 - index).map((year) => ({
      value: String(year),
      label: `${year}`,
    })),
  ];
  const sidoOptions: FilterOption[] = [
    { value: "all", label: "전국" },
    ...options.sidos.map((sido) => ({ value: sido ?? "", label: sido ?? "" })),
  ];
  const sigunguOptions: FilterOption[] = [
    {
      value: "all",
      label: filters.sido === "all" ? "시도 선택 후 표시" : "전체",
    },
    ...options.sigungus.map((sigungu) => ({
      value: sigungu ?? "",
      label: sigungu ?? "",
    })),
  ];

  return (
    <div className="map-page">
      <section className="map-pagebar" aria-label="지도 조건">
        <div className="map-pagebar-title">
          <span className="kicker">현황지도 · 공식 사고다발지역 선정지점</span>
          <h1>어린이 사고다발지역 현황</h1>
        </div>
        <div className="map-toolbar" aria-label="지도 필터">
          <FilterSelect
            label="유형"
            value={filters.source_type}
            options={sourceOptions}
            onChange={(value) => changeFilter("source_type", value)}
          />
          <FilterSelect
            label="연도"
            value={yearDisabled ? "disabled" : filters.accident_year}
            displayValue={yearDisabled ? "연도 미제공" : undefined}
            disabled={yearDisabled}
            disabledReason="이 데이터는 원본에 연도 필드가 제공되지 않습니다"
            options={
              yearDisabled
                ? [{ value: "disabled", label: "연도 미제공" }]
                : yearOptions
            }
            onChange={(value) => changeFilter("accident_year", value)}
          />
          <FilterSelect
            label="시도"
            value={filters.sido}
            options={sidoOptions}
            onChange={(value) => changeFilter("sido", value)}
          />
          <FilterSelect
            label="시군구"
            value={filters.sigungu}
            disabled={filters.sido === "all"}
            displayValue={
              filters.sido === "all" ? "시도 선택 후 이용" : undefined
            }
            disabledReason="시도를 먼저 선택하면 해당 시군구만 표시됩니다"
            options={sigunguOptions}
            onChange={(value) => changeFilter("sigungu", value)}
          />
          <button className="toolbar-reset" onClick={resetFilters}>
            초기화
          </button>
        </div>
        <div className="map-pagebar-status" aria-live="polite">
          <span>
            선정지점 <b>{loading ? "—" : formatNumber(total)}</b>
          </span>
          <span>
            보호구역 <b>{formatNumber(typeTotals.schoolzone_hotspot)}</b>
          </span>
          <span>
            보행어린이{" "}
            <b>{formatNumber(typeTotals.child_pedestrian_hotspot)}</b>
          </span>
        </div>
      </section>
      {filters.source_type === "all" && filters.accident_year !== "all" ? (
        <p className="year-filter-note">
          연도 선택 시 연도 정보가 있는 보행어린이 데이터에 적용됩니다.
        </p>
      ) : null}

      <section className="map-workspace">
        <HotspotList
          items={items}
          total={total}
          selectedId={selected ? `${selected.source_type}-${selected.hotspot_id}` : null}
          onSelect={setSelected}
        />

        <div className="map-stage">
          <div className="map-frame">
            {loading ? (
              <div className="map-overlay">
                <span className="loader" />
                지점을 불러오는 중
              </div>
            ) : null}
            {error ? (
              <div className="map-overlay error-state">
                <strong>API 연결 실패</strong>
                <span>{error}</span>
              </div>
            ) : null}
            <MapContainer
              className="leaflet-map"
              center={DEFAULT_CENTER}
              zoom={7.5}
              zoomControl
              scrollWheelZoom
            >
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                subdomains="abc"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>'
                maxZoom={19}
              />
              <DotLand />
              <StateBoundaries />
              <MapViewport selected={selected} onCenterChange={setMapCenter} />
              {items.map((item) => (
                <Marker
                  key={`${item.source_type}-${item.hotspot_id}`}
                  position={[item.latitude, item.longitude]}
                  icon={markerIcon(
                    item.source_type,
                    selected?.hotspot_id === item.hotspot_id,
                  )}
                  eventHandlers={{ click: () => setSelected(item) }}
                >
                  <Tooltip direction="top" offset={[0, -10]}>
                    {item.location_name ?? "사고다발지역 선정지점"}
                  </Tooltip>
                </Marker>
              ))}
            </MapContainer>
            <span className="result-status">
              전체 {formatNumber(total)}개 중 지도에는 최대{" "}
              {formatNumber(items.length)}개 표시
            </span>
            <div className="map-legend" aria-label="범례">
              <span>
                <i className="legend-marker square" /> 어린이보호구역
              </span>
              <span>
                <i className="legend-marker diamond" /> 보행어린이
              </span>
            </div>
          </div>
        </div>

        {selected ? (
          <HotspotDetail
            selected={selected}
            nearby={nearby}
            radius={radius}
            nearbyLoading={nearbyLoading}
            onRadiusChange={setRadius}
            onNearby={loadNearby}
            onClose={() => setSelected(null)}
            onOpenRegion={onOpenRegion}
            onSelectNearby={setSelected}
          />
        ) : (
          <aside className="detail-panel detail-empty" aria-label="지도 안내">
            <span className="kicker">데이터 해석</span>
            <h2>지점을 선택하세요</h2>
            <p>
              지도 위 표시나 왼쪽 목록에서 지점을 고르면 사고·사상자 정보와
              주변 사고다발지점을 확인할 수 있습니다.
            </p>
            <dl className="summary-stats">
              <div>
                <dt>선정지점 (현재 조건)</dt>
                <dd>{loading ? "—" : formatNumber(total)}</dd>
              </div>
              <div>
                <dt>어린이보호구역</dt>
                <dd>{formatNumber(typeTotals.schoolzone_hotspot)}</dd>
              </div>
              <div>
                <dt>보행어린이</dt>
                <dd>{formatNumber(typeTotals.child_pedestrian_hotspot)}</dd>
              </div>
            </dl>
            <div className="meaning-note">
              <strong>공식 기준에 따라 선정된 사고다발지역 지점입니다.</strong>
              <span>
                선정지점 수는 전체 사고건수가 아니며, 지역 전체 평가나 학교 단위
                평가를 의미하지 않습니다.
              </span>
            </div>
          </aside>
        )}
      </section>
    </div>
  );
}

function HotspotList({
  items,
  total,
  selectedId,
  onSelect,
}: {
  items: Hotspot[];
  total: number;
  selectedId: string | null;
  onSelect: (item: Hotspot) => void;
}) {
  const [query, setQuery] = useState("");
  const visible = query
    ? items.filter((item) =>
        [item.location_name, item.sido, item.sigungu]
          .filter(Boolean)
          .join(" ")
          .includes(query.trim()),
      )
    : items;
  return (
    <aside className="hotspot-list-panel" aria-label="선정지점 목록">
      <div className="list-search">
        <label htmlFor="hotspot-search">지점명·지역 검색</label>
        <input
          id="hotspot-search"
          type="search"
          value={query}
          placeholder="예: 원주시"
          onChange={(event) => setQuery(event.target.value)}
        />
      </div>
      <div className="list-head">
        <h2>지도에 표시된 선정지점</h2>
        <span>
          {formatNumber(visible.length)} / {formatNumber(total)}
        </span>
      </div>
      <p className="list-note">목록 순서는 순위가 아닙니다.</p>
      <ul className="hotspot-list">
        {visible.map((item) => {
          const id = `${item.source_type}-${item.hotspot_id}`;
          return (
            <li key={id}>
              <button
                onClick={() => onSelect(item)}
                className={`hotspot-list-row ${selectedId === id ? "selected" : ""}`}
                aria-pressed={selectedId === id}
              >
                <i
                  className={`legend-marker ${item.source_type === "schoolzone_hotspot" ? "square" : "diamond"}`}
                  aria-hidden="true"
                />
                <span>
                  <strong>{item.location_name ?? "사고다발지역 선정지점"}</strong>
                  <em>
                    {sourceMeta[item.source_type].short} · {formatYear(item)}
                  </em>
                </span>
                <b>사고 {formatNumber(item.accident_count)}건</b>
              </button>
            </li>
          );
        })}
        {visible.length === 0 ? (
          <li className="list-empty">조건에 맞는 지점이 없습니다.</li>
        ) : null}
      </ul>
    </aside>
  );
}

function HotspotDetail({
  selected,
  nearby,
  radius,
  nearbyLoading,
  onRadiusChange,
  onNearby,
  onClose,
  onOpenRegion,
  onSelectNearby,
}: {
  selected: Hotspot;
  nearby: NearbyItem[];
  radius: number;
  nearbyLoading: boolean;
  onRadiusChange: (radius: number) => void;
  onNearby: () => void;
  onClose: () => void;
  onOpenRegion: (region: string) => void;
  onSelectNearby: (item: Hotspot) => void;
}) {
  const regionName = regionNameFromHotspot(selected);
  return (
    <aside className="detail-panel" aria-label="지점 상세">
      <div className="detail-header">
        <span
          className={`detail-type type-${selected.source_type}`}
        >
          {sourceMeta[selected.source_type].label}
        </span>
        <button
          className="close-detail"
          onClick={onClose}
          aria-label="상세 닫기"
        >
          ×
        </button>
      </div>
      <h2 className="detail-title">
        {selected.location_name ?? "사고다발지역 선정지점"}
      </h2>
      <p className="detail-location">
        {[selected.sido, selected.sigungu].filter(Boolean).join(" ")}
      </p>
      <div className="detail-metrics">
        <Metric
          label="사고건수"
          value={`${formatNumber(selected.accident_count)}건`}
        />
        <Metric
          label="사상자수"
          value={`${formatNumber(selected.casualty_count)}명`}
        />
        <Metric label="사고연도" value={formatYear(selected)} />
        <Metric
          label="연도 확인 상태"
          value={selected.year_verification_status ?? "알 수 없음"}
        />
      </div>
      <div className="injury-list">
        <span>사상자 상세</span>
        <Metric
          label="사망"
          value={`${formatNumber(selected.death_count)}명`}
        />
        <Metric
          label="중상"
          value={`${formatNumber(selected.serious_injury_count)}명`}
        />
        <Metric
          label="경상"
          value={`${formatNumber(selected.minor_injury_count)}명`}
        />
        <Metric
          label="부상신고"
          value={`${formatNumber(selected.injury_report_count)}명`}
        />
      </div>
      <div className="meaning-note">
        <strong>공식 기준에 따라 선정된 사고다발지역 지점입니다.</strong>
        <span>지역 전체 평가나 학교 단위 평가를 의미하지 않습니다.</span>
      </div>
      <div className="nearby-action">
        <div>
          <strong>주변 사고다발지점</strong>
          <span>선택 지점 기준 거리순 조회</span>
        </div>
        <label className="sr-only" htmlFor="nearby-radius">
          조회 반경
        </label>
        <select
          id="nearby-radius"
          value={radius}
          onChange={(event) => onRadiusChange(Number(event.target.value))}
        >
          <option value={500}>500m</option>
          <option value={1000}>1km</option>
          <option value={2000}>2km</option>
          <option value={5000}>5km</option>
        </select>
        <button className="btn-outline" onClick={onNearby}>
          {nearbyLoading ? "조회 중" : "주변 지점 보기"}
        </button>
      </div>
      {nearby.length > 0 ? (
        <ul className="nearby-mini-list">
          {nearby.slice(0, 5).map((item) => (
            <li key={`${item.source_type}-${item.hotspot_id}`}>
              <button onClick={() => onSelectNearby(item)}>
                <strong>{item.location_name ?? "사고다발지역 선정지점"}</strong>
                <span>
                  {sourceMeta[item.source_type].short} ·{" "}
                  {formatNumber(item.distance_m)}m
                </span>
              </button>
            </li>
          ))}
        </ul>
      ) : null}
      <div className="detail-actions">
        <button
          className="btn-primary"
          disabled={!regionName}
          onClick={() => onOpenRegion(regionName)}
        >
          이 지역 분석 보기
        </button>
      </div>
    </aside>
  );
}

function RegionAnalysis({ initialRegion }: { initialRegion: string }) {
  const [aiRegions, setAiRegions] = useState<AiRiskRegion[]>([]);
  const [facilities, setFacilities] = useState<Region[]>([]);
  const [candidates, setCandidates] = useState<ImprovementCandidate[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [compareQuery, setCompareQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    Promise.all([
      fetchJson<AiRiskResponse>(
        "/api/ai/high-risk?limit=300",
        controller.signal,
      ),
      fetchJson<RegionResponse>("/api/facilities?limit=100", controller.signal),
      fetchJson<ImprovementResponse>(
        "/api/improvement-candidates?limit=100",
        controller.signal,
      ),
    ])
      .then(([aiData, facilityData, candidateData]) => {
        setAiRegions(aiData.items);
        setFacilities(facilityData.items);
        setCandidates(candidateData.items);
      })
      .catch((cause) => {
        if ((cause as Error).name !== "AbortError")
          setError("지역분석 API 연결에 실패했습니다.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    if (!initialRegion || aiRegions.length === 0) return;
    const exact = aiRegions.find((item) => item.sido_sgg_nm === initialRegion);
    const loose = aiRegions.find((item) =>
      item.sido_sgg_nm.includes(initialRegion),
    );
    const resolved = resolveRegionName(
      initialRegion,
      aiRegions.map((item) => item.sido_sgg_nm),
    );
    const nextId = exact?.sido_sgg_nm ?? loose?.sido_sgg_nm ?? resolved;
    if (nextId) setSelectedId(nextId);
  }, [aiRegions, initialRegion]);
  const facilityByRegion = useMemo(
    () => new Map(facilities.map((item) => [item.sido_sgg_nm, item])),
    [facilities],
  );
  const candidateByRegion = useMemo(
    () => new Map(candidates.map((item) => [item.sido_sgg_nm, item])),
    [candidates],
  );
  const selected =
    aiRegions.find((item) => item.sido_sgg_nm === selectedId) ?? null;
  const selectedFacility = selected
    ? (facilityByRegion.get(selected.sido_sgg_nm) ?? null)
    : null;
  const selectedCandidate = selected
    ? (candidateByRegion.get(selected.sido_sgg_nm) ?? null)
    : null;
  const regionOptions = query
    ? aiRegions.filter((item) => item.sido_sgg_nm.includes(query)).slice(0, 12)
    : [];
  const compareOptions = compareQuery
    ? aiRegions
        .filter(
          (item) =>
            item.sido_sgg_nm.includes(compareQuery) &&
            !compareIds.includes(item.sido_sgg_nm),
        )
        .slice(0, 8)
    : [];
  const comparison = compareIds
    .map((id) => aiRegions.find((item) => item.sido_sgg_nm === id))
    .filter(Boolean) as AiRiskRegion[];
  const national = useMemo(
    () => ({
      avgRate: average(
        aiRegions.map((item) => item.prev_accident_per_10k_child),
      ),
      avgAccident: average(aiRegions.map((item) => item.prev_accident_count)),
      medianScore:
        [...aiRegions]
          .map((item) => scoreNumber(item.predicted_high_risk_probability_2026))
          .filter((value): value is number => value != null)
          .sort((a, b) => a - b)[Math.floor(aiRegions.length / 2)] ?? null,
      rateValues: aiRegions
        .map((item) => item.prev_accident_per_10k_child)
        .filter((value): value is number => typeof value === "number"),
      scoreValues: aiRegions
        .map((item) => scoreNumber(item.predicted_high_risk_probability_2026))
        .filter((value): value is number => value != null),
    }),
    [aiRegions],
  );
  const facilityMax = {
    crosswalk: Math.max(
      1,
      ...facilities.map((item) => item.crosswalk_count ?? 0),
    ),
    traffic: Math.max(
      1,
      ...facilities.map((item) => item.traffic_light_count ?? 0),
    ),
    schoolzone: Math.max(
      1,
      ...facilities.map((item) => item.schoolzone_count ?? 0),
    ),
    cctv: Math.max(
      0.01,
      ...facilities.map((item) => item.cctv_install_rate ?? 0),
    ),
  };
  const toggleCompare = (id: string) =>
    setCompareIds((current) =>
      current.includes(id)
        ? current.filter((item) => item !== id)
        : current.length >= 4
          ? [...current.slice(1), id]
          : [...current, id],
    );
  const facilityCompareItems = comparison
    .map((item) => facilityByRegion.get(item.sido_sgg_nm))
    .filter(Boolean) as Region[];
  const revealRef = useReveal<HTMLElement>(!loading);
  const canCompareFacilities =
    comparison.length > 0 && facilityCompareItems.length === comparison.length;
  return (
    <section className="analysis-view region-view" ref={revealRef}>
      <div className="page-frame">
        <PageHeading
          kicker="지역분석 · 전국 229개 시군구"
          lead="지역별로 보면,"
          title="다음이 보입니다."
          copy="사고 수준과 AI 분석, 안전시설 현황을 같은 기준으로 비교합니다. 시설 상세분석은 데이터가 확보된 49개 지역에만 표시됩니다."
        />
        {error ? <div className="inline-error">{error}</div> : null}
        {loading ? (
          <div className="analysis-loading">
            <span className="loader" />
            전국 229개 지역 데이터를 불러오는 중입니다
          </div>
        ) : (
          <div className="analysis-canvas">
            <section className="region-stage rv">
              <div className="region-stage-copy">
              <div className="region-story">
                <span className="kicker">선택 지역 요약</span>
                <h2 key={`t-${selectedId}`} className="rv-swap">
                  {selected?.sido_sgg_nm ?? "지역 미선택"}
                </h2>
                <p key={`p-${selectedId}`} className="rv-swap late">
                  {selected
                    ? makeRegionDiagnosis(selected, aiRegions, national.avgRate)
                    : "지역을 선택하면 사고 현황과 AI 분석 결과를 확인할 수 있습니다."}
                </p>
              </div>
            <section className="region-search-strip" aria-label="지역 선택">
              <div className="strip-label">
                <strong>지역 선택</strong>
                <span>
                  <ScopeTag scope="229" /> <ScopeTag scope="49" />
                </span>
              </div>
              <div className="strip-search">
                <label className="sr-only" htmlFor="region-query">
                  지역명 검색
                </label>
                <input
                  id="region-query"
                  type="search"
                  value={query}
                  placeholder="지역명 검색 예: 서울특별시 강남구"
                  onChange={(event) => setQuery(event.target.value)}
                />
                {query ? (
                  <div className="region-option-popover" role="listbox">
                    {regionOptions.map((region) => (
                      <button
                        key={region.sido_sgg_nm}
                        role="option"
                        aria-selected={region.sido_sgg_nm === selectedId}
                        onClick={() => {
                          setSelectedId(region.sido_sgg_nm);
                          setQuery("");
                        }}
                      >
                        {region.sido_sgg_nm}
                        {facilityByRegion.has(region.sido_sgg_nm) ? (
                          <em>시설분석 가능</em>
                        ) : null}
                      </button>
                    ))}
                    {regionOptions.length === 0 ? (
                      <span className="popover-empty">일치하는 지역이 없습니다.</span>
                    ) : null}
                  </div>
                ) : null}
              </div>
              <select
                value={selectedId}
                onChange={(event) => setSelectedId(event.target.value)}
                aria-label="상세 지역 선택"
              >
                <option value="">지역을 선택하세요 (229개)</option>
                {aiRegions.map((region) => (
                  <option key={region.sido_sgg_nm} value={region.sido_sgg_nm}>
                    {region.sido_sgg_nm}
                  </option>
                ))}
              </select>
            </section>

              </div>
              <RegionNodeMap
                regions={aiRegions.map((item) => item.sido_sgg_nm)}
                selected={selectedId}
                compare={compareIds}
                onSelect={setSelectedId}
              />
            </section>

            {/* 지역을 고르기 전에는 수치 띠를 숨기고, 고르면 카운트업되며 나타남 */}
            {selected ? (
              <section className="region-summary-hero rv">
                <div className="summary-metrics">
                  <div className="metric-group">
                    <small>일반 현황 · 전년도 기준</small>
                    <div className="metric-row">
                      <BigMetric
                        label="어린이 사고건수"
                        value={`${formatNumber(selected?.prev_accident_count)}건`}
                      />
                      <BigMetric
                        label="어린이 1만명당 사고율"
                        value={formatDecimal(selected?.prev_accident_per_10k_child)}
                      />
                      <BigMetric
                        label="어린이 인구"
                        value={`${formatNumber(selected?.prev_child_population)}명`}
                      />
                    </div>
                  </div>
                  <div className="metric-group ai">
                    <small>
                      AI 분석 · 229개 지역 간 상대 비교 (발생확률 아님)
                    </small>
                    <div className="metric-row">
                      <BigMetric
                        label="AI 상대적 고위험 점수"
                        value={formatScore(
                          selected?.predicted_high_risk_probability_2026,
                        )}
                        tone="ai"
                      />
                      <BigMetric
                        label={`전국 AI 순위 (${formatNumber(aiRegions.length)}개 중)`}
                        value={
                          selected
                            ? `${formatNumber(selected.risk_rank_2026)}위`
                            : "—"
                        }
                        tone="ai"
                      />
                    </div>
                  </div>
                </div>
              </section>
            ) : null}

            {selected ? (
              <section className="region-insight-grid rv">
                <article className="analysis-panel wide">
                  <div className="panel-title">
                    <strong>전국 대비 위치</strong>
                    <span>평균·분포 기준의 분석용 비교</span>
                    <ScopeTag scope="229" />
                  </div>
                  <PositionBars
                    selected={selected}
                    avgRate={national.avgRate}
                    avgAccident={national.avgAccident}
                    medianScore={national.medianScore}
                    ratePercentile={percentile(
                      selected.prev_accident_per_10k_child,
                      national.rateValues,
                    )}
                    scorePercentile={percentile(
                      scoreNumber(
                        selected.predicted_high_risk_probability_2026,
                      ) ?? undefined,
                      national.scoreValues,
                    )}
                  />
                </article>
                <article className="analysis-panel">
                  <div className="panel-title">
                    <strong>최근 변화 지표</strong>
                    <span>기존 변화량 필드 기반</span>
                  </div>
                  <RecentTrendLine selected={selected} />
                  <div className="compact-metric-grid">
                    <Metric
                      label="최근 1년 사고건수 변화"
                      value={formatSigned(
                        selected.accident_count_change_1y,
                        "건",
                      )}
                    />
                    <Metric
                      label="최근 1년 사고율 변화"
                      value={formatSigned(selected.accident_rate_change_1y)}
                    />
                    <Metric
                      label="최근 2년 평균 사고건수"
                      value={`${formatDecimal(selected.accident_count_mean_2y)}건`}
                    />
                    <Metric
                      label="자동차 등록대수"
                      value={`${formatNumber(selected.prev_vehicle_count)}대`}
                    />
                  </div>
                </article>
              </section>
            ) : null}

            <section className="region-insight-grid rv">
              <article className="analysis-panel compare-builder">
                <div className="panel-title">
                  <strong>지역 비교</strong>
                  <span>{compareIds.length}/4 선택</span>
                </div>
                <label className="sr-only" htmlFor="compare-query">
                  비교할 지역 검색
                </label>
                <input
                  id="compare-query"
                  type="search"
                  value={compareQuery}
                  placeholder="비교할 지역 검색"
                  onChange={(event) => setCompareQuery(event.target.value)}
                />
                <div className="compare-chip-row">
                  {compareIds.map((id) => (
                    <button
                      key={id}
                      onClick={() => toggleCompare(id)}
                      aria-label={`${id} 비교에서 제외`}
                    >
                      {id}
                      <b aria-hidden="true">×</b>
                    </button>
                  ))}
                </div>
                <div className="compare-option-grid">
                  {compareOptions.map((region) => (
                    <button
                      key={region.sido_sgg_nm}
                      onClick={() => toggleCompare(region.sido_sgg_nm)}
                    >
                      + {region.sido_sgg_nm}
                    </button>
                  ))}
                </div>
              </article>
              <article className="analysis-panel wide">
                <div className="panel-title">
                  <strong>229개 공통지표 비교</strong>
                  <span>AI 점수는 실제 발생 가능성 수치가 아님</span>
                  <ScopeTag scope="229" />
                </div>
                {comparison.length === 0 ? (
                  <p className="compact-empty">
                    비교할 지역을 최대 4개까지 선택하세요.
                  </p>
                ) : (
                  <RegionCompareChart items={comparison} />
                )}
              </article>
            </section>

            <section className="analysis-panel facility-section rv">
              <div className="panel-title">
                <strong>시설 상세분석</strong>
                <span>49개 시설 상세분석 가능 지역에만 표시</span>
                <ScopeTag scope="49" />
              </div>
              {!selected ? (
                <p className="compact-empty">
                  지역을 선택하면 시설 상세분석 가능 여부가 표시됩니다.
                </p>
              ) : selectedFacility ? (
                <div className="facility-layout">
                  <FacilityRing
                    facility={selectedFacility}
                    candidate={selectedCandidate}
                  />
                  <div className="facility-side-metrics">
                    <Metric
                      label="시설 부족도"
                      value={formatScore(
                        selectedCandidate?.infrastructure_deficiency_score,
                      )}
                    />
                    <Metric
                      label="개선 검토 점수"
                      value={formatScore(
                        selectedCandidate?.improvement_priority_score,
                      )}
                    />
                    <Metric
                      label="개선 검토 후보 순위 (49개 중)"
                      value={
                        selectedCandidate
                          ? `${formatNumber(selectedCandidate.improvement_priority_rank)}위`
                          : "자료 없음"
                      }
                    />
                  </div>
                  <div className="facility-meter-grid">
                    <FacilityMeter
                      label="횡단보도"
                      value={selectedFacility.crosswalk_count}
                      max={facilityMax.crosswalk}
                    />
                    <FacilityMeter
                      label="신호등"
                      value={selectedFacility.traffic_light_count}
                      max={facilityMax.traffic}
                    />
                    <FacilityMeter
                      label="어린이보호구역"
                      value={selectedFacility.schoolzone_count}
                      max={facilityMax.schoolzone}
                    />
                    <FacilityMeter
                      label="CCTV 설치율"
                      value={selectedFacility.cctv_install_rate}
                      max={facilityMax.cctv}
                      format="percent"
                    />
                  </div>
                </div>
              ) : (
                <p className="facility-unavailable compact">
                  현재 이 지역은 상세 안전시설 비교 데이터가 제공되지 않습니다.
                  <br />
                  시설 분석은 49개 지역을 대상으로 제공됩니다.
                </p>
              )}
              {comparison.length > 0 ? (
                <div className="facility-compare-note">
                  {canCompareFacilities
                    ? `선택한 ${comparison.length}개 지역은 모두 시설 비교가 가능합니다.`
                    : "비교 지역 중 시설 상세분석 데이터가 없는 지역이 있어 시설값을 0으로 보정하지 않습니다."}
                </div>
              ) : null}
            </section>
          </div>
        )}
      </div>
    </section>
  );
}

// Hotspot records use short or legacy province names ("강원", "광주광역시(구)").
// Resolve them to the 229-region naming only when exactly one region matches;
// otherwise nothing is preselected rather than guessing.
const SIDO_SHORT: Record<string, string> = {
  서울특별시: "서울", 부산광역시: "부산", 대구광역시: "대구", 인천광역시: "인천",
  광주광역시: "광주", 대전광역시: "대전", 울산광역시: "울산", 세종특별자치시: "세종",
  경기도: "경기", 강원도: "강원", 강원특별자치도: "강원", 충청북도: "충북",
  충청남도: "충남", 전라북도: "전북", 전북특별자치도: "전북", 전라남도: "전남",
  경상북도: "경북", 경상남도: "경남", 제주특별자치도: "제주",
};
function sidoKeys(raw: string) {
  const sido = raw.replace(/\(.*\)/, "");
  if (sido === "전남광주통합특별시") return ["전남", "광주"];
  return [SIDO_SHORT[sido] ?? sido];
}
function resolveRegionName(initial: string, names: string[]) {
  const [rawSido, ...rest] = initial.trim().split(/\s+/);
  const sigungu = rest.join(" ");
  if (!rawSido || !sigungu) return null;
  const keys = sidoKeys(rawSido);
  const compact = sigungu.replace(/\s/g, "");
  const first = sigungu.split(/\s+/)[0];
  const city = compact.match(/^(.+?시)/)?.[1];
  const matches = names.filter((name) => {
    const [nameSido, ...nameRest] = name.split(" ");
    const nameSgg = nameRest.join(" ");
    if (!keys.includes(SIDO_SHORT[nameSido] ?? nameSido)) return false;
    const nameCompact = nameSgg.replace(/\s/g, "");
    return nameCompact === compact || nameCompact === first || (!!city && nameSgg === city);
  });
  return matches.length === 1 ? matches[0] : null;
}

function makeRegionDiagnosis(
  selected: AiRiskRegion,
  all: AiRiskRegion[],
  avgRate: number | null,
) {
  const rate = selected.prev_accident_per_10k_child;
  const rateText =
    typeof rate === "number" && avgRate != null
      ? rate >= avgRate
        ? "전국 평균보다 높은 편"
        : "전국 평균보다 낮은 편"
      : "전국 평균 비교 자료가 제한적";
  const change =
    typeof selected.accident_count_change_1y === "number"
      ? selected.accident_count_change_1y > 0
        ? "최근 1년 사고건수는 증가했습니다"
        : selected.accident_count_change_1y < 0
          ? "최근 1년 사고건수는 감소했습니다"
          : "최근 1년 사고건수는 큰 변화가 없습니다"
      : "최근 변화 정보는 제한적입니다";
  return `${withTopic(selected.sido_sgg_nm)} 전국 ${all.length || 229}개 지역 중 AI 상대적 고위험도 ${selected.risk_rank_2026}위입니다. 어린이 1만 명당 사고율은 ${rateText}이며, ${change}.`;
}
function PositionBars({
  selected,
  avgRate,
  avgAccident,
  medianScore,
  ratePercentile,
  scorePercentile,
}: {
  selected: AiRiskRegion;
  avgRate: number | null;
  avgAccident: number | null;
  medianScore: number | null;
  ratePercentile: number;
  scorePercentile: number;
}) {
  const rows = [
    {
      label: "사고율",
      base:
        avgRate == null ? "전국 평균 —" : `전국 평균 ${formatDecimal(avgRate)}`,
      selected: formatDecimal(selected.prev_accident_per_10k_child),
      pos: ratePercentile,
    },
    {
      label: "AI 점수",
      base:
        medianScore == null
          ? "전국 중앙값 —"
          : `전국 중앙값 ${formatDecimal(medianScore)}점`,
      selected: formatScore(selected.predicted_high_risk_probability_2026),
      pos: scorePercentile,
    },
    {
      label: "사고건수",
      base:
        avgAccident == null
          ? "전국 평균 —"
          : `전국 평균 ${formatDecimal(avgAccident)}건`,
      selected: `${formatNumber(selected.prev_accident_count)}건`,
      pos: 50,
    },
  ];
  return (
    <div className="position-bars">
      {rows.map((row) => (
        <div key={row.label} className="position-row">
          <span>{row.label}</span>
          <i>
            <b style={{ left: `${Math.max(4, Math.min(96, row.pos))}%` }} />
          </i>
          <strong>{row.selected}</strong>
          <em>{row.base}</em>
        </div>
      ))}
    </div>
  );
}
function RecentTrendLine({ selected }: { selected: AiRiskRegion }) {
  const rateNow =
    typeof selected.prev_accident_per_10k_child === "number"
      ? selected.prev_accident_per_10k_child
      : null;
  const countNow =
    typeof selected.prev_accident_count === "number"
      ? selected.prev_accident_count
      : null;
  const rateChange =
    typeof selected.accident_rate_change_1y === "number"
      ? selected.accident_rate_change_1y
      : null;
  const countChange =
    typeof selected.accident_count_change_1y === "number"
      ? selected.accident_count_change_1y
      : null;
  const data = [
    {
      date: new Date("2025-01-01"),
      rate:
        rateNow != null && rateChange != null
          ? rateNow - rateChange
          : (rateNow ?? 0),
      count:
        countNow != null && countChange != null
          ? countNow - countChange
          : (countNow ?? 0),
    },
    { date: new Date("2026-01-01"), rate: rateNow ?? 0, count: countNow ?? 0 },
  ];
  return (
    <div className="recent-trend-line">
      <LineChart
        data={data}
        xDataKey="date"
        aspectRatio="3.4 / 1"
        margin={{ top: 16, right: 28, bottom: 18, left: 28 }}
        animationDuration={620}
      >
        <Grid strokeOpacity={0.12} />
        <Line
          dataKey="rate"
          stroke={REGION_MINT}
          strokeWidth={2.4}
          showMarkers
        markers={{ radius: 3.5, fill: REGION_MINT }}
        />
        <Line
          dataKey="count"
          stroke={REGION_SLATE}
          strokeWidth={1.8}
          showMarkers
        markers={{ radius: 3, fill: REGION_SLATE }}
        />
      </LineChart>
      <div className="trend-line-legend">
        <span>사고율 변화</span>
        <span>사고건수 변화</span>
      </div>
    </div>
  );
}
function RegionCompareChart({ items }: { items: AiRiskRegion[] }) {
  const data = items.map((item) => ({
    region: item.sido_sgg_nm.replace("특별자치도 ", " ").replace("광역시 ", " "),
    accident: item.prev_accident_count ?? 0,
    rate: item.prev_accident_per_10k_child ?? 0,
    score: scoreNumber(item.predicted_high_risk_probability_2026) ?? 0,
  }));
  return (
    <div className="compare-chart-product">
      <BarChart
        data={data}
        xDataKey="region"
        aspectRatio="3.4 / 1"
        margin={{ top: 18, right: 24, bottom: 40, left: 24 }}
        barGap={0.24}
        animationDuration={650}
      >
        <Grid strokeOpacity={0.14} />
        <Bar dataKey="accident" fill={REGION_INK} lineCap="round" />
        <Bar dataKey="rate" fill={REGION_SLATE} lineCap="round" />
        <Bar dataKey="score" fill={REGION_TEAL} lineCap="round" />
        <BarXAxis showAllLabels />
      </BarChart>
      <div className="compare-chart-legend">
        <span>사고건수</span>
        <span>1만명당 사고율</span>
        <span>AI 점수</span>
      </div>
    </div>
  );
}

// "경상북도 영덕군" → "경북 영덕군" (chart labels only; reuses SIDO_SHORT)
function shortSidoName(name: string) {
  const [sido, ...rest] = name.split(" ");
  if (!rest.length) return name;
  return `${SIDO_SHORT[sido] ?? sido.slice(0, 2)} ${rest.join(" ")}`;
}

function AnalysisStoryCase({
  item,
  candidate,
}: {
  item: AiRiskRegion;
  candidate?: ImprovementCandidate;
}) {
  const hasFacility = Boolean(candidate);
  const interpretation = hasFacility
    ? `${withTopic(item.sido_sgg_nm)} 49개 시설분석 대상에 포함되어 AI 신호와 시설 부족도를 함께 검토할 수 있습니다.`
    : `${withTopic(item.sido_sgg_nm)} 229개 AI 분석 대상이지만 현재 49개 시설 상세분석에는 포함되지 않아 시설 부족 여부를 판단하지 않습니다.`;
  return (
    <article className="story-case">
      <div className="story-case-head">
        <strong>{item.sido_sgg_nm}</strong>
        <span>{item.risk_rank_2026}위</span>
      </div>
      <dl>
        <div>
          <dt>사고건수</dt>
          <dd>{formatNumber(item.prev_accident_count)}건</dd>
        </div>
        <div>
          <dt>1만명당 사고율</dt>
          <dd>{formatDecimal(item.prev_accident_per_10k_child, 3)}</dd>
        </div>
        <div>
          <dt>AI 상대적 고위험 점수</dt>
          <dd>{formatScore(item.predicted_high_risk_probability_2026)}</dd>
        </div>
        <div>
          <dt>시설 데이터</dt>
          <dd>{hasFacility ? "제공" : "미제공"}</dd>
        </div>
        <div>
          <dt>개선 검토</dt>
          <dd>
            {candidate
              ? `${candidate.improvement_priority_rank}위 · ${formatScore(candidate.improvement_priority_score)}`
              : "해당 없음"}
          </dd>
        </div>
      </dl>
      <p>{interpretation}</p>
    </article>
  );
}

function AiAnalysis({
  onOpenRegion,
}: {
  onOpenRegion: (region: string) => void;
}) {
  const [all, setAll] = useState<AiRiskRegion[]>([]);
  const [candidates, setCandidates] = useState<ImprovementCandidate[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [importance, setImportance] = useState<ImportanceItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [range, setRange] = useState<229 | 50 | 10>(229);
  const [hover, setHover] = useState<number | null>(null);
  const [hoverCand, setHoverCand] = useState<string | null>(null);
  const revealRef = useReveal<HTMLElement>(!loading);
  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      fetchJson<AiRiskResponse>("/api/ai/high-risk?limit=300", controller.signal),
      fetchJson<ImprovementResponse>("/api/improvement-candidates?limit=49", controller.signal),
      fetchJson<ModelInfo>("/api/model/info", controller.signal),
      fetchJson<ImportanceResponse>("/api/model/importance", controller.signal),
    ])
      .then(([riskData, candidateData, infoData, importanceData]) => {
        setAll([...riskData.items].sort((a, b) => a.risk_rank_2026 - b.risk_rank_2026));
        setCandidates(candidateData.items);
        setModelInfo(infoData);
        setImportance(importanceData.items);
      })
      .catch((cause) => {
        if ((cause as Error).name !== "AbortError") setError("AI 분석 API 연결에 실패했습니다.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, []);

  const scores = all.map((item) => scoreNumber(item.predicted_high_risk_probability_2026) ?? 0);
  const medianIndex = Math.floor((all.length - 1) / 2);
  const top10Score = scores[Math.min(9, scores.length - 1)];
  const shown = all.slice(0, range);
  const shownScores = scores.slice(0, range);
  const W = 1000, H = 260, PAD_T = 18, PAD_B = 26;
  const yMin = range === 229 ? 0 : Math.floor(Math.min(...shownScores, 100) - 2);
  const yMax = 100;
  const yOf = (score: number) => PAD_T + (1 - (score - yMin) / (yMax - yMin)) * (H - PAD_T - PAD_B);
  // bar chart: each region gets an equal slot; x is the slot centre
  const xOf = (index: number) => (shown.length ? ((index + 0.5) / shown.length) * W : W / 2);
  const refs = [
    { label: "TOP 10 경계", score: top10Score, show: range !== 10 },
    { label: "중앙값", score: scores[medianIndex], show: range === 229 },
  ].filter((r) => r.show && typeof r.score === "number" && r.score >= yMin);
  const hovered = hover != null ? shown[hover] : null;

  const tableRows = all.slice(0, 12);
  const minScore = Math.min(...scores.slice(0, 12), 100);
  const impMax = Math.max(...importance.map((i) => Math.max(0, i.importance_mean)), 0);
  const scatter = candidates.filter(
    (c) => typeof c.infrastructure_score === "number" && typeof c.predicted_high_risk_probability_2026 === "number",
  );
  const top5 = [...candidates].sort((a, b) => a.improvement_priority_rank - b.improvement_priority_rank).slice(0, 5);
  const top5Names = new Set(top5.map((c) => c.sido_sgg_nm));
  const candidateByRegion = new Map(candidates.map((c) => [c.sido_sgg_nm, c]));
  const storyCaseNames = [
    all[0]?.sido_sgg_nm,
    top5[0]?.sido_sgg_nm,
    top5[1]?.sido_sgg_nm,
  ].filter(Boolean);
  const storyCases = Array.from(new Set(storyCaseNames))
    .map((name) => all.find((item) => item.sido_sgg_nm === name))
    .filter((item): item is AiRiskRegion => Boolean(item))
    .slice(0, 3);
  const infraVals = scatter.map((c) => scoreNumber(c.infrastructure_score) ?? 0).sort((a, b) => a - b);
  const aiVals = scatter.map((c) => scoreNumber(c.predicted_high_risk_probability_2026) ?? 0).sort((a, b) => a - b);
  const xLo = Math.max(0, Math.floor(((infraVals[0] ?? 0) - 5) / 5) * 5), xHi = Math.min(100, Math.ceil(((infraVals[infraVals.length - 1] ?? 100) + 5) / 5) * 5);
  const yLo = Math.max(0, Math.floor(((aiVals[0] ?? 0) - 5) / 5) * 5), yHi = 100;
  const sx = (v: number) => 36 + ((v - xLo) / (xHi - xLo)) * 540;
  const sy = (v: number) => 22 + (1 - (v - yLo) / (yHi - yLo)) * 300;
  const infraMed = infraVals[Math.floor((infraVals.length - 1) / 2)] ?? 50;
  const aiMed = aiVals[Math.floor((aiVals.length - 1) / 2)] ?? 50;

  return (
    <section className="analysis-view ai-view ai-v2" ref={revealRef}>
      <div className="page-frame">
        <PageHeading
          kicker="AI 분석 · 전국 229개 시군구"
          lead="다음 연도,"
          title="상대적 고위험 신호가 강한 지역"
          copy="Logistic Regression 모델이 전국 229개 지역의 다음 연도 상대적 고위험 여부를 분석합니다. 예측은 확률이 아니라 지역 간 비교를 위한 분석 점수입니다."
          warning="AI 점수는 실제 발생 가능성 수치가 아닌 상대적 고위험 분석 결과입니다."
        />
        {error ? <div className="inline-error">{error}</div> : null}
        {loading ? (
          <div className="analysis-loading">
            <span className="loader" />
            AI 분석 데이터를 불러오는 중입니다
          </div>
        ) : (
          <div className="ai-dashboard analysis-canvas">
            {/* KPI header + distribution chart */}
            <section className="ai2-hero rv">
              <div className="ai2-hero-top">
                <div className="ai2-title">
                  <h2>AI 상대적 고위험 점수 분포</h2>
                  <span className="ai2-chip">● 229개 시군구</span>
                </div>
                <div className="ai2-meta">
                  <span>모델 <b>{modelInfo?.model.model_name ?? "Logistic Regression"}</b></span>
                  <span>예측 기준 <b>다음 연도 상대적 고위험군</b></span>
                  <span>설명 <b>Permutation importance</b></span>
                </div>
                <div className="ai2-range" role="tablist" aria-label="표시 범위">
                  {([229, 50, 10] as const).map((r) => (
                    <button key={r} role="tab" aria-selected={range === r} onClick={() => { setRange(r); setHover(null); }}>
                      {r === 229 ? "전체" : `상위 ${r}`}
                    </button>
                  ))}
                </div>
              </div>
              <div className="ai2-kpis">
                <div>
                  <dt>1위 · {all[0]?.sido_sgg_nm}</dt>
                  <dd><CountUpText text={formatDecimal(scores[0])} /><small>점</small></dd>
                </div>
                <div>
                  <dt>TOP 10 경계 · {all[9]?.sido_sgg_nm}</dt>
                  <dd><CountUpText text={formatDecimal(top10Score)} /><small>점</small></dd>
                </div>
                <div>
                  <dt>중앙값 · {medianIndex + 1}위</dt>
                  <dd><CountUpText text={formatDecimal(scores[medianIndex])} /><small>점</small></dd>
                </div>
                <p className="ai2-note">↑ 높을수록 다른 지역보다 상대적 고위험 신호가 강함 · 발생확률 아님</p>
              </div>
              <div className="ai2-chart" onMouseLeave={() => setHover(null)}>
                <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-label={`AI 점수 분포 막대그래프 · ${shown.length}개 지역`}>
                  {refs.map((r) => (
                    <line key={r.label} x1="0" x2={W} y1={yOf(r.score)} y2={yOf(r.score)} className="ai2-ref" />
                  ))}
                  <g key={range} className="ai2-bars">
                    {shown.map((item, i) => {
                      const slot = W / shown.length;
                      const gap = shown.length > 100 ? slot * 0.28 : slot * 0.22;
                      const y = yOf(shownScores[i]);
                      return (
                        <rect
                          key={item.sido_sgg_nm}
                          className={`ai2-bar${i < 10 ? " is-top" : ""}${hover === i ? " is-on" : ""}${hover != null && hover !== i ? " is-dim" : ""}`}
                          x={i * slot + gap / 2}
                          y={y}
                          width={Math.max(0.6, slot - gap)}
                          height={Math.max(0, H - PAD_B - y)}
                          style={{ animationDelay: `${Math.round((i / shown.length) * 520)}ms` }}
                        />
                      );
                    })}
                  </g>
                  {shown.map((item, i) => (
                    <rect
                      key={`hit-${item.sido_sgg_nm}`}
                      x={i * (W / shown.length)}
                      y={0}
                      width={W / shown.length}
                      height={H}
                      fill="transparent"
                      onMouseEnter={() => setHover(i)}
                      onClick={() => onOpenRegion(item.sido_sgg_nm)}
                    />
                  ))}
                </svg>
                {refs.map((r) => (
                  <span key={r.label} className="ai2-ref-label" style={{ top: `${(yOf(r.score) / H) * 100}%` }}>
                    {r.label} <b>{formatDecimal(r.score)}</b>
                  </span>
                ))}
                {hovered && hover != null ? (
                  <>
                    <div
                      className="ai2-tip"
                      style={{ left: `clamp(90px, ${(xOf(hover) / W) * 100}%, calc(100% - 110px))`, top: `${(yOf(shownScores[hover]) / H) * 100}%` }}
                    >
                      <b>{formatDecimal(shownScores[hover])}점</b>
                      <span>{hovered.risk_rank_2026}위 · {hovered.sido_sgg_nm}</span>
                    </div>
                  </>
                ) : null}
                {range === 10 ? (
                  <ol className="ai2-xlabels" aria-hidden="true">
                    {shown.map((item) => (
                      <li key={item.sido_sgg_nm}>
                        {shortSidoName(item.sido_sgg_nm)}
                      </li>
                    ))}
                  </ol>
                ) : null}
                <div className="ai2-axis">
                  <span>1위</span>
                  <span>{range === 229 ? "229위" : `${range}위`}</span>
                </div>
              </div>
            </section>

            <section className="analysis-panel story-panel rv">
              <div className="panel-title">
                <strong>분석 구조</strong>
                <span>229개 AI 예측과 49개 시설분석은 마지막 단계에서만 결합합니다</span>
              </div>
              <div className="story-flow">
                <div>
                  <b>전국 229개 지역 AI 예측</b>
                  <p>어린이 인구, 자동차 등록대수, 사고건수·사고율 변화 등 지역 단위 feature로 다음 연도 상대적 고위험 신호를 산출합니다.</p>
                </div>
                <div>
                  <b>49개 지역 시설분석</b>
                  <p>횡단보도, 신호등, 어린이보호구역, CCTV 등 확보된 시설 데이터를 바탕으로 인프라 공급수준과 부족도를 계산합니다.</p>
                </div>
                <div>
                  <b>개선 검토 후보</b>
                  <p>AI 상대적 고위험 점수와 시설 부족도를 결합한 분석용 지표입니다. 공식 정책 우선순위가 아니라 검토 대상을 좁히는 신호입니다.</p>
                </div>
              </div>
            </section>

            <section className="ai2-grid rv">
              <article className="analysis-panel ai2-list">
                <div className="panel-title">
                  <strong>고위험 예상지역</strong>
                  <span>상위 12 · 행 선택 시 지역분석으로 이동</span>
                </div>
                <div className="ai2-rows" role="table" aria-label="AI 고위험 예상지역">
                  <div className="ai2-row head" role="row">
                    <span>순위</span><span>지역</span><span>AI 점수</span><span>1년 사고율 변화</span>
                  </div>
                  {tableRows.map((item, i) => {
                    const s = scores[i];
                    const change = item.accident_rate_change_1y;
                    return (
                      <button key={item.sido_sgg_nm} role="row" className={`ai2-row ${i < 10 ? "is-top" : ""}`} onClick={() => onOpenRegion(item.sido_sgg_nm)}>
                        <b>{String(item.risk_rank_2026).padStart(2, "0")}</b>
                        <span>{item.sido_sgg_nm}</span>
                        <strong>
                          <i style={{ width: `${Math.max(6, ((s - minScore) / Math.max(1, 100 - minScore)) * 100)}%` }} />
                          {formatDecimal(s)}
                        </strong>
                        <em className={typeof change === "number" ? (change > 0 ? "up" : change < 0 ? "down" : "") : ""}>
                          {typeof change === "number" ? `${change > 0 ? "↗" : change < 0 ? "↘" : "→"} ${formatSigned(change)}` : "—"}
                        </em>
                      </button>
                    );
                  })}
                </div>
                <p className="analysis-footnote">막대는 상위 12개 지역 사이의 점수 차이를 확대해 보여줍니다. 1–10위가 AI 예측 기반 TOP 10입니다.</p>
              </article>

              <article className="analysis-panel ai2-imp">
                <div className="panel-title">
                  <strong>모델 설명 변수</strong>
                  <span>Permutation importance · SHAP 미사용</span>
                </div>
                <ol>
                  {importance.map((item, i) => {
                    const v = Math.max(0, item.importance_mean);
                    return (
                      <li key={item.feature} className={i < 3 ? "lead" : ""}>
                        <span>{featureLabels[item.feature] ?? item.feature}</span>
                        <i><b style={{ width: `${impMax ? Math.max(1, (v / impMax) * 100) : 0}%` }} /></i>
                        <strong>{formatDecimal(v, 2)}</strong>
                      </li>
                    );
                  })}
                </ol>
                <p className="analysis-footnote">변수를 섞었을 때 모델 성능이 얼마나 줄어드는지로 중요도를 측정합니다.</p>
              </article>
            </section>

            <section className="analysis-panel ai2-quad rv">
              <div className="panel-title">
                <strong>AI 위험 × 인프라 공급수준</strong>
                <span>시설분석 49개 지역 · 개선 검토 후보 TOP 5 표시</span>
                <ScopeTag scope="49" />
              </div>
              <div className="ai2-quad-body">
                <div className="ai2-scatter">
                  <svg viewBox="0 0 600 340" aria-label="AI 위험과 인프라 공급수준 분포">
                    <rect x={sx(xLo)} y={sy(100)} width={sx(infraMed) - sx(xLo)} height={sy(aiMed) - sy(100)} className="ai2-zone" />
                    <line x1={sx(infraMed)} x2={sx(infraMed)} y1={sy(100)} y2={sy(yLo)} className="ai2-ref" />
                    <line x1={sx(xLo)} x2={sx(xHi)} y1={sy(aiMed)} y2={sy(aiMed)} className="ai2-ref" />
                    <text x={sx(xLo) + 8} y={sy(100) - 8} className="ai2-zone-label">위험 높음 · 인프라 낮음</text>
                    <text x={sx(xHi)} y={sy(aiMed) - 6} textAnchor="end" className="ai2-axis-t">AI 중앙값 {formatDecimal(aiMed)}</text>
                    <text x={sx(infraMed) + 6} y={sy(yLo) + 14} className="ai2-axis-t">인프라 중앙값 {formatDecimal(infraMed)}</text>
                    {[...scatter].sort((a, b) => (top5Names.has(a.sido_sgg_nm) ? -a.improvement_priority_rank : -999) - (top5Names.has(b.sido_sgg_nm) ? -b.improvement_priority_rank : -999)).map((c) => {
                      const x = sx(scoreNumber(c.infrastructure_score) ?? 0);
                      const y = sy(scoreNumber(c.predicted_high_risk_probability_2026) ?? 0);
                      const isTop = top5Names.has(c.sido_sgg_nm);
                      const on = hoverCand === c.sido_sgg_nm;
                      return (
                        <g key={c.sido_sgg_nm} className={`ai2-pt ${isTop ? "top" : ""} ${on ? "on" : ""}`}
                           onMouseEnter={() => setHoverCand(c.sido_sgg_nm)} onMouseLeave={() => setHoverCand(null)}
                           onClick={() => onOpenRegion(c.sido_sgg_nm)}>
                          <circle cx={x} cy={y} r={isTop ? 6 : 3.5} />
                          {isTop ? <text x={x} y={y + 3.5} textAnchor="middle">{c.improvement_priority_rank}</text> : null}
                        </g>
                      );
                    })}
                  </svg>
                  <div className="scatter-axis-note">
                    <span>인프라 공급수준 →</span>
                    <span>↑ AI 상대적 고위험 점수</span>
                  </div>
                </div>
                <ol className="ai2-cands">
                  {top5.map((c) => (
                    <li key={c.sido_sgg_nm}>
                      <button
                        className={hoverCand === c.sido_sgg_nm ? "on" : ""}
                        onMouseEnter={() => setHoverCand(c.sido_sgg_nm)} onMouseLeave={() => setHoverCand(null)}
                        onClick={() => onOpenRegion(c.sido_sgg_nm)}
                      >
                        <b>{c.improvement_priority_rank}</b>
                        <span>
                          {c.sido_sgg_nm}
                          <small>AI {formatScore(c.predicted_high_risk_probability_2026)} · 부족도 {formatScore(c.infrastructure_deficiency_score)}</small>
                        </span>
                        <strong>{formatDecimal(scoreNumber(c.improvement_priority_score))}<small>점</small></strong>
                      </button>
                    </li>
                  ))}
                </ol>
              </div>
              <p className="analysis-footnote">왼쪽 위 영역일수록 상대적 위험은 높고 인프라 공급수준은 낮은 지역입니다. 개선 검토 점수는 분석용 휴리스틱이며 공식 정책 우선순위가 아닙니다.</p>
            </section>

            <section className="analysis-panel story-panel rv">
              <div className="panel-title">
                <strong>실제 데이터 사례</strong>
                <span>현재 API 응답에서 가져온 지역만 표시</span>
              </div>
              <div className="story-cases">
                {storyCases.map((item) => (
                  <AnalysisStoryCase
                    key={item.sido_sgg_nm}
                    item={item}
                    candidate={candidateByRegion.get(item.sido_sgg_nm)}
                  />
                ))}
              </div>
              <p className="analysis-footnote">
                사고건수와 사고율은 229개 AI 결과의 직전 연도 지표이며, 시설 데이터 여부와 개선 검토 순위는 49개 시설분석 데이터가 있는 지역에만 표시됩니다.
              </p>
            </section>
          </div>
        )}
      </div>
    </section>
  );
}

function Guide() {
  const revealRef = useReveal<HTMLElement>(true);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    fetchJson<ModelInfo>("/api/model/info", controller.signal)
      .then(setModelInfo)
      .catch(() => undefined);
    return () => controller.abort();
  }, []);
  const logisticMetric = modelInfo?.metrics?.find((row) => String(row.model) === "Logistic Regression");
  return (
    <section className="analysis-view guide-view" ref={revealRef}>
      <div className="page-frame">
        <PageHeading
          kicker="데이터안내 · 해석 기준"
          lead="숫자보다 먼저,"
          title="맥락을 읽으세요."
          copy="세이프스팟의 지도와 AI 결과를 올바르게 해석하기 위한 기준입니다. 데이터 범위가 다른 지표를 하나의 의미로 합치지 않습니다."
        />
        <section className="guide-story rv" aria-label="공모전 분석 스토리라인">
          <div className="guide-story-lead">
            <h2>어린이 교통안전 의사결정을 위한 데이터 파이프라인</h2>
            <p>
              이 서비스는 공식 사고다발지역 지점, 전국 229개 시군구 패널,
              그리고 확보 가능한 49개 지역 시설 데이터를 각각의 의미대로 보존한 뒤
              지도·AI 분석·개선 검토 후보로 연결합니다.
            </p>
          </div>
          <div className="story-rows">
            <div>
              <strong>문제 정의</strong>
              <p>사고다발지역 지점과 지역 단위 위험 신호, 시설 부족 신호를 분리해 담당자가 어디를 먼저 살펴볼지 판단할 수 있게 합니다.</p>
            </div>
            <div className="story-row-wide">
              <strong>데이터 수집 출처와 기간</strong>
              <div className="guide-data-split">
                <section>
                  <h3>AI 분석 데이터</h3>
                  <p>2020–2025 어린이 교통사고, 어린이 인구, 자동차 등록대수를 결합해 전국 229개 시군구를 분석합니다.</p>
                </section>
                <section>
                  <h3>지도 데이터</h3>
                  <p>공식 사고다발지역 선정지점 1,803개를 표시합니다. schoolzone_hotspot과 child_pedestrian_hotspot을 사용하며, 전체 사고건수 데이터가 아니라 공식 기준으로 선정된 지점 데이터입니다.</p>
                </section>
                <section>
                  <h3>시설 상세분석 데이터</h3>
                  <p>횡단보도, 신호등, 어린이보호구역, CCTV 4종 시설 데이터가 공통 확보된 49개 지역만 상세 시설분석에 포함합니다.</p>
                </section>
              </div>
            </div>
            <div>
              <strong>전처리 과정</strong>
              <p>시도·시군구 명칭을 표준화하고 좌표·연도·사고/사상자 수를 정리했습니다. 연도가 없는 어린이보호구역 자료에는 임의 연도를 붙이지 않았습니다.</p>
            </div>
            <div>
              <strong>AI 모델링 과정</strong>
              <p>전국 229개 시군구를 대상으로 직전 연도 어린이 인구, 자동차 등록대수, 사고건수·사고율 변화, 최근 2년 평균 사고건수 등을 feature로 구성했습니다.</p>
            </div>
            <div>
              <strong>모델 비교와 선택</strong>
              <p>Logistic Regression, Random Forest, HistGradientBoosting을 비교했고 평균 ROC-AUC와 Average Precision(PR-AUC)이 가장 안정적인 Logistic Regression을 최종 모델로 사용했습니다.</p>
            </div>
            <div>
              <strong>성능 검증</strong>
              <p>
                Walk-forward 검증과 평균 성능지표를 확인했습니다.
                {logisticMetric
                  ? ` 최종 모델 평균 ROC-AUC ${formatMetric(logisticMetric.roc_auc)}, Average Precision(PR-AUC) ${formatMetric(logisticMetric.average_precision)}, Recall ${formatMetric(logisticMetric.recall)}입니다.`
                  : " 성능지표는 /api/model/info 응답 기준으로 표시됩니다."}
              </p>
            </div>
            <div>
              <strong>분석 한계</strong>
              <p>사고다발지역은 전체 사고자료가 아니며, AI 점수는 실제 사고 발생확률이 아닙니다. 49개 밖 지역의 시설값은 0으로 채우지 않고 미제공으로 구분합니다.</p>
            </div>
            <div>
              <strong>활용방안</strong>
              <p>지도에서 지점을 확인하고, 지역분석에서 공통지표를 비교한 뒤, AI 분석에서 상대적 고위험 신호와 시설 부족 신호가 겹치는 후보를 검토합니다.</p>
            </div>
          </div>
        </section>
        <ol className="guide-steps rv" aria-label="서비스 해석 흐름">
          <li>
            <span>01</span>
            <strong>어디에서 나타나는가</strong>
            <p>공식 기준으로 선정된 사고다발지역을 지도에서 확인합니다.</p>
          </li>
          <li>
            <span>02</span>
            <strong>지역 수준은 어떤가</strong>
            <p>전국 229개 지역의 사고 수준과 AI 분석 점수를 비교합니다.</p>
          </li>
          <li>
            <span>03</span>
            <strong>무엇을 개선할까</strong>
            <p>
              시설 데이터가 있는 지역은 부족 신호와 개선 검토 후보를 함께
              봅니다.
            </p>
          </li>
        </ol>

        <section className="guide-scope rv" aria-label="분석 범위">
          <div className="guide-scope-title">
            <span className="kicker">DATA COVERAGE</span>
            <h2>분석 범위</h2>
            <p>229개 AI 분석과 49개 시설분석은 데이터 범위가 다릅니다.</p>
          </div>
          <dl className="scope-numbers">
            <div>
              <dt>사고다발지역 선정지점</dt>
              <dd>
                <CountUpText text="1,803" />
                <small>개 지점</small>
              </dd>
              <ScopeTag scope="1803" />
            </div>
            <div>
              <dt>AI 분석 지역</dt>
              <dd>
                <CountUpText text="229" />
                <small>개 시군구</small>
              </dd>
              <ScopeTag scope="229" />
            </div>
            <div>
              <dt>시설 상세분석 가능 지역</dt>
              <dd>
                <CountUpText text="49" />
                <small>개 시군구</small>
              </dd>
              <ScopeTag scope="49" />
            </div>
          </dl>
        </section>

        <div className="guide-grid">
          <article className="rv guide-block">
            <span className="kicker">A · MAP DATA</span>
            <h2>사고다발지역이란?</h2>
            <p>
              지도에 표시된 항목은 전체 어린이 교통사고 위치 목록이 아닙니다.
              공식 기준에 따라 별도로 선정된 사고다발지역 지점이며, 선정지점 수를
              전체 사고건수로 해석하지 않습니다.
            </p>
            <div className="guide-callout">
              <strong>1,803개 선정지점</strong>
              <em>선정지점 수는 전체 사고건수가 아닙니다.</em>
            </div>
          </article>
          <article className="rv guide-block">
            <span className="kicker">B · YEAR POLICY</span>
            <h2>연도 정보는 정확하게</h2>
            <p>
              보행어린이 사고다발지역은 원본에 제공된 실제 연도만 필터링합니다.
              어린이보호구역 사고다발지역은 공식 연도 필드가 확인되지 않는 경우
              특정 연도를 임의로 붙이지 않습니다.
            </p>
            <div className="guide-compare-line">
              <b>확인된 연도만 표시</b>
              <i aria-hidden="true" />
              <b>추정 연도 미생성</b>
            </div>
          </article>
          <article className="rv guide-block">
            <span className="kicker">C · AI INTERPRETATION</span>
            <h2>AI 점수와 순위의 의미</h2>
            <p>
              AI 상대적 고위험 점수는 실제 사고가 발생할 확률이 아니라, 모델이
              상위 25% 고위험군과 얼마나 유사하다고 판단했는지를 나타내는
              상대적 분류 점수입니다. 전국 순위 역시 공식 교통안전 평가 순위가
              아니라 본 프로젝트의 분석용 순위입니다.
            </p>
            <div className="guide-equation">
              <strong>분석점수</strong>
              <b aria-label="같지 않음">≠</b>
              <strong>확률값</strong>
            </div>
          </article>
          <article className="rv guide-block">
            <span className="kicker">E · MODEL</span>
            <h2>모델 설명</h2>
            <dl className="guide-dl">
              <div>
                <dt>모델</dt>
                <dd>Logistic Regression</dd>
              </div>
              <div>
                <dt>예측 목표</dt>
                <dd>다음 연도 어린이 1만명당 사고율이 해당 연도 전국 상위 25% 수준에 해당하는 고위험군인지 분류</dd>
              </div>
              <div>
                <dt>Target</dt>
                <dd>어린이 1만명당 사고율 연도별 Q75 이상</dd>
              </div>
              <div>
                <dt>설명 방식</dt>
                <dd>Permutation importance 기반 변수 영향 확인</dd>
              </div>
            </dl>
            <div className="guide-model-note">
              <p>AI 상대적 고위험 점수는 실제 사고가 발생할 확률이 아니라, 모델이 상위 25% 고위험군과 얼마나 유사하다고 판단했는지를 나타내는 상대적 분류 점수입니다.</p>
              <p>전국 순위 역시 공식 교통안전 평가 순위가 아니라 본 프로젝트의 분석용 순위입니다.</p>
            </div>
          </article>
          <article className="rv guide-block guide-warning">
            <span className="kicker">F · LIMITATION</span>
            <h2>해석 시 주의사항</h2>
            <ul>
              <li>사고다발지역은 전체 어린이 교통사고 위치 목록을 의미하지 않습니다.</li>
              <li>선정지점 미존재는 안전 지역을 의미하지 않습니다.</li>
              <li>시설 데이터가 없는 지역을 시설 0개로 해석하지 않습니다.</li>
              <li>개선 검토 점수는 공식 정책 우선순위가 아닙니다.</li>
            </ul>
          </article>
          <article className="rv guide-block guide-assistant">
            <span className="kicker">G · LLM ASSISTANT</span>
            <h2>자연어 분석 어시스턴트</h2>
            <p>
              메인 화면의 AI 질문 기능은 사용자의 질문을 해석한 뒤, Tool Calling으로
              프로젝트의 실제 API/CSV 기반 데이터를 조회하고 답변을 생성합니다.
            </p>
            <ol>
              <li>사용자 질문</li>
              <li>LLM 의도 판단</li>
              <li>Tool Calling</li>
              <li>프로젝트 API/CSV 데이터 조회</li>
              <li>근거 기반 답변 생성</li>
            </ol>
            <div className="guide-ruleline">
              <b>신뢰성 규칙</b>
              <span>임의 수치 생성 금지 · 프로젝트 데이터에 없는 내용 추정 금지 · AI 점수의 실제 사고확률 표현 금지 · 시설 미제공을 0개로 처리 금지 · 사고다발지역 미선정을 안전하다는 의미로 해석 금지</span>
            </div>
          </article>
          <article className="rv guide-block data-provenance">
            <span className="kicker">DATA PROVENANCE</span>
            <h2>분석 결과는 의사결정의 시작점입니다.</h2>
            <p>
              시설 설치나 정책 결정을 위해서는 현장 확인과 추가 데이터 검토가
              함께 필요합니다. 세이프스팟은 데이터를 연결해 검토 대상을 좁히는
              분석 도구입니다.
            </p>
          </article>
        </div>
      </div>
    </section>
  );
}
function DataSourceRow({
  title,
  scope,
  purpose,
  items,
}: {
  title: string;
  scope: string;
  purpose: string;
  items: string;
}) {
  return (
    <div>
      <strong>{title}</strong>
      <span>{scope}</span>
      <span>{purpose}</span>
      <em>{items}</em>
    </div>
  );
}

export default App;


