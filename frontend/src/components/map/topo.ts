/** Converts the bundled TopoJSON province file into GeoJSON (moved out of App.tsx unchanged so the map layers can share it). */
export function topoToGeoJSON(topology: any): GeoJSON.FeatureCollection {
  const [sx, sy] = topology.transform.scale;
  const [tx, ty] = topology.transform.translate;
  const arcs = topology.arcs.map((arc: number[][]) => {
    let x = 0;
    let y = 0;
    return arc.map(([dx, dy]) => {
      x += dx;
      y += dy;
      return [x * sx + tx, y * sy + ty];
    });
  });
  const getArc = (index: number) => {
    const arc = arcs[index < 0 ? ~index : index];
    return index < 0 ? [...arc].reverse() : arc;
  };
  const stitch = (indexes: number[]) =>
    indexes.flatMap((index, part) => {
      const arc = getArc(index);
      return part === 0 ? arc : arc.slice(1);
    });
  const object = Object.values(topology.objects)[0] as any;
  return {
    type: "FeatureCollection",
    features: object.geometries.map((geometry: any) => {
      const coordinates =
        geometry.type === "Polygon"
          ? geometry.arcs.map((ring: number[]) => stitch(ring))
          : geometry.arcs.map((polygon: number[][]) =>
              polygon.map((ring: number[]) => stitch(ring)),
            );
      return {
        type: "Feature",
        properties: geometry.properties,
        geometry: { type: geometry.type, coordinates },
      };
    }),
  };
}
