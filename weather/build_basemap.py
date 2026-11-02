"""Build the static high-resolution base map of Morocco (run once).

Morocco is drawn as ONE seamless territory: the Moroccan Sahara, Sebta and
Mlilya are merged into a single shape with no internal border lines.
Outputs (in data/):
  basemap.png   - cinematic relief map, Morocco bright, neighbours dimmed
  geo.json      - projection parameters
"""
import json, os
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageChops, ImageEnhance
from shapely.geometry import shape, Polygon, MultiPolygon, box
from shapely.ops import unary_union
from scipy.ndimage import gaussian_filter, distance_transform_edt

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---- projection: equirectangular with cos(lat0) correction ----
LON0, LON1, LAT0, LAT1 = -20.5, 2.5, 18.0, 41.0
PPD = 200                      # pixels per degree of latitude
K = np.cos(np.radians(28.5))  # horizontal shrink
W = int(round((LON1 - LON0) * K * PPD))
H = int(round((LAT1 - LAT0) * PPD))


def proj(lon, lat):
    return ((lon - LON0) * K * PPD, (LAT1 - lat) * PPD)


def poly_px(p):
    return [proj(x, y) for x, y in p.exterior.coords]


def parts(g):
    return list(g.geoms) if isinstance(g, MultiPolygon) else [g]


# ---- load countries ----
gj = json.load(open(os.path.join(D, "countries.geojson")))
feats = {}
for f in gj["features"]:
    p = f["properties"]
    feats[p.get("ADM0_A3") or p.get("ADMIN")] = shape(f["geometry"])

frame = box(LON0, LAT0, LON1, LAT1)
mar = feats["MAR"]
sah = feats.get("SAH")
morocco = unary_union([mar, sah]) if sah is not None else mar
# Sebta, Mlilya and nearby rocks: every Spanish polygon on the African side
esp = feats["ESP"]
afr = [q for q in parts(esp) if q.centroid.y < 36.0 and -6.0 < q.centroid.x < -2.0
       and q.distance(morocco) < 0.05]
morocco = unary_union([morocco] + afr).buffer(0.004).buffer(-0.004)
# remove any interior holes / slivers so the territory is one clean shape
clean = []
for q in parts(morocco):
    clean.append(Polygon(q.exterior))
morocco = unary_union(clean)

others = []
for k, g in feats.items():
    if k in ("MAR", "SAH"):
        continue
    if g.intersects(frame):
        g2 = g.intersection(frame)
        if k == "ESP":
            g2 = unary_union([q for q in parts(g2) if not any(q.equals(a) for a in afr)])
        others.append(g2)
others = unary_union(others).difference(morocco.buffer(0.01))


def mask_of(g, w=W, h=H):
    im = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(im)
    for q in parts(g):
        if q.is_empty:
            continue
        dr.polygon(poly_px(q), fill=255)
        for hole in q.interiors:
            dr.polygon([proj(x, y) for x, y in hole.coords], fill=0)
    return im


print("canvas", W, H)
m_mask = mask_of(morocco)
o_mask = mask_of(others)
land = np.maximum(np.array(m_mask), np.array(o_mask)).astype(np.float32) / 255

# ---- relief texture from shaded relief (30 px/deg) ----
src = Image.open(os.path.join(D, "shadedrelief.jpg"))
sw, sh = src.size
spd = sw / 360.0
crop = src.crop((int((LON0 + 180) * spd), int((90 - LAT1) * spd),
                 int((LON1 + 180) * spd), int((90 - LAT0) * spd)))
relief = crop.resize((W, H), Image.BICUBIC)
rel = np.array(relief).astype(np.float32) / 255

# luminance-based fake hillshade detail boost (local contrast)
lum = rel.mean(axis=2)
detail = lum - gaussian_filter(lum, 18)
# pseudo elevation from etopo for extra shading
et = Image.open(os.path.join(D, "etopo1.jpg")).convert("L")
ew, eh = et.size
epd = ew / 360.0
ec = et.crop((int((LON0 + 180) * epd), int((90 - LAT1) * epd),
              int((LON1 + 180) * epd), int((90 - LAT0) * epd))).resize((W, H), Image.BICUBIC)
elev = gaussian_filter(np.array(ec).astype(np.float32) / 255, 7)
gy, gx = np.gradient(elev)
shade = np.clip(0.5 + (-gx * 1.0 - gy * 1.0) * 40, 0, 1)

# land colour: natural relief graded towards warm cinematic tones
col = rel.copy()
g0 = col.mean(axis=2, keepdims=True)
col = np.clip(g0 + (col - g0) * 1.7, 0, 1)          # richer colours
col = np.clip((col - 0.5) * 1.3 + 0.47, 0, 1)
col *= (0.85 + 0.3 * shade)[..., None]
col += detail[..., None] * 1.2
warm = np.array([1.04, 0.97, 0.86])
col = np.clip(col * warm, 0, 1)

# ---- ocean: deep navy with bathymetry hint ----
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
dist_coast = distance_transform_edt(land < 0.5) / PPD   # degrees from land
ocean = np.zeros((H, W, 3), np.float32)
deep = np.array([0.020, 0.055, 0.115])
shelf = np.array([0.035, 0.20, 0.32])
t = np.clip(dist_coast / 1.6, 0, 1)[..., None] ** 0.6
ocean = shelf * (1 - t) + deep * t
ocean += (elev[..., None] - 0.3) * 0.12   # depth texture
ocean = np.clip(ocean, 0, 1)

mm = np.array(m_mask).astype(np.float32) / 255
om = np.array(o_mask).astype(np.float32) / 255
mm_s = gaussian_filter(mm, 0.8)
om_s = gaussian_filter(om, 0.8)

# neighbours: darkened + desaturated
gray = col.mean(axis=2, keepdims=True)
neigh = (col * 0.3 + gray * 0.7) * 0.30 + np.array([0.01, 0.02, 0.05])

img = ocean * (1 - mm_s[..., None] - om_s[..., None]).clip(0, 1)
img += neigh * om_s[..., None]
img += col * mm_s[..., None]

# ---- glowing outline around Morocco (logo teal) ----
edge = np.array(m_mask.filter(ImageFilter.FIND_EDGES)).astype(np.float32) / 255
glow1 = gaussian_filter(edge, 2.0) * 6
glow2 = gaussian_filter(edge, 9.0) * 14
teal = np.array([0.25, 0.95, 0.80])
outside = 1 - mm_s * 0.85
img += teal * np.clip(glow2, 0, 1)[..., None] * 0.55 * outside[..., None]
img += np.array([0.75, 1.0, 0.95]) * np.clip(glow1, 0, 1)[..., None] * 0.55

# soft inner light so Morocco "lifts" from the page
inner = gaussian_filter(mm, 25)
img *= (1 + 0.12 * inner)[..., None]
img = np.clip(img, 0, 1)

out = Image.fromarray((img * 255).astype(np.uint8))
out = out.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=2))
out.save(os.path.join(D, "basemap.png"))
m_mask.save(os.path.join(D, "morocco_mask.png"))
Image.fromarray((land * 255).astype(np.uint8)).save(os.path.join(D, "land_mask.png"))
json.dump({"LON0": LON0, "LON1": LON1, "LAT0": LAT0, "LAT1": LAT1, "PPD": PPD,
           "K": float(K), "W": W, "H": H}, open(os.path.join(D, "geo.json"), "w"))
out.resize((W // 4, H // 4)).save(os.path.join(D, "basemap_preview.jpg"), quality=85)
print("done")
