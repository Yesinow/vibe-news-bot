"""Qayeb (قايب) daily Morocco weather video renderer.

usage: python3 render.py today.json out.mp4
Data file schema: see today.json. Base map must exist (build_basemap.py).
"""
import json, math, os, subprocess, sys, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import gaussian_filter, zoom as ndzoom

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
FW, FH, FPS = 1080, 1920, 30
F_XB = os.path.join(D, "fonts", "Tajawal-ExtraBold.ttf")
F_B = os.path.join(D, "fonts", "Tajawal-Bold.ttf")
F_M = os.path.join(D, "fonts", "Tajawal-Medium.ttf")

TEAL = (63, 230, 190)
CYAN = (60, 180, 255)
PURPLE = (124, 58, 210)
GOLD = (255, 196, 64)
WHITE = (255, 255, 255)

CITIES = {  # key: (arabic, lat, lon)
    "tanger": ("طنجة", 35.76, -5.83), "tetouan": ("تطوان", 35.58, -5.37),
    "alhoceima": ("الحسيمة", 35.25, -3.94), "oujda": ("وجدة", 34.68, -1.91),
    "bouarfa": ("بوعرفة", 32.53, -1.96), "kenitra": ("القنيطرة", 34.26, -6.58),
    "rabat": ("الرباط", 34.02, -6.84), "casablanca": ("الدار البيضاء", 33.57, -7.59),
    "eljadida": ("الجديدة", 33.23, -8.50), "settat": ("سطات", 33.00, -7.62),
    "safi": ("آسفي", 32.30, -9.24), "khouribga": ("خريبكة", 32.88, -6.91),
    "benimellal": ("بني ملال", 32.34, -6.35), "marrakech": ("مراكش", 31.63, -7.98),
    "meknes": ("مكناس", 33.89, -5.55), "fes": ("فاس", 34.03, -5.00),
    "ifrane": ("إفران", 33.53, -5.11), "taounate": ("تاونات", 34.54, -4.64),
    "errachidia": ("الرشيدية", 31.93, -4.42), "ouarzazate": ("ورزازات", 30.92, -6.89),
    "agadir": ("أكادير", 30.43, -9.60), "essaouira": ("الصويرة", 31.51, -9.77),
    "laayoune": ("العيون", 27.13, -13.16), "smara": ("السمارة", 26.74, -11.67),
    "dakhla": ("الداخلة", 23.68, -15.96), "aousserd": ("أوسرد", 22.55, -14.33),
    "lagouira": ("الكويرة", 20.85, -17.08), "midelt": ("ميدلت", 32.68, -4.74),
    "tantan": ("طانطان", 28.44, -11.10), "guelmim": ("كلميم", 28.99, -10.06),
    "nador": ("الناظور", 35.17, -2.93), "boujdour": ("بوجدور", 26.13, -14.48),
}
COND_AR = {"sunny": "مشمس", "partly": "غائم جزئياً", "cloudy": "غائم", "fog": "ضباب صباحاً",
           "drizzle": "قطرات متفرقة", "rain": "زخات مطرية", "thunder": "زخات رعدية",
           "snow": "ثلوج", "wind": "رياح قوية"}

geo = json.load(open(os.path.join(D, "geo.json")))


def ll2map(lat, lon):
    return ((lon - geo["LON0"]) * geo["K"] * geo["PPD"], (geo["LAT1"] - lat) * geo["PPD"])


# ---------------------------------------------------------------- utils
_fc = {}


def font(path, size):
    k = (path, size)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)
    return _fc[k]


def tsize(txt, f):
    b = f.getbbox(txt, direction="rtl", language="ar")
    return b[2] - b[0], b[3] - b[1], b


def text_img(txt, f, fill=WHITE, shadow=True, stroke=0, stroke_fill=(0, 0, 0)):
    w, h, b = tsize(txt, f)
    pad = 14
    im = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    if shadow:
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((pad - b[0] + 2, pad - b[1] + 3), txt, font=f, fill=(0, 0, 0, 170),
                                direction="rtl", language="ar")
        sh = sh.filter(ImageFilter.GaussianBlur(4))
        im = Image.alpha_composite(im, sh)
    ImageDraw.Draw(im).text((pad - b[0], pad - b[1]), txt, font=f, fill=fill, direction="rtl",
                            language="ar", stroke_width=stroke, stroke_fill=stroke_fill)
    return im


def paste(dst, src, x, y, alpha=1.0):
    """alpha-composite src onto dst at (x,y) (top-left), clipped, with global alpha."""
    if alpha <= 0.003:
        return
    x, y = int(round(x)), int(round(y))
    if alpha < 0.999:
        src = src.copy()
        a = src.getchannel("A").point(lambda v: int(v * alpha))
        src.putalpha(a)
    dst.alpha_composite(src, (max(0, x), max(0, y)),
                        (max(0, -x), max(0, -y)))


def scaled(im, s):
    if abs(s - 1) < 0.01:
        return im
    w, h = max(1, int(im.width * s)), max(1, int(im.height * s))
    return im.resize((w, h), Image.LANCZOS)


def ease(t):
    t = min(max(t, 0), 1)
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def spring(t):
    """0->1 with overshoot (pop-in)."""
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    return 1 - math.exp(-6 * t) * math.cos(9 * t)


def clamp01(v):
    return min(max(v, 0.0), 1.0)


def temp_color(t):
    stops = [(-5, (120, 160, 255)), (8, (90, 200, 255)), (16, (80, 230, 190)),
             (22, (200, 235, 90)), (27, (255, 196, 64)), (32, (255, 120, 50)), (40, (255, 60, 70))]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if t <= b:
            u = clamp01((t - a) / (b - a))
            return tuple(int(ca[i] + (cb[i] - ca[i]) * u) for i in range(3))
    return stops[-1][1]


def rounded(size, r, fill, outline=None, width=0):
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), r, fill=fill,
                                         outline=outline, width=width)
    return im


def glass(size, r=26, tint=(12, 14, 40), a=200, border=(TEAL, PURPLE), bw=3, shadow=True):
    w, h = size
    pad = 30
    im = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    if shadow:
        sh = rounded((w, h), r, (0, 0, 0, 150))
        tmp = Image.new("RGBA", im.size, (0, 0, 0, 0))
        tmp.alpha_composite(sh, (pad, pad + 8))
        im = Image.alpha_composite(im, tmp.filter(ImageFilter.GaussianBlur(14)))
    # gradient border
    g = Image.new("RGBA", (w, h))
    gx = np.linspace(0, 1, w)[None, :, None]
    c0, c1 = np.array(border[0]), np.array(border[1])
    arr = np.zeros((h, w, 4), np.uint8)
    arr[..., :3] = (c0 * (1 - gx) + c1 * gx).astype(np.uint8)
    arr[..., 3] = 255
    g = Image.fromarray(arr)
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), r, fill=255)
    ImageDraw.Draw(m).rounded_rectangle((bw, bw, w - 1 - bw, h - 1 - bw), r - bw, fill=0)
    body = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    # body vertical gradient
    gy = np.linspace(0, 1, h)[:, None, None]
    barr = np.zeros((h, w, 4), np.uint8)
    t0 = np.array(tint) + 18
    barr[..., :3] = np.clip(t0 * (1 - gy) + np.array(tint) * gy, 0, 255).astype(np.uint8)
    barr[..., 3] = a
    bm = Image.new("L", (w, h), 0)
    ImageDraw.Draw(bm).rounded_rectangle((0, 0, w - 1, h - 1), r, fill=255)
    bimg = Image.fromarray(barr)
    bimg.putalpha(Image.fromarray(np.minimum(np.array(bm), a).astype(np.uint8)))
    body = Image.alpha_composite(body, bimg)
    g.putalpha(m)
    body = Image.alpha_composite(body, g)
    # top sheen
    sheen = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(sheen).rounded_rectangle((bw + 2, bw + 2, w - bw - 3, h // 2), r - bw,
                                            fill=(255, 255, 255, 16))
    body = Image.alpha_composite(body, sheen)
    im.alpha_composite(body, (pad, pad))
    _glass_geo[id(im)] = (pad, w, h, r)
    return im, pad


_glass_geo = {}


def frost(frame, img, x, y, alpha=1.0, scale=1.0, geo_=None):
    """Frosted glass: blur the frame behind the panel body, then paste the panel."""
    g = geo_ or _glass_geo.get(id(img))
    if g is not None and alpha > 0.02:
        pad, w, h, r = g
        bx, by = int(x + pad * scale), int(y + pad * scale)
        bw, bh = int(w * scale), int(h * scale)
        x0, y0, x1, y1 = max(0, bx), max(0, by), min(FW, bx + bw), min(FH, by + bh)
        if x1 - x0 > 4 and y1 - y0 > 4:
            ox, oy = max(0, x0 - 20), max(0, y0 - 20)
            reg = frame.crop((ox, oy, min(FW, x1 + 20), min(FH, y1 + 20)))
            bl = reg.filter(ImageFilter.GaussianBlur(16)).crop((x0 - ox, y0 - oy, x1 - ox, y1 - oy))
            m = Image.new("L", (bw, bh), 0)
            ImageDraw.Draw(m).rounded_rectangle((0, 0, bw - 1, bh - 1), max(2, int(r * scale)),
                                                fill=int(255 * alpha))
            m = m.crop((x0 - bx, y0 - by, x1 - bx, y1 - by))
            frame.paste(bl, (x0, y0), m)
    paste(frame, img, x, y, alpha)


# ---------------------------------------------------------------- icons
NLOOP = 24


def _cloud_mask(S, circles):
    m = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(m)
    for cx, cy, r in circles:
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    return m


def _cloud(S, circles, top, bot, shadow=True):
    m = _cloud_mask(S, circles)
    ys = np.linspace(0, 1, S)[:, None, None]
    arr = np.zeros((S, S, 4), np.uint8)
    arr[..., :3] = (np.array(top) * (1 - ys) + np.array(bot) * ys).astype(np.uint8)
    arr[..., 3] = np.array(m)
    im = Image.fromarray(arr)
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    if shadow:
        sh = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        sh.putalpha(m.point(lambda v: v * 0.45))
        out.alpha_composite(sh.filter(ImageFilter.GaussianBlur(S // 40)), (0, S // 50))
    out.alpha_composite(im)
    return out


def _sun(S, cx, cy, r, phase):
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((cx - r * 1.9, cy - r * 1.9, cx + r * 1.9, cy + r * 1.9),
                                 fill=(255, 190, 40, 90))
    im.alpha_composite(glow.filter(ImageFilter.GaussianBlur(S // 18)))
    d = ImageDraw.Draw(im)
    for i in range(10):
        a = phase * 2 * math.pi / 10 / NLOOP * 3 + i * 2 * math.pi / 10
        r0, r1, wdt = r * 1.25, r * 1.75, r * 0.16
        p = [(cx + math.cos(a) * r0 - math.sin(a) * wdt, cy + math.sin(a) * r0 + math.cos(a) * wdt),
             (cx + math.cos(a) * r1, cy + math.sin(a) * r1),
             (cx + math.cos(a) * r0 + math.sin(a) * wdt, cy + math.sin(a) * r0 - math.cos(a) * wdt)]
        d.polygon(p, fill=(255, 205, 60, 255))
    # disc radial gradient
    disc = np.zeros((S, S, 4), np.uint8)
    yy, xx = np.mgrid[0:S, 0:S]
    dd = np.sqrt((xx - cx + r * 0.3) ** 2 + (yy - cy + r * 0.3) ** 2) / (r * 1.6)
    c0, c1 = np.array([255, 245, 170]), np.array([255, 150, 30])
    u = np.clip(dd, 0, 1)[..., None]
    disc[..., :3] = (c0 * (1 - u) + c1 * u).astype(np.uint8)
    inside = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
    disc[..., 3] = inside * 255
    im.alpha_composite(Image.fromarray(disc))
    return im


def make_icon(kind, phase, size):
    S = 256
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    big = [(100, 150, 46), (140, 125, 58), (180, 155, 40), (75, 165, 30), (140, 168, 40), (195, 168, 26)]
    if kind == "sunny":
        im.alpha_composite(_sun(S, 128, 128, 58, phase))
    elif kind == "partly":
        im.alpha_composite(_sun(S, 100, 95, 46, phase))
        sm = [(x + 12, y + 18, r * 0.92) for x, y, r in big]
        im.alpha_composite(_cloud(S, sm, (255, 255, 255), (205, 215, 232)))
    elif kind in ("cloudy", "fog"):
        im.alpha_composite(_cloud(S, [(x, y - 10, r) for x, y, r in big], (245, 248, 255), (178, 190, 212)))
        if kind == "fog":
            d = ImageDraw.Draw(im)
            off = math.sin(phase / NLOOP * 2 * math.pi) * 10
            for i, yy in enumerate((198, 222, 244)):
                o = off if i % 2 == 0 else -off
                d.rounded_rectangle((52 + o, yy - 7, 204 + o, yy + 7), 7, fill=(225, 232, 245, 235))
    elif kind in ("rain", "drizzle", "thunder", "snow"):
        dark = kind == "thunder"
        top, bot = ((140, 150, 175), (70, 78, 105)) if dark else ((240, 244, 252), (165, 178, 205))
        im.alpha_composite(_cloud(S, [(x, y - 30, r) for x, y, r in big], top, bot))
        d = ImageDraw.Draw(im)
        u = phase / NLOOP
        if kind == "drizzle":
            for i, x0 in enumerate((95, 135, 175)):
                yy = 190 + ((u * 60 + i * 20) % 60)
                a = int(255 * (1 - ((u * 60 + i * 20) % 60) / 60))
                d.ellipse((x0 - 7, yy - 7, x0 + 7, yy + 7), fill=(110, 200, 255, a))
        elif kind == "snow":
            for i, x0 in enumerate((95, 135, 175)):
                yy = 185 + ((u * 70 + i * 23) % 70)
                d.ellipse((x0 - 9, yy - 9, x0 + 9, yy + 9), fill=(255, 255, 255, 240))
        else:
            n = 4 if kind == "rain" else 3
            for i in range(n):
                x0 = 80 + i * 32
                k = (u * 80 + i * 27) % 80
                y0 = 175 + k
                a = int(255 * (1 - k / 80))
                d.line((x0, y0, x0 - 12, y0 + 30), fill=(90, 190, 255, a), width=9)
        if kind == "thunder":
            flash = 1.0 if (phase % 12) in (0, 1, 3) else 0.75
            bolt = [(140, 150), (104, 212), (132, 212), (112, 256), (172, 188), (142, 188), (162, 150)]
            gl = Image.new("RGBA", (S, S), (0, 0, 0, 0))
            ImageDraw.Draw(gl).polygon(bolt, fill=(255, 220, 60, int(200 * flash)))
            im.alpha_composite(gl.filter(ImageFilter.GaussianBlur(10)))
            ImageDraw.Draw(im).polygon(bolt, fill=(255, 226, 70, int(255 * flash)))
    elif kind == "wind":
        d = ImageDraw.Draw(im)
        off = (phase / NLOOP) * 40
        for i, yy in enumerate((90, 130, 170)):
            d.arc((60 + off - 20 * i, yy - 20, 220 + off - 20 * i, yy + 60), 200, 340,
                  fill=(220, 235, 255), width=12)
    return im.resize((size, size), Image.LANCZOS)


_icache = {}


def icon_frames(kind, size):
    k = (kind, size)
    if k not in _icache:
        _icache[k] = [make_icon(kind, p, size) for p in range(NLOOP)]
    return _icache[k]


# ---------------------------------------------------------------- badges
def make_badge(key, c, big=False, safi=False):
    name = CITIES[key][0]
    sc = 1.25 if big else 1.0
    fn = font(F_B, int(30 * sc))
    fmax = font(F_XB, int(44 * sc))
    fmin = font(F_M, int(28 * sc))
    nm = text_img(name, fn, (255, 255, 255), shadow=False)
    tmax = text_img(f"{c['max']}°", fmax, temp_color(c["max"]), shadow=False)
    tmin = text_img(f"{c['min']}°", fmin, (185, 200, 225), shadow=False)
    isz = int(72 * sc)
    pad = 14
    tw = max(nm.width - 28, tmax.width - 28 + tmin.width - 28 + 10)
    w = int(pad + isz + 8 + tw + pad + 6)
    h = int(92 * sc)
    border = (GOLD, (255, 120, 60)) if safi else (TEAL, PURPLE)
    g, gp = glass((w, h), r=int(22 * sc), border=border, bw=3 if safi else 2, a=150)
    # text block on the right (RTL)
    right = gp + w - pad
    paste(g, nm, right - nm.width + 14, gp + int(4 * sc) - 14 + 6)
    paste(g, tmax, right - tmax.width + 14, gp + int(40 * sc) - 14)
    paste(g, tmin, right - tmax.width + 14 - 10 - tmin.width + 28 - 14, gp + int(52 * sc) - 14)
    icon_xy = (gp + pad - 4, gp + (h - isz) // 2)
    return {"geo": _glass_geo[id(g)], "img": g, "pad": gp, "w": w, "h": h, "icon": icon_xy, "isz": isz, "kind": c["cond"]}


# ---------------------------------------------------------------- map renderer
class MapR:
    def __init__(self):
        base = Image.open(os.path.join(D, "basemap.png")).convert("RGB")
        self.lv = [base]
        for _ in range(3):
            b = self.lv[-1]
            self.lv.append(b.resize((b.width // 2, b.height // 2), Image.LANCZOS))

    def render(self, cx, cy, s, vy):
        L = 0
        while L < len(self.lv) - 1 and s * (2 ** (L + 1)) <= 1.25:
            L += 1
        f = 1 / 2 ** L
        im = self.lv[L]
        se = s / f
        a = 1 / se
        data = (a, 0, cx * f - FW / 2 * a, 0, a, cy * f - vy * a)
        return im.transform((FW, FH), Image.AFFINE, data, resample=Image.BILINEAR,
                            fillcolor=(8, 28, 52))


def cloud_layers(data):
    """Return quarter-res geographic cloud alpha field and a tileable noise texture."""
    gw, gh = geo["W"] // 4, geo["H"] // 4
    field = np.zeros((gh, gw), np.float32)
    weight = {"partly": 0.35, "cloudy": 0.7, "fog": 0.45, "drizzle": 0.6, "rain": 0.85,
              "thunder": 1.0, "snow": 0.8}
    for k, c in data["cities"].items():
        wgt = weight.get(c["cond"], 0)
        if wgt <= 0 or k not in CITIES:
            continue
        x, y = ll2map(CITIES[k][1], CITIES[k][2])
        x, y = int(x / 4), int(y / 4)
        r = 32 if c["cond"] != "thunder" else 55
        yy, xx = np.ogrid[0:gh, 0:gw]
        field = np.maximum(field, wgt * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * r * r)))
    rng = np.random.default_rng(7)
    n = np.zeros((gh, gw), np.float32)
    for sig, amp in ((40, 1.0), (16, 0.55), (6, 0.3), (2.5, 0.15)):
        layer = gaussian_filter(rng.standard_normal((gh, gw)).astype(np.float32), sig, mode="wrap")
        n += amp * layer / layer.std()
    n = (n - n.min()) / (n.max() - n.min())
    return field, n


# ---------------------------------------------------------------- scenes
def cam_for(lat, lon, ppd):
    x, y = ll2map(lat, lon)
    return (x, y, ppd / geo["PPD"])


def build(data, out_path, preview=None):
    random.seed(3)
    mr = MapR()
    field, noise = cloud_layers(data)
    gh, gw = field.shape
    cities = {k: v for k, v in data["cities"].items() if k in CITIES}

    VY = 1010
    FULL = cam_for(27.8, -9.1, 68)
    NORTH = cam_for(34.25, -5.35, 175)
    CENTER = cam_for(32.15, -8.15, 190)
    SOUTH = cam_for(24.0, -13.8, 100)
    SEA = cam_for(28.6, -11.0, 58)
    SURF = cam_for(31.35, -9.7, 250)
    OUTLOOK = cam_for(32.6, -10.0, 75)

    # timeline
    T_INTRO = 4.2
    scenes = [  # (name, cam, start, end, safe area y0,y1)
        ("full", FULL, 3.8, 13.0, 330, 1530),
        ("temp", FULL, 13.6, 21.5, 330, 1530),
        ("north", NORTH, 24.0, 32.5, 360, 1700),
        ("center", CENTER, 35.0, 43.5, 360, 1390),
        ("south", SOUTH, 46.0, 54.0, 360, 1420),
        ("sea", SEA, 56.6, 65.5, 330, 1480),
        ("surf", SURF, 68.0, 78.0, 330, 1300),
        ("outlook", OUTLOOK, 80.5, 89.5, 330, 1300),
        ("final", FULL, 92.0, 99.5, 330, 900),
    ]
    T_OUT0, DUR = 99.5, 104.0
    BADGE_SCENES = {"full", "north", "center", "south", "final"}
    SC = {sc[0]: sc for sc in scenes}
    nfr = int(DUR * FPS)

    def camera(t):
        if t <= scenes[0][3]:
            c = scenes[0][1]
            u = clamp01((t - scenes[0][2]) / (scenes[0][3] - scenes[0][2]))
            return (c[0], c[1] + 30 * (1 - u), c[2] * (0.93 + 0.07 * ease(u)))
        for a, b in zip(scenes, scenes[1:]):
            if a[3] <= t <= b[2]:
                u = ease((t - a[3]) / (b[2] - a[3]))
                ca, cb = a[1], b[1]
                ls = math.log(ca[2]) * (1 - u) + math.log(cb[2]) * u - 0.35 * math.sin(math.pi * u)
                return (ca[0] + (cb[0] - ca[0]) * u, ca[1] + (cb[1] - ca[1]) * u, math.exp(ls))
            if b[2] <= t <= b[3]:
                u = clamp01((t - b[2]) / (b[3] - b[2]))
                c = b[1]
                return (c[0], c[1], c[2] * (1 + 0.05 * u))
        c = scenes[-1][1]
        return (c[0], c[1], c[2] * 1.05)

    def to_screen(cam, lat, lon):
        x, y = ll2map(lat, lon)
        return ((x - cam[0]) * cam[2] + FW / 2, (y - cam[1]) * cam[2] + VY)

    # -------- static layouts per scene (greedy, collision free)
    badges_small = {k: make_badge(k, c, safi=(k == "safi")) for k, c in cities.items()}
    badge_safi_big = make_badge("safi", cities["safi"], big=True, safi=True) if "safi" in cities else None

    def layout(scene):
        name, cam, t0, t1, y0, y1 = scene
        mid = (cam[0], cam[1], cam[2] * 1.025)
        placed, out = [], []
        imp = ["safi", "lagouira", "tanger", "fes", "oujda", "rabat", "casablanca", "marrakech", "agadir", "laayoune",
               "dakhla", "meknes", "tetouan", "alhoceima", "errachidia", "ouarzazate", "smara",
               "ifrane", "eljadida", "essaouira", "benimellal", "kenitra", "settat", "taounate", "midelt",
               "khouribga", "bouarfa", "aousserd", "nador", "tantan", "guelmim", "boujdour"]
        order = [k for k in imp if k in cities] + [k for k in cities if k not in imp]
        if name == "full":
            pri = ["safi", "lagouira", "tanger", "oujda", "fes", "casablanca", "marrakech", "errachidia", "agadir",
                   "ouarzazate", "laayoune", "smara", "dakhla", "lagouira", "ifrane", "rabat"]
            order = [k for k in pri if k in cities]
        for k in order:
            b = badges_small[k]
            if name in ("center",) and k == "safi" and badge_safi_big:
                b = badge_safi_big
            px, py = to_screen(mid, CITIES[k][1], CITIES[k][2])
            if not (40 < px < FW - 40 and y0 < py < y1):
                continue
            # don't let badges cover other cities' pins
            sides = ["L", "R", "U", "D"] if px > FW * 0.35 else ["R", "U", "D", "L"]
            if k in ("safi", "casablanca", "eljadida", "rabat", "kenitra", "agadir", "essaouira",
                     "laayoune", "dakhla", "lagouira", "tanger"):
                sides = ["L", "U", "D", "R"]
            ok = None
            for sd in sides:
                gap = 22
                if sd == "L":
                    bx, by = px - gap - b["w"], py - b["h"] / 2
                elif sd == "R":
                    bx, by = px + gap, py - b["h"] / 2
                elif sd == "U":
                    bx, by = px - b["w"] / 2, py - gap - b["h"]
                else:
                    bx, by = px - b["w"] / 2, py + gap
                box = (bx - 12, by - 12, bx + b["w"] + 12, by + b["h"] + 12)
                if box[0] < 12 or box[2] > FW - 12 or box[1] < y0 - 40 or box[3] > y1 + 60:
                    continue
                if any(not (box[2] < q[0] or box[0] > q[2] or box[3] < q[1] or box[1] > q[3]) for q in placed):
                    continue
                # avoid covering pins of already placed cities
                if any(box[0] < p[0] < box[2] and box[1] < p[1] < box[3] for p in [o[1] for o in out]):
                    continue
                ok = (sd, bx, by, box)
                break
            if ok is None:
                continue
            placed.append(ok[3])
            placed.append((px - 20, py - 20, px + 20, py + 20))
            out.append((k, (px, py), (ok[1], ok[2]), b))
        return out

    placements = {}
    for sc in scenes:
        placements[sc[0]] = layout(sc) if sc[0] in BADGE_SCENES else []
        _badge_pos[sc[0]] = {k: pos for k, _, pos, _ in placements[sc[0]]}

    # -------- static overlays
    logo = Image.open(os.path.join(D, "logo_src.png")).convert("RGBA")

    def header_img():
        im = Image.new("RGBA", (FW, 330), (0, 0, 0, 0))
        g = np.zeros((330, FW, 4), np.uint8)
        a = np.linspace(235, 0, 330) ** 1.0
        g[..., 0], g[..., 1], g[..., 2] = 6, 8, 26
        g[..., 3] = a[:, None].astype(np.uint8)
        im = Image.alpha_composite(im, Image.fromarray(g))
        lg = logo.resize((168, 168), Image.LANCZOS)
        glow = Image.new("RGBA", (260, 260), (0, 0, 0, 0))
        ImageDraw.Draw(glow).ellipse((30, 30, 230, 230), fill=(80, 220, 200, 120))
        paste(im, glow.filter(ImageFilter.GaussianBlur(18)), FW - 40 - 168 - 46, 42 - 46)
        paste(im, lg, FW - 40 - 168, 42)
        t1 = text_img("حالة الطقس في المغرب", font(F_XB, 64))
        paste(im, t1, FW - 40 - 168 - 28 - t1.width + 14, 50)
        t2 = text_img(data["date_ar"], font(F_B, 40), TEAL)
        paste(im, t2, FW - 40 - 168 - 28 - t2.width + 14, 128)
        # accent line
        line = Image.new("RGBA", (560, 5))
        arr = np.zeros((5, 560, 4), np.uint8)
        u = np.linspace(0, 1, 560)[None, :, None]
        arr[..., :3] = (np.array(PURPLE) * (1 - u) + np.array(TEAL) * u).astype(np.uint8)
        arr[..., 3] = (255 * np.minimum(1, u[..., 0] * 3)).astype(np.uint8)
        paste(im, Image.fromarray(arr), FW - 40 - 168 - 28 - 560, 196)
        return im

    HEADER = header_img()

    # ticker
    TK_Y, TK_H = 1800, 84
    tick_txt = text_img(data["ticker"] + "      •      ", font(F_B, 40), WHITE, shadow=False)

    def ticker_band():
        im = Image.new("RGBA", (FW, TK_H + 120), (0, 0, 0, 0))
        g = np.zeros((TK_H + 120, FW, 4), np.uint8)
        a = np.concatenate([np.linspace(0, 200, 120), np.full(TK_H, 235)])
        g[..., 0], g[..., 1], g[..., 2] = 8, 6, 28
        g[..., 3] = a[:, None].astype(np.uint8)
        im = Image.alpha_composite(im, Image.fromarray(g))
        # top accent
        arr = np.zeros((4, FW, 4), np.uint8)
        u = np.linspace(0, 1, FW)[None, :, None]
        arr[..., :3] = (np.array(TEAL) * (1 - u) + np.array(PURPLE) * u).astype(np.uint8)
        arr[..., 3] = 255
        paste(im, Image.fromarray(arr), 0, 120)
        return im

    TBAND = ticker_band()
    tag = rounded((230, TK_H - 4), 0, (0, 0, 0, 0))
    tg = np.zeros((TK_H - 4, 230, 4), np.uint8)
    u = np.linspace(0, 1, 230)[None, :, None]
    tg[..., :3] = (np.array(TEAL) * u + np.array((30, 160, 150)) * (1 - u)).astype(np.uint8)
    tg[..., 3] = 255
    tag = Image.fromarray(tg)
    tt = text_img("الأرصاد", font(F_XB, 42), (6, 20, 30), shadow=False)
    paste(tag, tt, (230 - tt.width) // 2, (TK_H - 4 - tt.height) // 2 + 2)

    # region chips
    def chip(txt, color=TEAL):
        t = text_img(txt, font(F_XB, 40), WHITE, shadow=False)
        g, gp = glass((t.width + 30, 74), r=37, border=(color, PURPLE), bw=3, a=165)
        paste(g, t, gp + 15, gp + (74 - t.height) // 2 + 3)
        return g

    CHIPS = {"full": chip("المغرب من طنجة إلى الكويرة"), "north": chip("الشمال والشرق والوسط"),
             "center": chip("آسفي وجهة مراكش"), "south": chip("الأقاليم الجنوبية", GOLD),
             "final": chip("تنبيهات اليوم", (255, 120, 60)),
             "temp": chip("طبقة درجات الحرارة العليا", (255, 150, 60)),
             "sea": chip("حالة البحر • للبحارة", CYAN),
             "surf": chip("ركوب الأمواج: آسفي وتاغازوت", TEAL),
             "outlook": chip("الأيام المقبلة", PURPLE)}

    def small_src(txt):
        return text_img(txt, font(F_M, 28), (200, 210, 235))

    SRC = {"sea": small_src("حسب " + data.get("source", "") + " و windfinder"),
           "surf": small_src("المصدر: " + data.get("surf_source", "")),
           "outlook": small_src(data.get("outlook_note", "")),
           "temp": small_src("حسب " + data.get("source", ""))}

    # -------- panels
    def stat_card(title, key, accent):
        c = cities[key]
        w, h = 470, 170
        g, gp = glass((w, h), r=30, border=(accent, PURPLE), bw=3, a=165)
        t = text_img(title, font(F_B, 34), (200, 210, 235), shadow=False)
        paste(g, t, gp + w - 24 - t.width + 14, gp + 10)
        n = text_img(CITIES[key][0], font(F_XB, 52), WHITE, shadow=False)
        paste(g, n, gp + w - 24 - n.width + 14, gp + 54)
        val = c["max"] if "أعلى" in title else c["min"]
        v = text_img(f"{val}°", font(F_XB, 88), temp_color(val), shadow=False)
        paste(g, v, gp + 26 - 14, gp + (h - v.height) // 2 + 6)
        return g

    hot = max(cities, key=lambda k: (cities[k]["max"], -CITIES[k][1]))
    cold = min(cities, key=lambda k: cities[k]["min"])
    CARD_HOT = stat_card("الأعلى حرارة اليوم", hot, (255, 120, 60))
    CARD_COLD = stat_card("الأدنى حرارة اليوم", cold, CYAN)

    def safi_panel():
        c = cities["safi"]
        w, h = 1000, 330
        g, gp = glass((w, h), r=36, border=(GOLD, (255, 110, 60)), bw=4, a=165)
        t = text_img("آسفي", font(F_XB, 88), WHITE)
        paste(g, t, gp + w - 40 - t.width + 14, gp + 18)
        ct = text_img(COND_AR.get(c["cond"], ""), font(F_B, 40), GOLD, shadow=False)
        paste(g, ct, gp + w - 40 - ct.width + 14, gp + 128)
        note = data.get("safi_note", "")
        if note:
            nt = text_img(note, font(F_M, 34), (215, 222, 240), shadow=False)
            paste(g, nt, gp + w - 40 - nt.width + 14, gp + 186)
        sea = data.get("sea", [])
        if sea:
            st = text_img("البحر: " + sea[0]["state"], font(F_M, 34), (140, 210, 255), shadow=False)
            paste(g, st, gp + w - 40 - st.width + 14, gp + 240)
        mx = text_img(f"{c['max']}°", font(F_XB, 130), temp_color(c["max"]), shadow=False)
        mn = text_img(f"{c['min']}°", font(F_XB, 90), (170, 200, 240), shadow=False)
        lab1 = text_img("العليا", font(F_B, 32), (200, 205, 225), shadow=False)
        lab2 = text_img("الدنيا", font(F_B, 32), (200, 205, 225), shadow=False)
        paste(g, lab1, gp + 40 + (mx.width - lab1.width) // 2, gp + 36)
        paste(g, mx, gp + 40, gp + 80)
        paste(g, lab2, gp + 270 + (mn.width - lab2.width) // 2, gp + 36)
        paste(g, mn, gp + 270, gp + 110)
        return g, (gp + 450, gp + 22)

    SAFI_P, SAFI_ICON_POS = safi_panel() if "safi" in cities else (None, None)

    LEVEL_COL = {"high": (255, 120, 50), "medium": (255, 200, 60), "low": (90, 200, 255)}

    def alert_rows(alerts, w):
        rows = []
        for a in alerts:
            col = LEVEL_COL.get(a.get("level"), (200, 200, 200))
            h = 118
            r = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            d = ImageDraw.Draw(r)
            d.rounded_rectangle((w - 14, 10, w - 2, h - 10), 6, fill=col + (255,))
            t = text_img(a["title"], font(F_XB, 42), WHITE, shadow=False)
            paste(r, t, w - 34 - t.width + 14, 2)
            s = text_img(a["areas"], font(F_M, 33), (205, 212, 235), shadow=False)
            paste(r, s, w - 34 - s.width + 14, 56)
            # warning triangle
            tri = Image.new("RGBA", (70, 64), (0, 0, 0, 0))
            ImageDraw.Draw(tri).polygon([(35, 2), (68, 62), (2, 62)], fill=col + (255,))
            ex = text_img("!", font(F_XB, 44), (20, 20, 30), shadow=False)
            paste(tri, ex, 35 - ex.width // 2, 6)
            paste(r, tri, 10, (h - 64) // 2)
            rows.append(r)
        return rows

    def alerts_panel():
        al = data.get("alerts", [])
        sea = data.get("sea", [])
        w = 1000
        rows = alert_rows(al, w - 60)
        h = 150 + sum(r.height for r in rows) + (60 + 52 * len(sea) if sea else 0) + 20
        g, gp = glass((w, h), r=36, border=((255, 120, 60), PURPLE), bw=4, a=165)
        ttl = "نشرة إنذارية" if data.get("alerts_official") else "تنبيهات الطقس لليوم"
        t = text_img(ttl, font(F_XB, 56), WHITE)
        paste(g, t, gp + w - 36 - t.width + 14, gp + 16)
        if not data.get("alerts_official"):
            s = text_img("حسب توقعات " + data.get("source", ""), font(F_M, 30), (180, 190, 215), shadow=False)
            paste(g, s, gp + w - 36 - s.width + 14, gp + 84)
        y = gp + 140
        for r in rows:
            paste(g, r, gp + 30, y)
            y += r.height
        if sea:
            ln = Image.new("RGBA", (w - 60, 2), (120, 140, 200, 120))
            paste(g, ln, gp + 30, y + 8)
            st = text_img("حالة البحر", font(F_XB, 40), (140, 210, 255), shadow=False)
            paste(g, st, gp + w - 36 - st.width + 14, y + 16)
            y += 70
            for s in sea:
                tx = text_img(f"{s['zone']}: {s['state']}", font(F_M, 34), (225, 230, 245), shadow=False)
                paste(g, tx, gp + w - 36 - tx.width + 14, y - 8)
                y += 52
        return g

    ALERTS_P = alerts_panel()

    def south_card():
        al = [a for a in data.get("alerts", []) if a.get("level") == "high"]
        if not al:
            return None
        rows = alert_rows(al[:1], 940)
        w, h = 1000, 150
        g, gp = glass((w, h), r=32, border=((255, 120, 60), GOLD), bw=4, a=165)
        paste(g, rows[0], gp + 30, gp + 16)
        return g

    SOUTH_P = south_card()

    # -------- intro / outro backgrounds
    def aurora(t, a_scale=1.0):
        w, h = 216, 384
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.zeros((h, w, 3), np.float32)
        for (cx, cy, rad, col, sp) in ((0.3, 0.35, 0.45, TEAL, 0.7), (0.75, 0.55, 0.5, PURPLE, 0.5),
                                       (0.5, 0.8, 0.45, CYAN, 0.9)):
            px = (cx + 0.12 * math.sin(t * sp + cy * 5)) * w
            py = (cy + 0.08 * math.cos(t * sp * 1.3 + cx * 4)) * h
            d = ((xx - px) ** 2 + (yy - py) ** 2) / (rad * w) ** 2
            r += np.exp(-d * 2.2)[..., None] * np.array(col) / 255 * 0.55
        r += np.array([0.02, 0.02, 0.07])
        im = Image.fromarray((np.clip(r * a_scale, 0, 1) * 255).astype(np.uint8)).resize((FW, FH), Image.BICUBIC)
        return im.convert("RGBA")

    def intro_layer(t):
        im = aurora(t)
        u = clamp01(t / 1.0)
        s = 0.6 + 0.4 * spring(u)
        size = int(520 * s)
        lg = logo.resize((size, size), Image.LANCZOS)
        glow = Image.new("RGBA", (size + 200, size + 200), (0, 0, 0, 0))
        ImageDraw.Draw(glow).ellipse((60, 60, size + 140, size + 140),
                                     fill=(90, 230, 210, int(110 + 60 * math.sin(t * 4))))
        cy = 700
        paste(im, glow.filter(ImageFilter.GaussianBlur(40)), FW / 2 - (size + 200) / 2, cy - (size + 200) / 2, u)
        paste(im, lg, FW / 2 - size / 2, cy - size / 2, clamp01(t / 0.4))
        a2 = clamp01((t - 0.7) / 0.5)
        t1 = text_img("نشرة الطقس", font(F_XB, 110), WHITE)
        paste(im, t1, FW / 2 - t1.width / 2, 1040 + 40 * (1 - ease(a2)), a2)
        a3 = clamp01((t - 1.1) / 0.5)
        t2 = text_img(data["date_ar"], font(F_B, 56), TEAL)
        paste(im, t2, FW / 2 - t2.width / 2, 1190 + 40 * (1 - ease(a3)), a3)
        a4 = clamp01((t - 1.5) / 0.5)
        t3 = text_img("من طنجة إلى الكويرة", font(F_M, 46), (210, 215, 240))
        paste(im, t3, FW / 2 - t3.width / 2, 1290, a4)
        return im

    def outro_layer(t):
        im = aurora(t + 10)
        u = clamp01(t / 0.8)
        size = int(420 * (0.7 + 0.3 * spring(u)))
        lg = logo.resize((size, size), Image.LANCZOS)
        paste(im, lg, FW / 2 - size / 2, 760 - size / 2, u)
        a2 = clamp01((t - 0.5) / 0.5)
        t1 = text_img("تابعوا ڤايب كل صباح", font(F_XB, 76), WHITE)
        paste(im, t1, FW / 2 - t1.width / 2, 1040, a2)
        t2 = text_img("المصدر: " + data.get("source", ""), font(F_M, 40), (190, 200, 225))
        paste(im, t2, FW / 2 - t2.width / 2, 1160, a2)
        return im

    # -------- frame loop
    # ================= extra layers =================
    land_q = np.array(Image.open(os.path.join(D, "land_mask.png")).resize((gw, gh), Image.BILINEAR)).astype(np.float32) / 255
    mor_q = np.array(Image.open(os.path.join(D, "morocco_mask.png")).resize((gw, gh), Image.BILINEAR)).astype(np.float32) / 255
    YQ, XQ = np.mgrid[0:gh, 0:gw].astype(np.float32)
    LATQ = geo["LAT1"] - YQ * 4 / geo["PPD"]
    LONQ = geo["LON0"] + XQ * 4 / (geo["PPD"] * geo["K"])

    def to_screen_layer(rgba_arr, cam):
        im = Image.fromarray(rgba_arr, "RGBA")
        a = 1 / (cam[2] * 4)
        return im.transform((FW, FH), Image.AFFINE, (a, 0, cam[0] / 4 - FW / 2 * a, 0, a, cam[1] / 4 - VY * a),
                            Image.BILINEAR)

    LUT = np.array([temp_color(v / 10) for v in range(-50, 451)], np.float32)

    def build_temp_layer():
        num = np.zeros((gh, gw), np.float32)
        den = np.zeros((gh, gw), np.float32)
        for k, c in cities.items():
            x, y = ll2map(CITIES[k][1], CITIES[k][2])
            d2 = (XQ - x / 4) ** 2 + (YQ - y / 4) ** 2 + 30.0
            w = 1 / d2 ** 1.6
            num += w * c["max"]
            den += w
        T = num / den
        idx = np.clip(((T + 5) * 10).astype(int), 0, len(LUT) - 1)
        rgb = LUT[idx]
        m = gaussian_filter(mor_q, 1.2)
        arr = np.zeros((gh, gw, 4), np.uint8)
        arr[..., :3] = rgb.astype(np.uint8)
        arr[..., 3] = (np.clip(m, 0, 1) * 175).astype(np.uint8)
        return arr

    TEMP_L = build_temp_layer()

    def legend_panel(title, stops, labels, w=1000, h=170):
        g, gp = glass((w, h), r=30, border=(TEAL, PURPLE), bw=3, a=165)
        t = text_img(title, font(F_XB, 38), WHITE, shadow=False)
        paste(g, t, gp + w - 30 - t.width + 14, gp + 8)
        bw_, bx, by = w - 80, gp + 40, gp + 78
        bar = np.zeros((26, bw_, 4), np.uint8)
        for i in range(bw_):
            bar[:, i, :3] = stops(i / (bw_ - 1))
        bar[..., 3] = 255
        bm = Image.new("L", (bw_, 26), 0)
        ImageDraw.Draw(bm).rounded_rectangle((0, 0, bw_ - 1, 25), 13, fill=255)
        bi = Image.fromarray(bar)
        bi.putalpha(bm)
        paste(g, bi, bx, by)
        for u, lab in labels:
            lt = text_img(lab, font(F_B, 28), (225, 230, 245), shadow=False)
            paste(g, lt, bx + u * (bw_ - 1) - lt.width / 2, by + 26)
        return g

    TEMP_LEG = legend_panel("درجات الحرارة العليا لليوم", lambda u: temp_color(5 + u * 30),
                            [(0, "5°"), (1 / 3, "15°"), (2 / 3, "25°"), (1, "35°")])
    TEMP_LABEL_KEYS = [k for k in ["tanger", "oujda", "fes", "casablanca", "safi", "marrakech", "errachidia",
                                   "agadir", "laayoune", "dakhla", "lagouira"] if k in cities]

    def temp_label(k):
        c = cities[k]
        im = Image.new("RGBA", (200, 120), (0, 0, 0, 0))
        v = text_img(f"{c['max']}°", font(F_XB, 46), WHITE, stroke=2, stroke_fill=(20, 20, 40))
        n = text_img(CITIES[k][0], font(F_B, 26), (240, 240, 250), stroke=2, stroke_fill=(20, 20, 40))
        paste(im, v, 100 - v.width / 2, 0)
        paste(im, n, 100 - n.width / 2, 50)
        return im

    TEMP_LABELS = {k: temp_label(k) for k in TEMP_LABEL_KEYS}

    # ---- sea / waves
    def wave_color(h):
        stops = [(0, (40, 110, 255)), (0.8, (50, 200, 230)), (1.6, (90, 230, 160)), (2.5, (255, 210, 60)),
                 (4, (255, 80, 60))]
        for (a, ca), (b, cb) in zip(stops, stops[1:]):
            if h <= b:
                u = clamp01((h - a) / (b - a))
                return tuple(int(ca[i] + (cb[i] - ca[i]) * u) for i in range(3))
        return stops[-1][1]

    zones = data.get("sea_zones", [])
    from scipy.ndimage import distance_transform_edt
    dist_q = distance_transform_edt(land_q < 0.5)
    ocean_a = np.exp(-dist_q / 45.0) * (1 - land_q)
    atl = [z for z in zones if z["lon"] < -5.6]
    if atl:
        zl = sorted(atl, key=lambda z: z["lat"])
        H = np.interp(LATQ, [z["lat"] for z in zl], [z.get("h", 1.0) for z in zl])
    else:
        H = np.ones_like(LATQ)
    med = [z for z in zones if z["lon"] >= -5.6]
    if med:
        mm_ = (LONQ > -5.6) & (LATQ > 35.0)
        H = np.where(mm_, med[0].get("h", 0.5), H)
    WLUT = np.array([wave_color(v / 20) for v in range(0, 81)], np.float32)
    WRGB = WLUT[np.clip((H * 20).astype(int), 0, 80)]
    sd = math.radians(data.get("swell_from_deg", 315))
    # travel direction in map coords (x east, y south): from NW -> toward SE
    vx, vy = -math.sin(sd), math.cos(sd)
    PROJ = XQ * vx + YQ * vy

    def wave_layer(t, strength=1.0):
        crest = 0.55 * (0.5 + 0.5 * np.sin(PROJ / 9.0 * 2 * math.pi / 3.2 - t * 1.8)) ** 6
        a = ocean_a * (0.42 + 0.5 * crest) * strength
        arr = np.zeros((gh, gw, 4), np.uint8)
        arr[..., :3] = np.clip(WRGB * (1 - 0.6 * crest[..., None]) + 255 * 0.6 * crest[..., None], 0, 255).astype(np.uint8)
        arr[..., 3] = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        return arr

    def zone_card(z):
        n = text_img(z["zone"], font(F_B, 28), WHITE, shadow=False)
        st = text_img(z["state"], font(F_M, 26), (200, 215, 240), shadow=False)
        wv = text_img(z["waves"], font(F_XB, 30), wave_color(z.get("h", 1)), shadow=False)
        w = max(n.width, st.width, wv.width) - 28 + 40
        h = 128
        g, gp = glass((w, h), r=22, border=(CYAN, PURPLE), bw=2, a=160)
        for i, im in enumerate((n, st, wv)):
            paste(g, im, gp + w - 20 - im.width + 14, gp + 2 + i * 38)
        return g

    ZONE_C = [(z, zone_card(z)) for z in zones]
    SEA_LEG = legend_panel("ارتفاع الموج (م) • " + data.get("sea_note", "")[:0] + "للبحارة",
                           lambda u: wave_color(u * 4), [(0, "0"), (0.25, "1"), (0.5, "2"), (0.75, "3"), (1, "4")],
                           h=150)
    SEA_NOTE = text_img(data.get("sea_note", ""), font(F_M, 30), (220, 230, 250))

    # ---- surf cards
    def arrow_img(deg_from, size=110, col=TEAL):
        im = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        c = size
        d.polygon([(c, c - size * 0.85), (c + size * 0.45, c + size * 0.1), (c + size * 0.15, c + size * 0.05),
                   (c + size * 0.15, c + size * 0.8), (c - size * 0.15, c + size * 0.8), (c - size * 0.15, c + size * 0.05),
                   (c - size * 0.45, c + size * 0.1)], fill=col + (255,))
        # arrow points up = travelling north; travel direction = from + 180
        travel = (deg_from + 180) % 360
        im = im.rotate(-travel, resample=Image.BICUBIC)
        return im.resize((size, size), Image.LANCZOS)

    def surf_card(sp):
        w, h = 490, 520
        g, gp = glass((w, h), r=32, border=(TEAL, CYAN) if sp["name"] != "آسفي" else (GOLD, (255, 120, 60)),
                      bw=3, a=170)
        R_ = gp + w - 26
        n = text_img(sp["name"], font(F_XB, 54), WHITE, shadow=False)
        paste(g, n, R_ - n.width + 14, gp + 4)
        s_ = text_img(sp["spot"], font(F_M, 28), (190, 205, 230), shadow=False)
        paste(g, s_, R_ - s_.width + 14, gp + 72)
        hv = float(sp["wave"])
        wv = text_img(sp["wave"] + " م", font(F_XB, 92), wave_color(hv), shadow=False)
        paste(g, wv, R_ - wv.width + 14, gp + 104)
        pr = text_img("الدورة " + sp["period"] + " ث", font(F_B, 34), WHITE, shadow=False)
        paste(g, pr, R_ - pr.width + 14, gp + 214)
        dr = text_img("الانتفاخ: " + sp["dir"], font(F_M, 28), (200, 215, 240), shadow=False)
        paste(g, dr, R_ - dr.width + 14, gp + 262)
        wd = text_img("الرياح: " + sp["wind"], font(F_M, 28), (200, 215, 240), shadow=False)
        paste(g, wd, R_ - wd.width + 14, gp + 304)
        y = gp + 346
        extra = []
        if sp.get("wind_state"):
            extra.append(("حالة الريح: " + sp["wind_state"], (140, 230, 200)))
        if sp.get("water"):
            extra.append(("حرارة الماء " + sp["water"] + "°", (140, 210, 255)))
        if sp.get("tide"):
            extra.append((sp["tide"], (200, 205, 225)))
        for txt, col in extra[:3]:
            e = text_img(txt, font(F_M, 26), col, shadow=False)
            paste(g, e, R_ - e.width + 14, y)
            y += 36
        ar = arrow_img(sp.get("dir_deg", 315), 120, TEAL)
        paste(g, ar, gp + 30, gp + 120)
        lab = text_img("اتجاه الموج", font(F_M, 24), (190, 200, 220), shadow=False)
        paste(g, lab, gp + 90 - lab.width / 2, gp + 240)
        return g

    SURF_C = [surf_card(sp) for sp in data.get("surf", [])[:2]]

    def spot_chip(sp):
        t = text_img(sp["name"] + " • " + sp["spot"], font(F_B, 30), WHITE, shadow=False)
        g, gp = glass((t.width + 10, 62), r=31, border=(GOLD, TEAL), bw=2, a=170)
        paste(g, t, gp + 5, gp + (62 - t.height) / 2 + 2)
        return g

    SPOT_C = [(sp, spot_chip(sp)) for sp in data.get("surf", [])[:2]]

    # ---- outlook
    def wrap(txt, f, maxw):
        words, lines, cur = txt.split(), [], ""
        for wd in words:
            cand = (cur + " " + wd).strip()
            if tsize(cand, f)[0] <= maxw or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = wd
        if cur:
            lines.append(cur)
        return lines

    def outlook_panel():
        days = data.get("outlook", [])[:3]
        w, h = 1000, 440
        g, gp = glass((w, h), r=34, border=(PURPLE, TEAL), bw=3, a=170)
        t = text_img("الأيام المقبلة", font(F_XB, 46), WHITE, shadow=False)
        paste(g, t, gp + w - 30 - t.width + 14, gp + 6)
        colw = (w - 40) / max(1, len(days))
        icons = []
        for i, dd in enumerate(days):
            cx = gp + w - 20 - colw * (i + 0.5)       # RTL: first day on the right
            dn = text_img(dd["day"], font(F_XB, 40), TEAL, shadow=False)
            paste(g, dn, cx - dn.width / 2, gp + 70)
            icons.append((dd["cond"], cx - 60, gp + 126))
            y = gp + 252
            for ln in wrap(dd["text"], font(F_M, 26), colw - 40)[:3]:
                lt = text_img(ln, font(F_M, 26), (220, 228, 245), shadow=False)
                paste(g, lt, cx - lt.width / 2, y)
                y += 34
            if dd.get("safi_max") is not None:
                lab = text_img("آسفي", font(F_B, 28), GOLD, shadow=False)
                mx = text_img(f"{dd['safi_max']}°", font(F_XB, 34), temp_color(dd["safi_max"]), shadow=False)
                mn = text_img(f"{dd['safi_min']}°", font(F_B, 28), (180, 200, 230), shadow=False)
                tot = lab.width + mx.width + mn.width - 84 + 20
                x0 = cx + tot / 2
                paste(g, lab, x0 - lab.width + 14, gp + 372)
                x0 -= lab.width - 28 + 10
                paste(g, mx, x0 - mx.width + 14, gp + 368)
                x0 -= mx.width - 28 + 10
                paste(g, mn, x0 - mn.width + 14, gp + 374)
            if i > 0:
                ln_ = Image.new("RGBA", (2, h - 110), (140, 150, 200, 90))
                paste(g, ln_, cx + colw / 2, gp + 80)
        return g, icons

    OUT_P, OUT_ICONS = outlook_panel() if data.get("outlook") else (None, [])
    osys = data.get("outlook_system")

    def sys_chip():
        if not osys:
            return None
        t = text_img(osys["label"] + " • " + osys.get("when", ""), font(F_B, 30), WHITE, shadow=False)
        g, gp = glass((t.width + 10, 62), r=31, border=(PURPLE, CYAN), bw=2, a=175)
        paste(g, t, gp + 5, gp + (62 - t.height) / 2 + 2)
        return g

    SYS_C = sys_chip()

    def band_layer(t, u):
        f = ll2map(*osys["from"])
        to = ll2map(*osys["to"])
        cx = (f[0] + (to[0] - f[0]) * u) / 4
        cy = (f[1] + (to[1] - f[1]) * u) / 4
        dx, dy = to[0] - f[0], to[1] - f[1]
        L = math.hypot(dx, dy)
        nx, ny = dx / L, dy / L
        dn = (XQ - cx) * nx + (YQ - cy) * ny
        dp = (XQ - cx) * (-ny) + (YQ - cy) * nx
        n1 = np.roll(noise, (int(t * 4) % gh, int(t * 7) % gw), axis=(0, 1))
        a = np.exp(-dn ** 2 / (2 * 38 ** 2)) * np.exp(-dp ** 2 / (2 * 190 ** 2))
        a = a * np.clip(0.45 + 1.1 * (n1 - 0.4), 0, 1)
        core = np.exp(-dn ** 2 / (2 * 14 ** 2)) * np.exp(-dp ** 2 / (2 * 120 ** 2))
        arr = np.zeros((gh, gw, 4), np.uint8)
        col = np.array([235, 240, 250], np.float32) * (1 - 0.35 * core[..., None]) + \
            np.array([60, 120, 220], np.float32) * 0.35 * core[..., None]
        arr[..., :3] = np.clip(col, 0, 255).astype(np.uint8)
        arr[..., 3] = (np.clip(a * 1.15, 0, 1) * 225).astype(np.uint8)
        return arr, (cx * 4, cy * 4), (nx, ny)

    frames_idx = range(nfr) if preview is None else [int(p * FPS) for p in preview]
    proc = None if preview is not None else subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{FW}x{FH}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                             "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p",
                             "-movflags", "+faststart", out_path + ".video.mp4"], stdin=subprocess.PIPE)
    flashes = sorted(random.uniform(SC['south'][2] + 0.5, SC['south'][3]) for _ in range(8))
    south_keys = [k for k in cities if cities[k]["cond"] == "thunder"]

    for fi in frames_idx:
        t = fi / FPS
        if t < T_INTRO - 0.4:
            frame = intro_layer(t)
        elif t >= T_OUT0 + 0.4:
            frame = outro_layer(t - T_OUT0)
        else:
            cam = camera(t)
            frame = mr.render(cam[0], cam[1], cam[2], VY).convert("RGBA")
            # clouds (geographic, drifting)
            sh1 = int(t * 9) % gw
            sh2 = int(t * 5) % gh
            n1 = np.roll(noise, (sh2, sh1), axis=(0, 1))
            n2 = np.roll(noise[::-1, ::-1], (int(t * 3) % gh, int(t * 6) % gw), axis=(0, 1))
            dens = np.clip((0.55 * n1 + 0.45 * n2 - 0.38) * 2.6, 0, 1) * field
            ca = Image.fromarray((np.clip(dens, 0, 1) * 215).astype(np.uint8))
            s4 = cam[2] * 4
            a = 1 / s4
            ca = ca.transform((FW, FH), Image.AFFINE,
                              (a, 0, cam[0] / 4 - FW / 2 * a, 0, a, cam[1] / 4 - VY * a), Image.BILINEAR)
            ca = ca.filter(ImageFilter.GaussianBlur(3))
            cl = Image.new("RGBA", (FW, FH), (236, 242, 252, 0))
            cl.putalpha(ca)
            frame.alpha_composite(cl)
            # lightning flashes in the south
            for ft in flashes:
                if 0 <= t - ft < 0.18 and south_keys:
                    k = south_keys[int(ft * 100) % len(south_keys)]
                    px, py = to_screen(cam, CITIES[k][1] + 0.3, CITIES[k][2] + 0.4)
                    inten = 1 - (t - ft) / 0.18
                    fl = Image.new("RGBA", (500, 500), (0, 0, 0, 0))
                    ImageDraw.Draw(fl).ellipse((50, 50, 450, 450), fill=(235, 235, 255, int(150 * inten)))
                    paste(frame, fl.filter(ImageFilter.GaussianBlur(60)), px - 250, py - 250)
            cur = None
            for sc in scenes:
                if sc[2] - 0.05 <= t <= sc[3] + 0.6:
                    cur = sc
            # geographic layers that live across scene boundaries
            sea_on = 0.0
            for nm_, k_ in (("sea", 1.0), ("surf", 0.5)):
                a0 = SC[nm_][2] - 1.4
                a1 = SC[nm_][3] + 0.6
                if a0 <= t <= a1:
                    sea_on = max(sea_on, k_ * clamp01((t - a0) / 1.2) * clamp01((a1 - t) / 0.6))
            if sea_on > 0.01:
                frame.alpha_composite(to_screen_layer(wave_layer(t, sea_on), cam))
            if cur is not None:
                name = cur[0]
                tin = t - cur[2]
                tout = cur[3] + 0.6 - t
                fade_out = clamp01(tout / 0.45)
                if name == "temp":
                    la = clamp01(tin / 1.0) * fade_out
                    lay = TEMP_L.copy()
                    lay[..., 3] = (lay[..., 3] * la).astype(np.uint8)
                    frame.alpha_composite(to_screen_layer(lay, cam))
                    for i, k in enumerate(TEMP_LABEL_KEYS):
                        u = clamp01((tin - 0.8 - i * 0.12) / 0.5)
                        if u > 0:
                            px, py = to_screen(cam, CITIES[k][1], CITIES[k][2])
                            im = TEMP_LABELS[k]
                            paste(frame, im, px - 100, py - 40, u * fade_out)
                if name == "outlook" and osys:
                    u = ease(clamp01(tin / (cur[3] - cur[2])))
                    lay, (bxm, bym), (nx, ny) = band_layer(t, u)
                    la = clamp01(tin / 0.8) * fade_out
                    lay[..., 3] = (lay[..., 3] * la).astype(np.uint8)
                    frame.alpha_composite(to_screen_layer(lay, cam))
                    sx = (bxm - cam[0]) * cam[2] + FW / 2
                    sy = (bym - cam[1]) * cam[2] + VY
                    # motion arrow
                    ar = Image.new("RGBA", (FW, FH), (0, 0, 0, 0))
                    da = ImageDraw.Draw(ar)
                    ex, ey = sx + nx * 170, sy + ny * 170
                    da.line((sx + nx * 40, sy + ny * 40, ex, ey), fill=(255, 255, 255, int(220 * la)), width=8)
                    px_, py_ = -ny, nx
                    da.polygon([(ex + nx * 30, ey + ny * 30), (ex + px_ * 22, ey + py_ * 22), (ex - px_ * 22, ey - py_ * 22)],
                               fill=(255, 255, 255, int(230 * la)))
                    frame.alpha_composite(ar)
                    if SYS_C is not None:
                        cxs = min(max(sx - SYS_C.width / 2, 10), FW - SYS_C.width - 10)
                        cys = min(max(sy - 150, 330), 1250)
                        frost(frame, SYS_C, cxs, cys, la)
                if name == "sea":
                    for i, (z, zc) in enumerate(ZONE_C):
                        u = clamp01((tin - 0.6 - i * 0.35) / 0.6)
                        if u <= 0:
                            continue
                        px, py = to_screen(cam, z["lat"], z["lon"])
                        x0 = min(max(px - zc.width / 2, 0), FW - zc.width)
                        frost(frame, zc, x0, py - zc.height / 2, ease(u) * fade_out)
                if name == "surf":
                    for i, (sp, chp) in enumerate(SPOT_C):
                        u = clamp01((tin - 0.4 - i * 0.4) / 0.6)
                        if u <= 0:
                            continue
                        px, py = to_screen(cam, sp["lat"], sp["lon"])
                        pulse = (t * 1.0 + i * 0.5) % 1
                        pin = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
                        dp = ImageDraw.Draw(pin)
                        rr = 12 + 44 * pulse
                        dp.ellipse((60 - rr, 60 - rr, 60 + rr, 60 + rr), outline=GOLD + (int(230 * (1 - pulse)),), width=4)
                        dp.ellipse((48, 48, 72, 72), fill=GOLD + (255,))
                        dp.ellipse((54, 54, 66, 66), fill=(255, 255, 255, 255))
                        paste(frame, pin, px - 60, py - 60, u * fade_out)
                        frost(frame, chp, px + 30, py - chp.height / 2, u * fade_out)
                # chip
                ch = CHIPS[name]
                ca_ = clamp01(tin / 0.4) * fade_out
                frost(frame, ch, FW - 40 - ch.width + 30 + 60 * (1 - ease(clamp01(tin / 0.5))), 222, ca_)
                if name in SRC:
                    sr = SRC[name]
                    paste(frame, sr, FW - 40 - sr.width + 14, 322, ca_)
                for i, (k, (px0, py0), _bp, b) in enumerate(placements[name]):
                    px, py = to_screen(cam, CITIES[k][1], CITIES[k][2])
                    dx, dy = px - px0, py - py0
                    delay = 0.35 + i * 0.16
                    u = clamp01((tin - delay) / 0.75)
                    if u <= 0:
                        continue
                    sp_ = spring(u)
                    al = clamp01(u * 3) * fade_out
                    # pin
                    pulse = (t * 1.2 + i * 0.3) % 1
                    pin = Image.new("RGBA", (90, 90), (0, 0, 0, 0))
                    dp = ImageDraw.Draw(pin)
                    pc = GOLD if k == "safi" else TEAL
                    rr = 10 + 30 * pulse
                    dp.ellipse((45 - rr, 45 - rr, 45 + rr, 45 + rr), outline=pc + (int(220 * (1 - pulse)),), width=3)
                    dp.ellipse((35, 35, 55, 55), fill=pc + (255,))
                    dp.ellipse((40, 40, 50, 50), fill=(255, 255, 255, 255))
                    paste(frame, pin, px - 45, py - 45, al)
                    # badge (pre-laid out in mid camera; follow camera drift)
                    bi = b["img"]
                    sc_ = 0.6 + 0.4 * sp_
                    # find original top-left from layout recompute: stored as screen pos via side
                    bx, by = _badge_pos[name][k]
                    bx += dx
                    by += dy
                    cxb, cyb = bx + b["w"] / 2, by + b["h"] / 2
                    bimg = scaled(bi, sc_)
                    frost(frame, bimg, cxb - bimg.width / 2, cyb - bimg.height / 2, al, sc_, b["geo"])
                    fr = icon_frames(b["kind"], max(8, int(b["isz"] * sc_)))[(fi + i * 5) % NLOOP]
                    ix = cxb + (b["icon"][0] - b["pad"] - b["w"] / 2) * sc_
                    iy = cyb + (b["icon"][1] - b["pad"] - b["h"] / 2) * sc_
                    paste(frame, fr, ix, iy, al)
                # panels
                pu = ease(clamp01((tin - 0.9) / 0.6)) * fade_out
                if name == "full" and pu > 0:
                    yb = 1500 + 120 * (1 - pu)
                    yb = 1590 + 120 * (1 - pu)
                    frost(frame, CARD_HOT, FW / 2 + 8 - 30, yb, pu)
                    frost(frame, CARD_COLD, FW / 2 - 8 - CARD_COLD.width + 30, yb, pu)
                if name == "center" and SAFI_P is not None and pu > 0:
                    yb = 1420 + 140 * (1 - pu)
                    frost(frame, SAFI_P, FW / 2 - SAFI_P.width / 2, yb, pu)
                    fr = icon_frames(cities["safi"]["cond"], 160)[fi % NLOOP]
                    paste(frame, fr, FW / 2 - SAFI_P.width / 2 + SAFI_ICON_POS[0],
                          yb + SAFI_ICON_POS[1], pu)
                if name == "south" and SOUTH_P is not None and pu > 0:
                    yb = 1560 + 140 * (1 - pu)
                    frost(frame, SOUTH_P, FW / 2 - SOUTH_P.width / 2, yb, pu)
                if name == "temp" and pu > 0:
                    frost(frame, TEMP_LEG, FW / 2 - TEMP_LEG.width / 2, 1560 + 120 * (1 - pu), pu)
                if name == "sea" and pu > 0:
                    yb = 1545 + 120 * (1 - pu)
                    frost(frame, SEA_LEG, FW / 2 - SEA_LEG.width / 2, yb, pu)
                    paste(frame, SEA_NOTE, FW / 2 - SEA_NOTE.width / 2, yb - 40, pu)
                if name == "surf" and pu > 0 and SURF_C:
                    yb = 1255 + 140 * (1 - pu)
                    for i, c_ in enumerate(SURF_C):
                        x = FW - 20 - c_.width + 30 - i * (c_.width - 60 + 20) if len(SURF_C) > 1 else FW / 2 - c_.width / 2
                        frost(frame, c_, x, yb, pu)
                if name == "outlook" and pu > 0 and OUT_P is not None:
                    yb = 1330 + 140 * (1 - pu)
                    frost(frame, OUT_P, FW / 2 - OUT_P.width / 2, yb, pu)
                    for i, (cond, ix, iy) in enumerate(OUT_ICONS):
                        fr = icon_frames(cond, 120)[(fi + i * 7) % NLOOP]
                        paste(frame, fr, FW / 2 - OUT_P.width / 2 + ix, yb + iy, pu)
                if name == "final" and pu > 0:
                    yb = 1770 - ALERTS_P.height + 30 + 140 * (1 - pu)
                    frost(frame, ALERTS_P, FW / 2 - ALERTS_P.width / 2, yb, pu)
            # header + ticker
            frame.alpha_composite(HEADER, (0, 0))
            frame.alpha_composite(TBAND, (0, TK_Y - 120))
            strip = Image.new("RGBA", (FW, TK_H), (0, 0, 0, 0))
            tx = -tick_txt.width + (t - T_INTRO + 2) * 105
            tx = (tx % tick_txt.width) - tick_txt.width
            for k in range(3):
                paste(strip, tick_txt, tx + k * tick_txt.width, (TK_H - tick_txt.height) // 2 + 2)
            strip = strip.crop((0, 0, FW - 236, TK_H))
            fa = np.full((TK_H, FW - 236), 255, np.uint8)
            ramp = np.linspace(0, 255, 60).astype(np.uint8)
            fa[:, :60] = np.minimum(fa[:, :60], ramp)
            fa[:, -60:] = np.minimum(fa[:, -60:], ramp[::-1])
            strip.putalpha(Image.fromarray(np.minimum(np.array(strip.getchannel("A")), fa)))
            frame.alpha_composite(strip, (0, TK_Y))
            frame.alpha_composite(tag, (FW - 230, TK_Y + 2))
            # crossfades with intro/outro
            if t < T_INTRO:
                frame = Image.blend(intro_layer(t), frame, clamp01((t - (T_INTRO - 0.4)) / 0.4))
            if t > T_OUT0:
                frame = Image.blend(frame, outro_layer(t - T_OUT0), clamp01((t - T_OUT0) / 0.4))
        # global fade in/out
        g = clamp01(t / 0.3) * clamp01((DUR - t) / 0.5)
        rgb = frame.convert("RGB")
        if g < 1:
            rgb = Image.blend(Image.new("RGB", (FW, FH)), rgb, g)
        if proc is None:
            rgb.save(f"{out_path}_{t:05.2f}.jpg", quality=88)
            continue
        proc.stdin.write(rgb.tobytes())
        if fi % 150 == 0:
            print(f"frame {fi}/{nfr}", flush=True)
    if proc is None:
        return
    proc.stdin.close()
    proc.wait()
    flights = [b[2] - 1.9 for a, b in zip(scenes, scenes[1:]) if a[1] != b[1]]
    make_audio(out_path + ".wav", DUR, flights, [T_INTRO, T_OUT0])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out_path + ".video.mp4", "-i", out_path + ".wav",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", out_path], check=True)
    os.remove(out_path + ".video.mp4")
    os.remove(out_path + ".wav")


_badge_pos = {}


def make_audio(path, dur, whooshes, chimes):
    from scipy.io import wavfile
    from scipy.signal import butter, lfilter
    sr = 44100
    n = int(dur * sr)
    t = np.arange(n) / sr
    out = np.zeros(n)
    chords = [(57, 60, 64), (53, 57, 60), (48, 55, 64), (55, 59, 62)]   # Am F C G
    seg = 4.0
    for i in range(int(dur / seg) + 1):
        notes = chords[i % 4]
        s0 = int(i * seg * sr)
        s1 = min(n, int((i + 1) * seg * sr + sr))
        tt = t[s0:s1] - i * seg
        env = np.minimum(1, tt / 0.8) * np.exp(-np.maximum(0, tt - seg) * 3)
        for m in notes:
            f = 440 * 2 ** ((m - 69) / 12)
            for det in (-0.12, 0.12):
                out[s0:s1] += 0.045 * env * np.sin(2 * np.pi * f * (1 + det / 100) * tt)
            out[s0:s1] += 0.02 * env * np.sin(2 * np.pi * f * 2 * tt)
    b, a = butter(2, 2200 / (sr / 2))
    out = lfilter(b, a, out)
    # beat
    bpm = 96
    beat = 60 / bpm
    for k in range(int(dur / beat)):
        tb = k * beat
        if tb < 3.2 or tb > dur - 2:
            continue
        s0 = int(tb * sr)
        L = int(0.35 * sr)
        tt = np.arange(min(L, n - s0)) / sr
        f = 45 + 80 * np.exp(-tt * 30)
        out[s0:s0 + len(tt)] += 0.32 * np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-tt * 9)
        # hat on off-beat
        s1 = int((tb + beat / 2) * sr)
        Lh = int(0.05 * sr)
        if s1 + Lh < n:
            hn = np.random.default_rng(k).standard_normal(Lh) * np.exp(-np.arange(Lh) / sr * 70)
            out[s1:s1 + Lh] += 0.05 * np.diff(np.concatenate([[0], hn]))
    # whooshes
    bh, ah = butter(2, [400 / (sr / 2), 3000 / (sr / 2)], btype="band")
    for w0 in whooshes:
        s0 = int(max(0, w0) * sr)
        L = int(1.6 * sr)
        if s0 + L > n:
            continue
        nz = lfilter(bh, ah, np.random.default_rng(int(w0 * 10)).standard_normal(L))
        env = np.sin(np.linspace(0, np.pi, L)) ** 2
        out[s0:s0 + L] += 0.25 * nz * env
    for c0 in chimes:
        s0 = int(max(0, c0 - 0.3) * sr)
        L = int(2.5 * sr)
        if s0 + L > n:
            continue
        tt = np.arange(L) / sr
        for f in (880, 1318.5, 1760):
            out[s0:s0 + L] += 0.06 * np.sin(2 * np.pi * f * tt) * np.exp(-tt * 2.2)
    # intro chime at start
    L = int(2.5 * sr)
    tt = np.arange(L) / sr
    for f in (659.3, 987.8, 1318.5):
        out[int(0.2 * sr):int(0.2 * sr) + L] += 0.07 * np.sin(2 * np.pi * f * tt) * np.exp(-tt * 1.8)
    fade = np.minimum(1, t / 0.5) * np.minimum(1, (dur - t) / 1.5)
    out *= fade
    out /= np.max(np.abs(out)) + 1e-9
    out *= 0.7
    st = np.stack([out, np.roll(out, 300)], axis=1)
    wavfile.write(path, sr, (st * 32767).astype(np.int16))


if __name__ == "__main__":
    data = json.load(open(sys.argv[1]))
    out = sys.argv[2] if len(sys.argv) > 2 else "weather.mp4"
    pv = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None
    build(data, out, pv)
    print("OK", out)
