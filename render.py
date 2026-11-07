#!/usr/bin/env python3
"""Carte ڤايب 1080x1350.
Usage: python3 render.py --titre "..." --ligne "..." [--ligne "..."] --photo photo.jpg --out carte.jpg
       [--photo-top 520] [--photo-pos "50% 40%"] [--titre-size 108]
"""
import argparse, html, os, shutil, subprocess, tempfile
from PIL import Image
here = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument('--titre', required=True); p.add_argument('--ligne', action='append', default=[])
p.add_argument('--photo', required=True); p.add_argument('--out', required=True)
p.add_argument('--photo-top', type=int, default=520); p.add_argument('--photo-pos', default='50% 40%')
p.add_argument('--titre-size', type=int, default=108)
a = p.parse_args()
tmp = tempfile.mkdtemp()
for f in ('SalmaArabic-Black.otf', 'logo-vibe.jpg', 'grain.png'):
    shutil.copy(os.path.join(here, f), tmp)
ext = os.path.splitext(a.photo)[1] or '.jpg'
shutil.copy(a.photo, os.path.join(tmp, 'photo' + ext))
bars = ''.join(f'<span class="bar">{html.escape(l)}</span>' for l in a.ligne)
t = open(os.path.join(here, 'template.html'), encoding='utf-8').read()
t = (t.replace('{{TITRE}}', html.escape(a.titre)).replace('{{BARS}}', bars)
      .replace('{{PHOTO}}', 'photo' + ext).replace('{{PHOTO_TOP}}', str(a.photo_top))
      .replace('{{PHOTO_POS}}', a.photo_pos).replace('{{TITLE_SIZE}}', str(a.titre_size)))
open(os.path.join(tmp, 'card.html'), 'w', encoding='utf-8').write(t)
chrome = shutil.which('google-chrome') or shutil.which('chromium') or shutil.which('chromium-browser')
png = os.path.join(tmp, 'out.png')
subprocess.run([chrome, '--headless=new', '--no-sandbox', '--disable-gpu', '--hide-scrollbars',
                '--window-size=1080,1350', f'--screenshot={png}', 'file://' + os.path.join(tmp, 'card.html')],
               check=True, capture_output=True)
Image.open(png).convert('RGB').save(a.out, quality=92)
print(a.out)
