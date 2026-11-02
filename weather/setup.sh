#!/usr/bin/env bash
# One-time setup in a fresh container: deps + raw map data + base map.
set -e
cd "$(dirname "$0")"
pip install -q --break-system-packages shapely scipy numpy pillow 2>/dev/null || true
mkdir -p data /tmp/bm
if [ ! -f data/shadedrelief.jpg ]; then
  pip download basemap-data==2.0.0 --no-deps -d /tmp/bm -q
  (cd /tmp/bm && unzip -o -q basemap_data-2.0.0-py3-none-any.whl 'mpl_toolkits/basemap_data/*.jpg')
  cp /tmp/bm/mpl_toolkits/basemap_data/shadedrelief.jpg /tmp/bm/mpl_toolkits/basemap_data/etopo1.jpg data/
fi
if [ ! -f data/countries.geojson ]; then
  curl -sS -L -o data/countries.geojson \
    https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson
fi
[ -f data/basemap.png ] || python3 build_basemap.py
echo "setup ok"
