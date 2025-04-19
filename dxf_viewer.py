#!/usr/bin/env python3
# dxf_viewer.py – Generate a standalone HTML viewer for DXF files

import os
import sys
import argparse
import json
import ezdxf
from typing import Any, Dict, List, Tuple

def load_dxf(file_path: str) -> ezdxf.document.Drawing:
    try:
        doc = ezdxf.readfile(file_path)
        print(f"Loaded DXF: {file_path}")
        return doc
    except IOError:
        print(f"I/O error or not a DXF: {file_path}")
        sys.exit(1)
    except ezdxf.DXFError:
        print(f"Invalid/corrupted DXF: {file_path}")
        sys.exit(2)

def ezdxf_color_to_rgb(idx: int) -> str:
    cmap = {
        1: "#FF0000", 2: "#FFFF00", 3: "#00FF00",
        4: "#00FFFF", 5: "#0000FF", 6: "#FF00FF",
        7: "#FFFFFF", 8: "#808080", 9: "#C0C0C0",
    }
    return cmap.get(idx, "#000000")

def extract_entities(doc: ezdxf.document.Drawing) -> List[Dict[str, Any]]:
    msp = doc.modelspace()
    layers = {L.dxf.name: L for L in doc.layers}
    out: List[Dict[str, Any]] = []
    for e in msp:
        layer = e.dxf.layer
        col = ezdxf_color_to_rgb(layers.get(layer, layers.get('0')).dxf.color)
        base = {'layer': layer, 'color': col, 'linetype': getattr(e.dxf, 'linetype', 'CONTINUOUS')}
        t = e.dxftype()
        if t == 'LINE':
            out.append({**base, 'type':'line',
                        'start':[e.dxf.start.x, e.dxf.start.y],
                        'end':  [e.dxf.end.x, e.dxf.end.y]})
        elif t == 'LWPOLYLINE':
            pts = [[x,y] for x,y, *_ in e.get_points()]
            out.append({**base, 'type':'polyline','points':pts,'closed':e.is_closed})
        elif t == 'CIRCLE':
            out.append({**base,'type':'circle',
                        'center':[e.dxf.center.x,e.dxf.center.y],
                        'radius': e.dxf.radius})
        elif t == 'ARC':
            out.append({**base,'type':'arc',
                        'center':[e.dxf.center.x,e.dxf.center.y],
                        'radius':e.dxf.radius,
                        'startAngle': e.dxf.start_angle,
                        'endAngle':   e.dxf.end_angle})
        elif t == 'TEXT':
            out.append({**base,'type':'text',
                        'text':e.dxf.text,
                        'position':[e.dxf.insert.x,e.dxf.insert.y],
                        'height': e.dxf.height})
        elif t == 'DIMENSION':
            # Skip dimensions without required points
            if not hasattr(e.dxf, 'defpoint') or not hasattr(e.dxf, 'text_midpoint'):
                continue
            if e.dxf.defpoint is None or e.dxf.text_midpoint is None:
                continue
            out.append({**base,'type':'dimension',
                        'defPoint':[e.dxf.defpoint.x,e.dxf.defpoint.y],
                        'textMidpoint':[e.dxf.text_midpoint.x,e.dxf.text_midpoint.y],
                        'dimText': getattr(e.dxf, 'text', '')})
    return out

def get_drawing_extents(ents: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    if not ents:
        return [0,0],[100,100]
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    for e in ents:
        ty = e['type']
        if ty == 'line':
            xs = [e['start'][0], e['end'][0]]; ys = [e['start'][1], e['end'][1]]
        elif ty == 'polyline':
            xs, ys = zip(*e['points'])
        elif ty in ('circle','arc'):
            cx,cy,r = e['center'][0], e['center'][1], e['radius']
            xs, ys = [cx-r, cx+r], [cy-r, cy+r]
        elif ty == 'text':
            x,y,h = e['position'][0], e['position'][1], e['height']
            xs, ys = [x, x+len(e['text'])*h], [y, y+h]
        elif ty == 'dimension':
            xs = [e['defPoint'][0], e['textMidpoint'][0]]
            ys = [e['defPoint'][1], e['textMidpoint'][1]]
        else:
            continue
        min_x, max_x = min(min_x, *xs), max(max_x, *xs)
        min_y, max_y = min(min_y, *ys), max(max_y, *ys)
    m = max(max_x-min_x, max_y-min_y) * 0.05
    return [min_x-m, min_y-m], [max_x+m, max_y+m]

def generate_html(entities: List[Dict[str, Any]],
                  extents:   Tuple[List[float], List[float]],
                  out_path:  str) -> None:
    ent_json = json.dumps(entities)
    ext_json = json.dumps(extents)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>DXF Viewer</title>
    <style>
  body,html{{margin:0;padding:0;overflow:hidden;width:100%;height:100%;background:#1e1e1e;}}
  canvas{{display:block;cursor:grab;}}
  #controls{{position:absolute;top:10px;left:10px;z-index:1;
    background:rgba(255,255,255,0.8);padding:5px;border-radius:4px;}}
  #layer-control{{position:absolute;top:10px;right:10px;z-index:1;
    max-height:80vh;overflow:auto;background:rgba(255,255,255,0.8);
    padding:5px;border-radius:4px;font-size:12px;}}
  .layer-item{{display:flex;align-items:center;margin:2px 0;gap:4px;}}
  .swatch{{width:12px;height:12px;border:1px solid #000;}}
    </style>
</head><body>
        <canvas id="drawing-canvas"></canvas>
        <div id="controls">
            <button id="zoom-in">+</button>
    <button id="zoom-out">–</button>
            <button id="fit-view">Fit</button>
        </div>
  <div id="layer-control"><strong>Layers</strong></div>

    <script>
  // 1) State
        let scale = 1;
        let offsetX = 0;
        let offsetY = 0;
        let dragging = false;
        let lastX = 0;
        let lastY = 0;
        let layerVisibility = {{}};
  let needsRedraw = true;

  // 2) Canvas
  const canvas = document.getElementById('drawing-canvas');
  const ctx = canvas.getContext('2d');
  
  function resizeCanvas() {{
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    needsRedraw = true;
  }}
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  // 3) Data
  const entities = {ent_json};
  const extents = {ext_json};

  // 4) Init view
  function initView() {{
    const [min, max] = extents;
    const width = Math.abs(max[0] - min[0]);
    const height = Math.abs(max[1] - min[1]);
    
    // Calculate scale to fit the drawing in the viewport
    const scaleX = (canvas.width * 0.8) / width;
    const scaleY = (canvas.height * 0.8) / height;
    scale = Math.min(scaleX, scaleY);
    
    // Center the drawing
    offsetX = canvas.width/2;
    offsetY = canvas.height/2;
    
    setupLayers();
    needsRedraw = true;
  }}

  function setupLayers() {{
    const lc = document.getElementById('layer-control');
    const layers = new Set(entities.map(e => e.layer));
    lc.innerHTML = '<strong>Layers</strong>';
    layers.forEach(layer => {{
      layerVisibility[layer] = true;
      const div = document.createElement('div');
      div.className = 'layer-item';
      const sw = document.createElement('div');
      sw.className = 'swatch';
      sw.style.backgroundColor = entities.find(e => e.layer === layer).color;
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = true;
      cb.onchange = () => {{
        layerVisibility[layer] = cb.checked;
        needsRedraw = true;
      }};
      const lbl = document.createElement('span');
      lbl.textContent = layer;
      div.append(sw, cb, lbl);
      lc.append(div);
    }});
  }}

  // 5) Draw
        function draw() {{
    if (!needsRedraw) return;
    
            // Clear canvas
    ctx.fillStyle = '#1e1e1e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Set up coordinate system
    const [min, max] = extents;
    const centerX = (min[0] + max[0]) / 2;
    const centerY = (min[1] + max[1]) / 2;
    
            ctx.save();
            ctx.translate(offsetX, offsetY);
    ctx.scale(scale, -scale);  // Flip Y axis
    ctx.translate(-centerX, -centerY);  // Center the drawing
            
            // Draw entities
    entities.forEach(e => {{
      if (!layerVisibility[e.layer]) return;
      
      ctx.strokeStyle = e.color;
      ctx.fillStyle = e.color;
      ctx.lineWidth = 1/scale;
      ctx.setLineDash(e.linetype === 'DASHED' ? [5/scale] : []);
      
      switch(e.type) {{
        case 'line':
          ctx.beginPath();
          ctx.moveTo(e.start[0], e.start[1]);
          ctx.lineTo(e.end[0], e.end[1]);
          ctx.stroke();
          break;
        case 'polyline':
                    ctx.beginPath();
          ctx.moveTo(...e.points[0]);
          e.points.slice(1).forEach(p => ctx.lineTo(...p));
          if (e.closed) ctx.closePath();
                    ctx.stroke();
          break;
        case 'circle':
                    ctx.beginPath();
          ctx.arc(e.center[0], e.center[1], e.radius, 0, 2*Math.PI);
                    ctx.stroke();
          break;
        case 'arc':
                    ctx.beginPath();
          ctx.arc(e.center[0], e.center[1], e.radius,
              e.startAngle*Math.PI/180, e.endAngle*Math.PI/180);
                    ctx.stroke();
          break;
        case 'text':
                    ctx.save();
          ctx.scale(1, -1);  // Flip text right side up
          ctx.font = `${{e.height}}px Arial`;
          ctx.fillText(e.text, e.position[0], -e.position[1]);
                    ctx.restore();
          break;
        case 'dimension':
                    ctx.beginPath();
          ctx.moveTo(e.defPoint[0], e.defPoint[1]);
          ctx.lineTo(e.textMidpoint[0], e.textMidpoint[1]);
                    ctx.stroke();
                    
                    ctx.save();
          ctx.scale(1, -1);
          ctx.font = `${{2/scale}}px Arial`;
          ctx.fillText(e.dimText, e.textMidpoint[0], -e.textMidpoint[1]);
                    ctx.restore();
          break;
                }}
            }});
            
            ctx.restore();
    needsRedraw = false;
  }}

  // 6) Animate & pan/zoom
  function animate() {{
    draw();
    requestAnimationFrame(animate);
  }}

        canvas.addEventListener('mousedown', e => {{
            dragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
            canvas.style.cursor = 'grabbing';
        }});
        
        canvas.addEventListener('mousemove', e => {{
            if (dragging) {{
      offsetX += e.clientX - lastX;
      offsetY += e.clientY - lastY;
                lastX = e.clientX;
                lastY = e.clientY;
      needsRedraw = true;
            }}
        }});
        
  ['mouseup', 'mouseleave'].forEach(evt => 
    canvas.addEventListener(evt, () => {{
            dragging = false;
            canvas.style.cursor = 'grab';
    }})
  );
        
        canvas.addEventListener('wheel', e => {{
            e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left);
    const my = (e.clientY - rect.top);
    const worldX = (mx - offsetX)/scale;
    const worldY = (offsetY - my)/scale;
            const factor = e.deltaY < 0 ? 1.1 : 0.9;
            scale *= factor;
    offsetX = mx - worldX*scale;
    offsetY = my + worldY*scale;
    needsRedraw = true;
  }});

  // 7) Buttons
  document.getElementById('zoom-in').onclick = () => {{
            scale *= 1.2;
    needsRedraw = true;
  }};
  document.getElementById('zoom-out').onclick = () => {{
            scale *= 0.8;
    needsRedraw = true;
  }};
  document.getElementById('fit-view').onclick = initView;
            
  // 8) Start
            initView();
  animate();
    </script>
</body></html>"""

    with open(out_path, 'w') as f:
        f.write(html)
    print(f"HTML viewer written to: {out_path}")

def main():
    p = argparse.ArgumentParser(description="DXF → standalone HTML viewer")
    p.add_argument('input', help="source .dxf file")
    p.add_argument('-o','--output', help="output .html", default=None)
    args = p.parse_args()

    src = args.input
    dst = args.output or os.path.splitext(src)[0] + "_viewer.html"

    doc = load_dxf(src)
    ents = extract_entities(doc)
    ext = get_drawing_extents(ents)
    generate_html(ents, ext, dst)

if __name__ == "__main__":
    main()
