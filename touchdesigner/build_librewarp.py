# =============================================================================
#  LibreWarp - constructor de red para TouchDesigner (2023.x / 2025.x)
#
#  Uso:
#   1. En TouchDesigner crea un Text DAT y pega TODO este archivo.
#   2. Clic derecho en el DAT > "Run Script"  (o Ctrl+R con el DAT seleccionado).
#   3. Conecta tu contenido al input del COMP 'librewarp'.
#   4. En el COMP, pulsa "Abrir ventanas".
#
#  Crea un Base COMP 'librewarp' con:
#   - un In TOP como entrada de contenido
#   - un GLSL TOP por proyector: warp (mapa UV) + blend + nivel de negro
#   - un canvas GLSL del tamaño del display, con cada proyector en su rect
#   - una Window COMP que cubre el display (p.ej. QuadHead2Go 5760x1080)
#   - parámetros en vivo: Gamma, Curva de blend, Adelgazar blend, Nivel de negro, Feather, Modo...
#
#  Para reutilizarlo: clic derecho en el COMP > Save Component .tox
# =============================================================================
import json
import os

# Carpeta de calibración. Se prueban en orden y se usa la primera que tenga
# calibration.json + todos los mapas/negros que ese JSON referencia.
CALIB_DIR_CANDIDATES = [
    r'C:\Users\CELARG\Desktop\calibrator\output',
    r'C:\Users\CELARG\Desktop\calibrator\sim_output',
    r'../calibrator/output',
]
COMP_NAME = 'librewarp'
WINDOW_MODE = 'canvas'      # 'canvas' (1 ventana por display) o 'separate' (1 por proyector)

PIXEL_SHADER = r'''
// LibreWarp - warp & blend
// input 0: mapa UV (R=u, G=v, B=peso de blend)   float
// input 1: contenido
// input 2: negro/uniformidad (R=negro, G=uniformidad)
uniform vec4 uParams;   // x=gamma  y=curva  z=negro  w=feather
uniform vec4 uMode;     // x=modo  y=divisiones rejilla  z=dither  w=uniformidad
uniform vec4 uBlend;    // x=adelgazar la rampa (1 = como el mapa, 3 = 3x más fina)
out vec4 fragColor;

float blendCurve(float w, float p, float thin){
    // thin>1 comprime la rampa hacia el centro de la costura (w=0.5). Como
    // f(w) + f(1-w) == 1, la suma de los dos proyectores sigue dando 1: la
    // costura se ve más fina SIN banda brillante. No usar donde se solapen
    // tres proyectores a la vez (ahí la suma no se conserva).
    if (thin > 1.0) w = clamp(0.5 + (w - 0.5) * thin, 0.0, 1.0);
    if (abs(p - 1.0) < 1e-4) return w;
    float a = pow(w, p); float b = pow(1.0 - w, p);
    return a / max(a + b, 1e-6);
}
float hash12(vec2 p){
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
vec3 gridPattern(vec2 uv, float n){
    float aspect = uTD2DInfos[1].res.w / uTD2DInfos[1].res.z;
    vec2 cells = vec2(n, max(1.0, floor(n * aspect + 0.5)));
    vec2 g = uv * cells;
    vec2 f = abs(fract(g + 0.5) - 0.5);
    vec2 px = max(fwidth(g), vec2(1e-5));
    float lx = 1.0 - smoothstep(0.0, 1.5 * px.x, f.x);
    float ly = 1.0 - smoothstep(0.0, 1.5 * px.y, f.y);
    vec3 base = vec3(uv.x, uv.y, 1.0 - uv.x) * 0.55;
    vec2 d = (uv - 0.5) * vec2(1.0, aspect);
    float r = abs(length(d) - 0.25 * aspect);
    float circ = 1.0 - smoothstep(0.0, 2.0 * length(fwidth(d)), r);
    float crs = max(1.0 - smoothstep(0.0, 2.0 * px.x / cells.x, abs(uv.x - 0.5)),
                    1.0 - smoothstep(0.0, 2.0 * px.y / cells.y, abs(uv.y - 0.5)));
    return mix(base, vec3(1.0), max(max(lx, ly), max(circ, crs)));
}
void main(){
    vec4 m = texelFetch(sTD2DInputs[0], ivec2(gl_FragCoord.xy), 0);
    vec2 uv = m.rg;
    if (uv.x < 0.0 || uv.y < 0.0) { fragColor = TDOutputSwizzle(vec4(0,0,0,1)); return; }
    vec4 bk = texelFetch(sTD2DInputs[2], ivec2(gl_FragCoord.xy), 0);  // R=negro G=uniformidad
    float gamma = max(uParams.x, 0.01);
    float w = blendCurve(clamp(m.b, 0.0, 1.0), max(uParams.y, 0.01), max(uBlend.x, 1.0));
    w *= mix(1.0, bk.g, clamp(uMode.w, 0.0, 1.0));
    if (uParams.w > 0.0) {
        float e = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
        w *= smoothstep(0.0, uParams.w, e);
    }
    int mode = int(uMode.x + 0.5);
    vec3 c;
    if (mode == 0)      c = texture(sTD2DInputs[1], uv).rgb;
    else if (mode == 1) c = gridPattern(uv, max(uMode.y, 1.0));
    else if (mode == 2) { fragColor = TDOutputSwizzle(vec4(vec3(w), 1.0)); return; }
    else if (mode == 3) c = vec3(1.0);
    else                c = vec3(0.0);
    vec3 lin = pow(max(c, vec3(0.0)), vec3(gamma)) * w;
    float lb = clamp(bk.r * uParams.z, 0.0, 1.0);
    lin = vec3(lb) + lin * (1.0 - lb);
    vec3 o = pow(clamp(lin, 0.0, 1.0), vec3(1.0 / gamma));
    if (uMode.z > 0.5) o += (hash12(gl_FragCoord.xy) - 0.5) / 255.0;
    fragColor = TDOutputSwizzle(vec4(o, 1.0));
}
'''

PAREXEC = r'''
def onPulse(par):
    comp = par.owner
    if par.name == 'Reload':
        for o in comp.findChildren(type=moviefileinTOP, depth=1):
            o.par.reloadpulse.pulse()
    elif par.name == 'Openwindows':
        for w in comp.findChildren(type=windowCOMP, depth=1):
            w.par.winopen.pulse()
    elif par.name == 'Closewindows':
        for w in comp.findChildren(type=windowCOMP, depth=1):
            w.par.winclose.pulse()
    return

def onValueChange(par, prev):
    return
'''

warnings = []


def _set(o, name, value, expr=False):
    """Asigna un parámetro si existe (los nombres cambian entre versiones de TD)."""
    p = getattr(o.par, name, None)
    if p is None:
        warnings.append('%s: no existe el parámetro "%s"' % (o.path, name))
        return False
    try:
        if expr:
            p.expr = value
        else:
            p.val = value
        return True
    except Exception as e:
        warnings.append('%s.%s: %s' % (o.path, name, e))
        return False


def _set_choice(o, name, exact, fuzzy):
    """Elige una opción de menú probando tokens exactos y después por texto.

    Los nombres de las opciones de pixel format cambian entre versiones de TD
    ('rgba32f' vs 'rgba32float'), así que se valida contra el menú real.
    """
    p = getattr(o.par, name, None)
    if p is None:
        warnings.append('%s: no existe el parámetro "%s"' % (o.path, name))
        return False
    try:
        names = list(getattr(p, 'menuNames', None) or [])
        labels = list(getattr(p, 'menuLabels', None) or [])
        for cand in exact:
            if cand in names:
                p.val = cand
                return True
        for i, n in enumerate(names):
            hay = (n + ' ' + (labels[i] if i < len(labels) else '')).lower()
            if all(k in hay for k in fuzzy):
                p.val = n
                return True
        warnings.append('%s.%s: ninguna opción coincide con %r (hay: %s)'
                        % (o.path, name, fuzzy, names))
    except Exception as e:
        warnings.append('%s.%s: %s' % (o.path, name, e))
    return False


def _uniform(g, idx, name, exprs):
    """Uniform vec4 en la página Vectors (TD 2023+: 'vec'; antes: uniname0/value0x)."""
    if getattr(g.par, 'vec0name', None) is not None:
        try:
            if g.seq.vec.numBlocks < idx + 1:
                g.seq.vec.numBlocks = idx + 1
        except Exception as e:
            warnings.append('secuencia vec: %s' % e)
        _set(g, 'vec%dname' % idx, name)
        for c, e in zip('xyzw', exprs):
            _set(g, 'vec%dvalue%s' % (idx, c), e, expr=True)
    else:
        _set(g, 'uniname%d' % idx, name)
        for c, e in zip('xyzw', exprs):
            _set(g, 'value%d%s' % (idx, c), e, expr=True)


def _connect(src, dst, index):
    src.outputConnectors[0].connect(dst.inputConnectors[index])


def _load_calibration():
    """Devuelve (carpeta, metadatos) de la primera calibración utilizable."""
    checked = []
    for d in CALIB_DIR_CANDIDATES:
        path = os.path.abspath(d)
        meta_path = os.path.join(path, 'calibration.json')
        if not os.path.isfile(meta_path):
            checked.append('%s (sin calibration.json)' % path)
            continue
        try:
            with open(meta_path, encoding='utf-8') as f:
                meta = json.load(f)
        except Exception as e:
            checked.append('%s (JSON inválido: %s)' % (path, e))
            continue
        projs = meta.get('projectors') or []
        if not projs:
            checked.append('%s (sin proyectores)' % path)
            continue
        missing = []
        for p in projs:
            for key in ('file', 'black_file'):
                fn = p.get(key)
                if fn and not os.path.isfile(os.path.join(path, fn)):
                    missing.append(fn)
        if missing:
            checked.append('%s (faltan mapas: %s)' % (path, ', '.join(missing)))
            continue
        print('LibreWarp: calibración %r con %d proyectores' % (path, len(projs)))
        return path, meta
    raise RuntimeError(
        'LibreWarp: no encontré una calibración utilizable. Revisé:\n  - '
        + '\n  - '.join(checked)
        + '\nCambiá CALIB_DIR_CANDIDATES arriba del script.'
    )


def _window_warnings(meta):
    """Avisos si las ventanas de contenido no reparten el lienzo en partes parecidas.

    Cuando la calibración normaliza mal u/v, cada proyector acaba viendo sólo una
    franja del lienzo: en TouchDesigner se traduce en "todo un color" y en una
    rejilla de prueba sin rejilla. Mejor saberlo antes de proyectar.
    """
    out = []
    wins = meta.get('content_windows') or []
    if len(wins) < 2:
        return out
    widths = [float(w[1]) - float(w[0]) for w in wins]
    mx = max(widths) or 1.0
    if min(widths) < 0.6 * mx:
        detalle = ', '.join('P%d %.0f%%' % (i, wd * 100) for i, wd in enumerate(widths))
        out.append('las ventanas de contenido no son parecidas (%s): cada proyector '
                   'verá sólo una franja del lienzo. Vuelve a generar los mapas con '
                   'calibrate.py (paso Cálculo), no los uses así.' % detalle)
    return out


def _canvas_shader(rects, H):
    """Compone los proyectores (ya warpeados) en su rect de escritorio."""
    lines = ['out vec4 fragColor;', 'void main(){',
             '    ivec2 c = ivec2(gl_FragCoord.xy);',
             '    int yd = %d - 1 - c.y;  // y hacia abajo (coordenadas de escritorio)' % H]
    for i, (x, y, w, h) in enumerate(rects):
        lines += ['    { ivec2 l = ivec2(c.x - %d, yd - %d);' % (x, y),
                  '      if (l.x >= 0 && l.y >= 0 && l.x < %d && l.y < %d) {' % (w, h),
                  '        fragColor = TDOutputSwizzle(texelFetch(sTD2DInputs[%d], ivec2(l.x, %d - 1 - l.y), 0));' % (i, h),
                  '        return; } }']
    lines += ['    fragColor = TDOutputSwizzle(vec4(0.0, 0.0, 0.0, 1.0));', '}']
    return '\n'.join(lines)


def _custom_pars(comp, calib_dir):
    page = comp.appendCustomPage('LibreWarp')
    # Str (no Folder): el valor se usa en las expresiones de los Movie File In
    p = page.appendStr('Calibdir', label='Carpeta de calibración')[0]
    p.val = calib_dir
    page.appendPulse('Reload', label='Recargar mapas')

    def flt(name, label, default, lo, hi):
        q = page.appendFloat(name, label=label)[0]
        q.default = default
        q.val = default
        q.normMin, q.normMax = lo, hi
        return q

    flt('Gamma', 'Gamma del proyector', 2.2, 1.0, 3.0)
    flt('Blendcurve', 'Curva de blend (S)', 1.0, 0.5, 3.0)
    flt('Blendwidth', 'Adelgazar blend (x)', 1.0, 1.0, 8.0)
    flt('Blacklevel', 'Nivel de negro', 0.0, 0.0, 0.1)
    flt('Feather', 'Feather borde pantalla', 0.0, 0.0, 0.1)
    flt('Uniformity', 'Uniformidad de brillo', 0.0, 0.0, 1.0)
    m = page.appendMenu('Mode', label='Modo')[0]
    m.menuNames = ['content', 'grid', 'blendmask', 'white', 'black']
    m.menuLabels = ['Contenido', 'Rejilla de prueba', 'Máscara de blend',
                    'Blanco', 'Negro (ajustar nivel de negro)']
    m.default = 'content'
    m.val = 'content'
    gd = page.appendInt('Griddiv', label='Divisiones rejilla')[0]
    gd.default = 16
    gd.val = 16
    gd.normMin, gd.normMax = 2, 64
    d = page.appendToggle('Dither', label='Dither (anti banding)')[0]
    d.default = True
    d.val = True
    page.appendPulse('Openwindows', label='Abrir ventanas')
    page.appendPulse('Closewindows', label='Cerrar ventanas')


def build():
    calib_dir, meta = _load_calibration()
    projs = meta['projectors']
    for wmsg in _window_warnings(meta):
        print('LibreWarp AVISO:', wmsg)

    net = me.parent()
    old = net.op(COMP_NAME)
    if old is not None:
        old.destroy()
    comp = net.create(baseCOMP, COMP_NAME)
    comp.nodeX, comp.nodeY = me.nodeX + 250, me.nodeY
    _custom_pars(comp, calib_dir)

    shader = comp.create(textDAT, 'warp_shader')
    shader.text = PIXEL_SHADER
    shader.nodeX, shader.nodeY = -400, 300

    pe = comp.create(parameterexecuteDAT, 'callbacks')
    pe.text = PAREXEC
    pe.nodeX, pe.nodeY = -200, 300
    _set(pe, 'op', '..')
    _set(pe, 'pars', 'Reload Openwindows Closewindows')
    _set(pe, 'onpulse', True)
    _set(pe, 'valuechange', False)

    cin = comp.create(inTOP, 'content_in')
    cin.nodeX, cin.nodeY = -600, 0

    finals = []
    for i, p in enumerate(projs):
        y = -i * 200
        mp = comp.create(moviefileinTOP, 'map%d' % i)
        bk = comp.create(moviefileinTOP, 'black%d' % i)
        for o, key in ((mp, 'file'), (bk, 'black_file')):
            if p.get(key):
                _set(o, 'file', "parent().par.Calibdir.eval() + '/%s'" % p[key], expr=True)
            # los mapas son datos float: hay que leerlos sin perder precisión
            _set_choice(o, 'format', ['rgba32f', 'rgba32float'], ['32', 'float'])
            _set(o, 'inputcolorspace', 'passthrough')   # TD 2025+: no convertir datos
        mp.nodeX, mp.nodeY = -400, y
        bk.nodeX, bk.nodeY = -400, y - 100

        g = comp.create(glslTOP, 'warp%d' % i)
        g.nodeX, g.nodeY = -150, y
        _set(g, 'pixeldat', shader.path)
        # la salida del proyector es exactamente su resolución nativa
        _set(g, 'outputresolution', 'custom')
        _set(g, 'resolutionw', p['width'])
        _set(g, 'resolutionh', p['height'])
        _connect(mp, g, 0)
        _connect(cin, g, 1)
        _connect(bk, g, 2)
        _uniform(g, 0, 'uParams', ['parent().par.Gamma', 'parent().par.Blendcurve',
                                   'parent().par.Blacklevel', 'parent().par.Feather'])
        _uniform(g, 1, 'uMode', ['parent().par.Mode.menuIndex', 'parent().par.Griddiv',
                                 '1 if parent().par.Dither else 0', 'parent().par.Uniformity'])
        _uniform(g, 2, 'uBlend', ['max(parent().par.Blendwidth, 1.0)', '0', '0', '0'])

        nl = comp.create(nullTOP, 'proj%d' % i)
        nl.nodeX, nl.nodeY = 50, y
        _connect(g, nl, 0)
        out = comp.create(outTOP, 'out_proj%d' % i)
        out.nodeX, out.nodeY = 250, y
        _connect(nl, out, 0)
        finals.append(nl)

        if WINDOW_MODE == 'separate':
            w = comp.create(windowCOMP, 'window%d' % i)
            w.nodeX, w.nodeY = 450, y
            _set(w, 'winop', nl.path)
            _set(w, 'justifyoffsetto', 'primarydisplay')
            _set(w, 'winoffsetx', p.get('desktop_x', 0))
            _set(w, 'winoffsety', p.get('desktop_y', 0))
            _set(w, 'size', 'custom')
            _set(w, 'winw', p['width'])
            _set(w, 'winh', p['height'])
            _set(w, 'borders', False)

    # ---- canvas por display físico (p.ej. el QuadHead2Go completo: 5760x1080) ----
    groups = []
    displays = meta.get('displays') or []
    if displays:
        for di, d in enumerate(displays):
            members = [(k, p['rect']) for k, p in enumerate(projs) if p.get('display', 0) == di]
            if members:
                groups.append((d.get('name', 'display%d' % di), d['x'], d['y'],
                               d['width'], d['height'], members))
    else:   # calibraciones antiguas: caja que envuelve a todos los proyectores
        x0 = min(p.get('desktop_x', 0) for p in projs)
        y0 = min(p.get('desktop_y', 0) for p in projs)
        members = [(k, [p.get('desktop_x', 0) - x0, p.get('desktop_y', 0) - y0,
                        p['width'], p['height']]) for k, p in enumerate(projs)]
        W = max(r[0] + r[2] for _, r in members)
        H = max(r[1] + r[3] for _, r in members)
        groups.append(('canvas', x0, y0, W, H, members))

    for gi, (gname, gx, gy, W, H, members) in enumerate(groups):
        suffix = '' if len(groups) == 1 else str(gi)
        cs = comp.create(textDAT, 'canvas_shader' + suffix)
        cs.text = _canvas_shader([tuple(r) for _, r in members], H)
        cs.nodeX, cs.nodeY = 250, 300 + gi * 150
        cv = comp.create(glslTOP, 'canvas' + suffix)
        cv.nodeX, cv.nodeY = 450, -gi * 300
        _set(cv, 'pixeldat', cs.path)
        _set(cv, 'outputresolution', 'custom')
        _set(cv, 'resolutionw', W)
        _set(cv, 'resolutionh', H)
        _set_choice(cv, 'format', ['rgba8'], ['8-bit', 'fixed'])
        for slot, (k, _) in enumerate(members):
            _connect(finals[k], cv, slot)
        co = comp.create(outTOP, 'out_canvas' + suffix)
        co.nodeX, co.nodeY = 650, -gi * 300
        _connect(cv, co, 0)
        if WINDOW_MODE == 'canvas':
            w = comp.create(windowCOMP, 'window_canvas' + suffix)
            w.nodeX, w.nodeY = 650, -gi * 300 - 150
            _set(w, 'winop', cv.path)
            _set(w, 'justifyoffsetto', 'primarydisplay')
            _set(w, 'winoffsetx', gx)
            _set(w, 'winoffsety', gy)
            _set(w, 'size', 'custom')
            _set(w, 'winw', W)
            _set(w, 'winh', H)
            _set(w, 'borders', False)
        print('LibreWarp: canvas "%s" %dx%d en (%d,%d) con %d proyectores'
              % (gname, W, H, gx, gy, len(members)))

    for wmsg in warnings:
        print('  aviso:', wmsg)
    return comp


build()
