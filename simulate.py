
import json
import math
from urllib.parse import parse_qs

# ------------------------------- CONFIG (defaults) --------------------------
CFG = dict(
    T1=87.0, T2=60.0,                    # degC
    P1_mmWC=2000.0, P2_mmWC=400.0,       # gauge, mm water column
    P_atm_bar=1.0,                       # convention that gives 1.196 / 1.039 bar abs
    y_O2_dry=1.5,                        # mol % O2 on dry basis
    Cl2_kg_h=1000.0,                     # basis: Cl2 produced, kg/h
    brine_in=50.0, brine_out=56.0,       # degC (assumed; must stay < T2)
    cp_brine=3.3,                        # kJ/kg/K for ~25 wt% NaCl
    rho_brine=1190.0,                    # kg/m3 at 310 g/L
    k_gas=1.3, xT=0.70,                  # valve: heat-capacity ratio, xT factor
)
# (min, max) accepted for each input
LIMITS = dict(
    T1=(30, 99), T2=(10, 98), P1_mmWC=(50, 20000), P2_mmWC=(0, 19000),
    P_atm_bar=(0.5, 1.5), y_O2_dry=(0, 20), Cl2_kg_h=(1, 1e6),
    brine_in=(0, 90), brine_out=(1, 95), cp_brine=(2, 5), rho_brine=(1000, 1400),
    k_gas=(1.05, 1.67), xT=(0.2, 0.95),
)
MW = dict(Cl2=70.906, O2=31.998, H2O=18.015)
R = 8.314462618  # J/mol/K
CP = dict(Cl2=35.0, O2=29.6, H2O=33.7)  # J/mol/K, ideal gas, ~330 K
MMWC_PA = 9.80665


# ------------------------------- THERMO -------------------------------------
_A, _B, _C = 8.07131, 1730.63, 233.426   # Antoine, water, 1-100 C, mmHg


def p_sat_bar(T_C):
    return 10 ** (_A - _B / (_C + T_C)) * 1.33322e-3


def T_sat_C(p_bar):
    """Analytic inverse of the Antoine equation."""
    return _B / (_A - math.log10(p_bar / 1.33322e-3)) - _C


def hvap_J_mol(T_C):
    """Watson correlation from 40.66 kJ/mol @ 100 C."""
    Tc, Tb = 647.1, 373.15
    return 40660.0 * ((1 - (T_C + 273.15) / Tc) / (1 - Tb / Tc)) ** 0.38


def rho_mix(y_w, y_o2, T_C, P_bar):
    y_cl = 1 - y_w - y_o2
    mw = y_cl * MW["Cl2"] + y_o2 * MW["O2"] + y_w * MW["H2O"]
    return P_bar * 1e5 * mw * 1e-3 / (R * (T_C + 273.15)), mw


def cv_gas(W_kg_h, rho1, P1_kPa, dP_kPa, k, xT):
    """IEC 60534-2-1 gas sizing (N6 = 27.3: kg/h, kPa, kg/m3) -> Kv -> Cv."""
    x = dP_kPa / P1_kPa
    Fk = k / 1.4
    choked = x >= Fk * xT
    if choked:
        x = Fk * xT
    Y = 1 - x / (3 * Fk * xT)
    Kv = W_kg_h / (27.3 * Y * math.sqrt(x * P1_kPa * rho1))
    return 1.156 * Kv, x, choked


# ------------------------------- MODEL --------------------------------------
def simulate(c):
    P1 = c["P_atm_bar"] + c["P1_mmWC"] * MMWC_PA / 1e5
    P2 = c["P_atm_bar"] + c["P2_mmWC"] * MMWC_PA / 1e5
    dP_kPa = (P1 - P2) * 100.0
    y_o2_dry = c["y_O2_dry"] / 100.0

    n_cl2 = c["Cl2_kg_h"] / MW["Cl2"] * 1000 / 3600            # mol/s
    n_dry = n_cl2 / (1 - y_o2_dry)
    n_o2 = n_dry * y_o2_dry
    y_w1 = p_sat_bar(c["T1"]) / P1
    n_w_in = n_dry * y_w1 / (1 - y_w1)

    def wet(n_w, T, P):
        n_tot = n_dry + n_w
        yw, yo = n_w / n_tot, n_o2 / n_tot
        rho, mw = rho_mix(yw, yo, T, P)
        W = (n_cl2 * MW["Cl2"] + n_o2 * MW["O2"] + n_w * MW["H2O"]) * 3.6
        return dict(y_w=yw, rho=rho, mw=mw, W=W,
                    Q=n_tot * R * (T + 273.15) / (P * 1e5) * 3600)

    # ---- Case A: cooler at P1, valve after cooler ----
    yA = p_sat_bar(c["T2"]) / P1
    nwA = n_dry * yA / (1 - yA)
    vA = wet(nwA, c["T2"], P1)
    cvA, xA, chA = cv_gas(vA["W"], vA["rho"], P1 * 100, dP_kPa, c["k_gas"], c["xT"])
    A = dict(P_scrub=P1, n_w_out=nwA, v=vA, T_valve=c["T2"], Cv=cvA, choked=chA,
             y_exit=yA, RH_after=yA * P2 / p_sat_bar(c["T2"]) * 100,
             scrub_in_Q=wet(n_w_in, c["T1"], P1)["Q"])

    # ---- Case B: valve first (raw cell gas), cooler at P2 ----
    vB = wet(n_w_in, c["T1"], P1)
    cvB, xB, chB = cv_gas(vB["W"], vB["rho"], P1 * 100, dP_kPa, c["k_gas"], c["xT"])
    yB = p_sat_bar(c["T2"]) / P2
    nwB = n_dry * yB / (1 - yB)
    B = dict(P_scrub=P2, n_w_out=nwB, v=vB, T_valve=c["T1"], Cv=cvB, choked=chB,
             y_exit=yB, RH_after=100.0,
             scrub_in_Q=wet(n_w_in, c["T1"], P2)["Q"])
    dew_after_valve = T_sat_C(y_w1 * P2)
    T_B_match = T_sat_C(yA * P2)

    # ---- energy balance & derived metrics ----
    for r in (A, B):
        n_cond = n_w_in - r["n_w_out"]
        dT = c["T1"] - c["T2"]
        Q = ((n_cl2 * CP["Cl2"] + n_o2 * CP["O2"]) * dT
             + n_w_in * CP["H2O"] * dT
             + n_cond * hvap_J_mol(c["T2"])) / 1000.0
        m_brine = Q / (c["cp_brine"] * (c["brine_out"] - c["brine_in"])) * 3600
        r.update(n_cond=n_cond, cond_kg_h=n_cond * MW["H2O"] * 3.6,
                 resid_kg_h=r["n_w_out"] * MW["H2O"] * 3.6,
                 eff=100 * n_cond / n_w_in, Q_kW=Q,
                 brine_m3h=m_brine / c["rho_brine"],
                 dew_out=c["T2"])

    inlet_kg_h = n_w_in * MW["H2O"] * 3.6
    extra = B["resid_kg_h"] - A["resid_kg_h"]
    corr = 2 ** ((B["T_valve"] - A["T_valve"]) / 10)

    # ---- sweep: removal efficiency vs outlet temperature ----
    w1 = y_w1 / (1 - y_w1)
    sweep = []
    t = 30
    while t < c["T1"]:
        ya, yb = p_sat_bar(t) / P1, p_sat_bar(t) / P2
        sweep.append(dict(T=t,
                          A=100 * (1 - (ya / (1 - ya)) / w1),
                          B=100 * (1 - (yb / (1 - yb)) / w1)))
        t += 1

    def row(label, unit, a, b, dec, pref=None):
        return dict(label=label, unit=unit, a=a, b=b, dec=dec, pref=pref)

    rows = [
        row("Cooler pressure", "bar abs", A["P_scrub"], B["P_scrub"], 3),
        row("Cooler outlet temperature", "°C", c["T2"], c["T2"], 1),
        row("Exit-gas H2O mole fraction", "mol %", A["y_exit"] * 100, B["y_exit"] * 100, 2, "low"),
        row("Water condensed", "mol/s", A["n_cond"], B["n_cond"], 3, "high"),
        row("Water condensed", "kg/h", A["cond_kg_h"], B["cond_kg_h"], 1, "high"),
        row("Moisture removal efficiency", "%", A["eff"], B["eff"], 2, "high"),
        row("Residual water vapour in exit gas", "kg/h", A["resid_kg_h"], B["resid_kg_h"], 1, "low"),
        row("Valve inlet temperature", "°C", A["T_valve"], B["T_valve"], 1, "low"),
        row("Valve inlet water content", "mol %", A["v"]["y_w"] * 100, B["v"]["y_w"] * 100, 2, "low"),
        row("Valve inlet gas density", "kg/m³", A["v"]["rho"], B["v"]["rho"], 3, "high"),
        row("Valve inlet mass flow", "kg/h", A["v"]["W"], B["v"]["W"], 0),
        row("Valve inlet volumetric flow", "m³/h", A["v"]["Q"], B["v"]["Q"], 0, "low"),
        row("Required valve Cv (same ΔP)", "-", A["Cv"], B["Cv"], 1, "low"),
        row("Cooler inlet gas volumetric flow", "m³/h", A["scrub_in_Q"], B["scrub_in_Q"], 0, "low"),
        row("Cooling duty", "kW", A["Q_kW"], B["Q_kW"], 0),
        row("Brine flow", "m³/h", A["brine_m3h"], B["brine_m3h"], 1),
        row("Gas relative humidity after PCV / at line", "%", A["RH_after"], B["RH_after"], 1, "low"),
    ]

    wins_a = sum(1 for r in rows if r["pref"] and
                 ((r["pref"] == "high" and r["a"] > r["b"]) or (r["pref"] == "low" and r["a"] < r["b"])))
    wins_b = sum(1 for r in rows if r["pref"] and
                 ((r["pref"] == "high" and r["b"] > r["a"]) or (r["pref"] == "low" and r["b"] < r["a"])))
    better = "A" if wins_a > wins_b else "B" if wins_b > wins_a else "tie"

    notes = [
        f"Feed gas at {c['T1']:.0f} °C, {P1:.3f} bar abs carries {y_w1*100:.1f} mol% water "
        f"({inlet_kg_h:.0f} kg/h per {c['Cl2_kg_h']:.0f} kg/h Cl2).",
        f"Case B: after the PCV the water dew point falls {c['T1']:.0f} → {dew_after_valve:.1f} °C, so the gas "
        f"is superheated and nothing condenses in the valve.",
        f"Case B must cool to {T_B_match:.1f} °C to match Case A's exit moisture "
        f"({c['T2'] - T_B_match:.1f} K colder than the {c['T2']:.0f} °C target).",
        f"Case B carries {extra:+.1f} kg/h more water downstream "
        f"({extra / A['resid_kg_h'] * 100:+.1f} %).",
        f"Corrosion heuristic (rate doubles per +10 K): Case B valve runs about {corr:.1f}x faster than Case A.",
    ]
    if A["choked"] or B["choked"]:
        notes.append("Warning: the valve is choked at this ΔP/P1 ratio; Cv uses the choked limit.")
    if c["brine_out"] >= c["T2"]:
        notes.append("Warning: brine outlet temperature is not below the gas outlet temperature.")

    return dict(P1=P1, P2=P2, dP_kPa=dP_kPa, psat1=p_sat_bar(c["T1"]), psat2=p_sat_bar(c["T2"]),
                y_w1=y_w1 * 100, inlet_kg_h=inlet_kg_h, rows=rows, sweep=sweep, notes=notes,
                better=better, wins=dict(A=wins_a, B=wins_b),
                kpi=dict(effA=A["eff"], effB=B["eff"], cvA=A["Cv"], cvB=B["Cv"],
                         TvA=A["T_valve"], TvB=B["T_valve"],
                         residA=A["resid_kg_h"], residB=B["resid_kg_h"], extra=extra, corr=corr),
                T2=c["T2"])


def parse_inputs(query):
    c, errors = dict(CFG), []
    for k, default in CFG.items():
        if k in query and query[k][0] != "":
            try:
                v = float(query[k][0])
            except ValueError:
                errors.append(f"{k}: not a number")
                continue
            lo, hi = LIMITS[k]
            if not (lo <= v <= hi) or math.isnan(v):
                errors.append(f"{k}: must be between {lo} and {hi}")
                continue
            c[k] = v
    if not errors:
        if c["T2"] >= c["T1"]:
            errors.append("T2 must be lower than T1")
        if c["P2_mmWC"] >= c["P1_mmWC"]:
            errors.append("P2 must be lower than P1")
        if c["brine_out"] <= c["brine_in"]:
            errors.append("Brine outlet must be higher than brine inlet")
    return c, errors


# ------------------------------- DASHBOARD (HTML) ---------------------------
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PCV-101 Placement Study</title>
<style>
:root{--bg:#f5f6f8;--card:#fff;--tx:#1b2430;--mut:#64707f;--bd:#dfe3e9;--a:#0b7a75;--b:#c2410c;--ok:#15803d;--ac:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#0f141a;--card:#171e27;--tx:#e6ebf1;--mut:#92a0b1;--bd:#2a3441;--a:#2dd4bf;--b:#fb923c;--ok:#4ade80;--ac:#60a5fa}}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--tx)}
header{padding:20px 24px 4px}h1{margin:0;font-size:22px}header p{margin:4px 0 0;color:var(--mut);font-size:14px}
.wrap{display:grid;grid-template-columns:290px 1fr;gap:16px;padding:16px 24px 32px;align-items:start}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:16px}
.card h2{margin:0 0 10px;font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut)}
label{display:block;font-size:12px;color:var(--mut);margin:10px 0 3px}
input{width:100%;padding:7px 9px;border:1px solid var(--bd);border-radius:7px;background:var(--bg);color:var(--tx);font-size:14px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:8px}
button{margin-top:14px;width:100%;padding:8px;border:1px solid var(--bd);background:transparent;color:var(--tx);border-radius:7px;cursor:pointer}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-bottom:16px}
.kpi .t{font-size:12px;color:var(--mut)}.kpi .v{display:flex;gap:14px;align-items:baseline;margin-top:4px}
.kpi .v b{font-size:24px;font-weight:600}.kpi .a{color:var(--a)}.kpi .b{color:var(--b)}
.kpi .s{font-size:11px;color:var(--mut);margin-top:2px}
.verdict{border-left:4px solid var(--ok);margin-bottom:16px}
.verdict h3{margin:0 0 4px;font-size:16px}.verdict p{margin:0;color:var(--mut);font-size:14px}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{padding:7px 8px;border-bottom:1px solid var(--bd);text-align:right}
th:first-child,td:first-child{text-align:left}th{font-size:12px;color:var(--mut);font-weight:600}
td.win{font-weight:700;color:var(--ok)}td .u{color:var(--mut);font-size:11px;margin-left:3px}
.tw{overflow-x:auto}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
ul{margin:0;padding-left:18px;color:var(--mut);font-size:13.5px}li{margin:4px 0}
.err{color:#dc2626;font-size:13px;margin-top:10px}svg text{fill:var(--mut);font-size:11px}
.leg span{display:inline-block;margin-right:14px;font-size:12px;color:var(--mut)}.leg i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}
.foot{font-size:12px;color:var(--mut);margin-top:16px}
</style></head><body>
<header><h1>PCV-101 Placement Study</h1>
<p>Wet chlorine cooler: Case A (PCV after cooler, as sketched) vs Case B (PCV before cooler). Live thermodynamic and hydraulic comparison.</p></header>
<div class="wrap">
<aside class="card"><h2>Inputs</h2><div id="form"></div><div class="err" id="err"></div>
<button id="reset">Reset to defaults</button></aside>
<main>
<div class="card verdict" id="verdict"></div>
<div class="kpis" id="kpis"></div>
<div class="card tw"><h2>Side-by-side metrics</h2><table id="tbl"></table></div>
<div class="grid2">
<div class="card"><h2>Removal efficiency vs cooler outlet temperature</h2>
<div class="leg"><span><i style="background:var(--a)"></i>Case A</span><span><i style="background:var(--b)"></i>Case B</span></div>
<svg id="chart" viewBox="0 0 520 260" width="100%"></svg></div>
<div class="card"><h2>Key observations</h2><ul id="notes"></ul></div></div>
<p class="foot">Ideal gas and Dalton mixing; water vapour pressure from the Antoine equation; pure-water condensate (indirect brine-cooled HX); Cl2 solubility and hydrolysis neglected; valve Cv per IEC 60534 compressible-gas equation. Results scale linearly with the Cl2 flow basis; Cv and corrosion factors are estimates for comparison, not vendor sizing.</p>
</main></div>
<script>
const DEF=__DEFAULTS__;
const FIELDS=[["T1","Feed gas T1 (°C)"],["T2","Outlet T2 (°C)"],["P1_mmWC","P1 (mm WC, gauge)"],["P2_mmWC","P2 (mm WC, gauge)"],
["Cl2_kg_h","Cl2 basis (kg/h)"],["y_O2_dry","O2, dry basis (mol %)"],["brine_in","Brine in (°C)"],["brine_out","Brine out (°C)"],
["cp_brine","Brine cp (kJ/kg·K)"],["rho_brine","Brine density (kg/m³)"],["k_gas","Gas k (Cp/Cv)"],["xT","Valve xT"],["P_atm_bar","Atmospheric (bar)"]];
const $=id=>document.getElementById(id);
$("form").innerHTML=FIELDS.map(([k,l])=>`<label for="${k}">${l}</label><input id="${k}" type="number" step="any" value="${DEF[k]}">`).join("");
const fmt=(v,d)=>Number(v).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});
let timer;function schedule(){clearTimeout(timer);timer=setTimeout(run,250)}
FIELDS.forEach(([k])=>$(k).addEventListener("input",schedule));
$("reset").onclick=()=>{FIELDS.forEach(([k])=>$(k).value=DEF[k]);run()};
async function run(){
  const q=new URLSearchParams();FIELDS.forEach(([k])=>q.set(k,$(k).value));
  const r=await fetch("/api/simulate?"+q);const d=await r.json();
  if(!r.ok){$("err").textContent=(d.errors||["error"]).join("; ");return}
  $("err").textContent="";render(d)}
function kpi(t,a,b,dec,u,sub){return `<div class="card kpi"><div class="t">${t}</div><div class="v"><span class="a"><small>A</small> <b>${fmt(a,dec)}</b></span><span class="b"><small>B</small> <b>${fmt(b,dec)}</b></span> <small>${u}</small></div><div class="s">${sub}</div></div>`}
function render(d){
  const k=d.kpi,w=d.better;
  $("verdict").innerHTML=w==="A"?`<h3>Case A (PCV after the cooler, as sketched) is the better layout</h3><p>It wins ${d.wins.A} of ${d.wins.A+d.wins.B} comparable metrics: ${fmt(k.effA-k.effB,1)} points more moisture removed, ${fmt(k.extra,1)} kg/h less water carried downstream, a ${fmt(k.TvB-k.TvA,0)} °C cooler valve and a Cv ${fmt((k.cvB/k.cvA-1)*100,0)}% smaller.</p>`
   :w==="B"?`<h3>Case B (PCV before the cooler) scores better at these inputs</h3><p>It wins ${d.wins.B} of ${d.wins.A+d.wins.B} comparable metrics. Check the table for details.</p>`
   :`<h3>The two layouts are tied at these inputs</h3><p>${d.wins.A} metrics each.</p>`;
  $("kpis").innerHTML=kpi("Moisture removal efficiency",k.effA,k.effB,1,"%","higher is better")
   +kpi("Residual water in exit gas",k.residA,k.residB,1,"kg/h","lower is better")
   +kpi("Valve inlet temperature",k.TvA,k.TvB,0,"°C","corrosion about "+fmt(k.corr,1)+"x faster in B")
   +kpi("Required valve Cv",k.cvA,k.cvB,1,"","same ΔP of "+fmt(d.dP_kPa,1)+" kPa");
  let h=`<tr><th>Metric</th><th>Case A (sketch)</th><th>Case B (PCV first)</th></tr>`;
  d.rows.forEach(r=>{let wa="",wb="";
    if(r.pref==="high"){if(r.a>r.b)wa="win";else if(r.b>r.a)wb="win"}
    if(r.pref==="low"){if(r.a<r.b)wa="win";else if(r.b<r.a)wb="win"}
    const u=r.unit==="-"?"":`<span class="u">${r.unit}</span>`;
    h+=`<tr><td>${r.label}</td><td class="${wa}">${fmt(r.a,r.dec)}${u}</td><td class="${wb}">${fmt(r.b,r.dec)}${u}</td></tr>`});
  $("tbl").innerHTML=h;
  $("notes").innerHTML=d.notes.map(n=>`<li>${n}</li>`).join("");
  chart(d)}
function chart(d){
  const s=d.sweep,W=520,H=260,L=44,Rr=12,T=12,B=34;if(!s.length){$("chart").innerHTML="";return}
  const xs=s.map(p=>p.T),xmin=xs[0],xmax=xs[xs.length-1];
  const ys=s.flatMap(p=>[p.A,p.B]),ymin=Math.floor(Math.min(...ys)/10)*10,ymax=Math.ceil(Math.max(...ys)/10)*10||100;
  const X=x=>L+(x-xmin)/(xmax-xmin||1)*(W-L-Rr),Y=y=>H-B-(y-ymin)/(ymax-ymin||1)*(H-B-T);
  const path=key=>s.map((p,i)=>(i?"L":"M")+X(p.T).toFixed(1)+" "+Y(p[key]).toFixed(1)).join("");
  let g="";for(let y=ymin;y<=ymax;y+=(ymax-ymin)/5){g+=`<line x1="${L}" x2="${W-Rr}" y1="${Y(y)}" y2="${Y(y)}" stroke="currentColor" opacity=".12"/><text x="${L-6}" y="${Y(y)+4}" text-anchor="end">${y.toFixed(0)}</text>`}
  for(let x=Math.ceil(xmin/10)*10;x<=xmax;x+=10){g+=`<text x="${X(x)}" y="${H-B+16}" text-anchor="middle">${x}</text>`}
  const cx=X(d.T2);
  $("chart").innerHTML=g+`<path d="${path("A")}" fill="none" stroke="var(--a)" stroke-width="2.2"/><path d="${path("B")}" fill="none" stroke="var(--b)" stroke-width="2.2"/>
  <line x1="${cx}" x2="${cx}" y1="${T}" y2="${H-B}" stroke="var(--ac)" stroke-dasharray="4 3"/><text x="${cx+4}" y="${T+10}">T2 = ${d.T2}</text>
  <text x="${(L+W)/2}" y="${H-4}" text-anchor="middle">Cooler outlet temperature (°C)</text>
  <text transform="rotate(-90)" x="${-(H-B)/2}" y="11" text-anchor="middle">Removal (%)</text>`}
run();
</script></body></html>"""


# ------------------------------- HTTP LAYER ---------------------------------
def route(path, query_string):
    """Return (status, content_type, body_bytes)."""
    if path in ("/api/simulate", "/simulate"):
        c, errors = parse_inputs(parse_qs(query_string))
        if errors:
            return "400 Bad Request", "application/json", json.dumps({"errors": errors}).encode()
        return "200 OK", "application/json", json.dumps(simulate(c)).encode()
    if path in ("/", "/index.html"):
        html = PAGE.replace("__DEFAULTS__", json.dumps(CFG))
        return "200 OK", "text/html; charset=utf-8", html.encode()
    return "404 Not Found", "text/plain", b"Not found"


def app(environ, start_response):  # WSGI entry point (Vercel looks for `app`)
    status, ctype, body = route(environ.get("PATH_INFO", "/"), environ.get("QUERY_STRING", ""))
    start_response(status, [("Content-Type", ctype), ("Content-Length", str(len(body))),
                            ("Cache-Control", "no-store")])
    return [body]


if __name__ == "__main__":  # local run: python app.py
    from wsgiref.simple_server import make_server
    print("PCV-101 dashboard on http://localhost:8000")
    make_server("", 8000, app).serve_forever()