"""Draw the README explainer diagrams (docs/images/{optimiser,backtest,pipeline}.svg).

They are animated SVGs in the banner's style. The numbers are fixed: the optimiser uses the real
gameweek 6 options (if the manager had had 1 free transfer) and the backtest the figures in
docs/backtest.md, so update them here when those change.

    uv run python -m scripts.readme_diagrams
"""

# ruff: noqa: E501  (SVG markup reads better on one line)

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"

# Banner palette (surfaces, ink, accents) + validated data colours (dark surface #0D1410).
BG, CARD, LINE, GRID = "#0D1410", "#122019", "#2A3A30", "#16201A"
INK, INK2, MUTED, FAINT = "#EAF2EC", "#B7C7BC", "#9DB0A3", "#4A5E50"
LIME, AMBER = "#B8F34A", "#FFB547"
GAIN, HIT, OTHER = "#6C9F2A", "#D45BA0", "#7E8F84"

SANS = 'system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif'
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


class Anim:
    """Per-element keyframes on one shared loop, so every loop stays in step. The element's
    normal style is its final state, which is what reduced-motion readers see."""

    def __init__(self, loop: float, repeat: bool = True) -> None:
        self.loop, self.repeat, self.n = loop, repeat, 0
        self.rules: list[str] = []

    def _pct(self, t: float) -> str:
        return f"{max(0.0, min(100.0, 100 * t / self.loop)):.2f}%"

    def add(
        self,
        hidden: str,
        shown: str,
        start: float,
        dur: float = 0.5,
        end: float | None = None,
        origin: str = "center",
    ) -> str:
        """Hidden until `start`, then eased in to the final state; hidden again at `end`
        (default: just before the loop restarts)."""
        self.n += 1
        name = f"a{self.n}"
        end = self.loop - 0.4 if end is None else end
        if self.repeat:
            frames = (
                f"0%,{self._pct(start)}{{{hidden}}}"
                f"{self._pct(start + dur)},{self._pct(end)}{{{shown}}}"
                f"{self._pct(min(end + 0.35, self.loop))},100%{{{hidden}}}"
            )
            it = "infinite"
        else:
            frames = f"0%,{self._pct(start)}{{{hidden}}}{self._pct(start + dur)},100%{{{shown}}}"
            it = "1"
        self.rules.append(f"@keyframes {name}{{{frames}}}")
        self.rules.append(
            f".{name}{{animation:{name} {self.loop}s cubic-bezier(.2,.8,.3,1) {it} both;"
            f"transform-box:fill-box;transform-origin:{origin}}}"
        )
        return name

    def fade(self, start: float, dur: float = 0.5, end: float | None = None) -> str:
        return self.add(
            "opacity:0;transform:translateY(8px)", "opacity:1;transform:none", start, dur, end
        )

    def grow(self, start: float, dur: float = 0.6, origin: str = "bottom") -> str:
        axis = "scaleX" if origin in ("left", "right") else "scaleY"
        return self.add(f"transform:{axis}(0)", f"transform:{axis}(1)", start, dur, origin=origin)

    def dim(self, start: float, dur: float = 0.4) -> str:
        return self.add("opacity:1", "opacity:.3", start, dur)

    def style(self) -> str:
        return (
            "<style>"
            + f".sans{{font-family:{SANS}}}.mono{{font-family:{MONO}}}"
            + "".join(self.rules)
            + "@media (prefers-reduced-motion:reduce){*{animation:none!important}}</style>"
        )


def frame(w: int, h: int, title: str, desc: str, body: str, anim: Anim, defs: str = "") -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-labelledby="t d">
<title id="t">{title}</title>
<desc id="d">{desc}</desc>
{anim.style()}
<defs>
<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" stroke="{GRID}" stroke-width="1"/></pattern>
<pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="7" height="7" fill="{HIT}"/><line x1="0" y1="0" x2="0" y2="7" stroke="#8E2F66" stroke-width="3"/></pattern>
{defs}
</defs>
<rect width="{w}" height="{h}" rx="24" fill="{BG}"/>
<rect width="{w}" height="{h}" rx="24" fill="url(#grid)"/>
<rect x=".5" y=".5" width="{w - 1}" height="{h - 1}" rx="24" fill="none" stroke="{LINE}"/>
{body}
</svg>
"""


def text(
    x: float,
    y: float,
    s: str,
    size: int = 16,
    fill: str = INK,
    cls: str = "sans",
    weight: int = 400,
    anchor: str = "start",
    extra: str = "",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {extra}>{s}</text>'
    )


def vbar(x: float, w: float, y0: float, y1: float, fill: str, r: float = 4) -> str:
    """A vertical bar from baseline y0 to y1, rounded only at the far (data) end."""
    if abs(y1 - y0) < 0.5:
        return ""
    r = min(r, abs(y1 - y0), w / 2)
    if y1 < y0:  # grows up
        return (
            f'<path d="M{x} {y0}V{y1 + r}Q{x} {y1} {x + r} {y1}H{x + w - r}'
            f'Q{x + w} {y1} {x + w} {y1 + r}V{y0}Z" fill="{fill}"/>'
        )
    return (
        f'<path d="M{x} {y0}V{y1 - r}Q{x} {y1} {x + r} {y1}H{x + w - r}'
        f'Q{x + w} {y1} {x + w} {y1 - r}V{y0}Z" fill="{fill}"/>'
    )


def hbar(x0: float, x1: float, y: float, h: float, fill: str, r: float = 4) -> str:
    r = min(r, x1 - x0, h / 2)
    return (
        f'<path d="M{x0} {y}H{x1 - r}Q{x1} {y} {x1} {y + r}V{y + h - r}'
        f'Q{x1} {y + h} {x1 - r} {y + h}H{x0}Z" fill="{fill}"/>'
    )


def step_badge(x: float, y: float, n: int) -> str:
    return f'<circle cx="{x}" cy="{y}" r="15" fill="{LIME}"/>' + text(
        x, y + 5.5, str(n), 16, BG, "mono", 800, "middle"
    )


def check(x: float, y: float) -> str:
    return (
        f'<circle cx="{x}" cy="{y}" r="10" fill="{GAIN}"/>'
        f'<path d="M{x - 4.5} {y}l3 3.2 6-6.4" fill="none" stroke="{BG}" stroke-width="2.4" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
    )


# --------------------------------------------------------------------------------------------
# 1. The optimiser, in three steps (real gameweek 6 numbers).
def optimiser() -> str:
    a = Anim(loop=19.0)
    W, H = 1200, 600
    out: list[str] = []
    cards = [(24, 368), (416, 368), (808, 368)]
    for x, w in cards:
        out.append(
            f'<rect x="{x}" y="24" width="{w}" height="{H - 48}" rx="18" fill="{CARD}" stroke="{LINE}"/>'
        )

    # --- panel 1: predictions
    x = 24
    out.append(
        f'<g class="{a.fade(0.1)}">{step_badge(x + 36, 66, 1)}'
        + text(x + 62, 72, "Predict every player", 22, INK, weight=700)
        + text(x + 26, 108, "Expected points (xPts) for the next", 15, MUTED)
        + text(x + 26, 128, "gameweek, from the trained model", 15, MUTED)
        + "</g>"
    )
    players = [
        ("B.Fernandes", "MID · MUN · £11.9m", 6.4),
        ("Tarkowski", "DEF · EVE · £6.2m", 5.6),
        ("Haaland", "FWD · MCI · £15.6m", 5.4),
        ("Thiaw", "DEF · NEW · £5.0m", 5.1),
        ("Gibbs-White", "MID · NFO · £8.0m", 5.1),
        ("E.Le Fée", "MID · SUN · £5.7m", 5.0),
    ]
    bx0, bw = x + 196, 110
    for i, (name, meta, xp) in enumerate(players):
        y = 170 + i * 56
        t0 = 0.5 + i * 0.22
        out.append(
            f'<g class="{a.fade(t0)}">'
            + text(x + 26, y, name, 17, INK, weight=600)
            + text(x + 26, y + 20, meta, 12, MUTED, "mono")
            + "</g>"
        )
        out.append(f'<rect x="{bx0}" y="{y - 13}" width="{bw}" height="16" rx="4" fill="{GRID}"/>')
        out.append(
            f'<g class="{a.grow(t0 + 0.2, origin="left")}">'
            + hbar(bx0, bx0 + bw * xp / 6.5, y - 13, 16, GAIN)
            + "</g>"
        )
        out.append(
            f'<g class="{a.fade(t0 + 0.5)}">'
            + text(bx0 + bw + 10, y + 1, f"{xp:.1f}", 15, INK, "mono", 700)
            + "</g>"
        )
    out.append(
        f'<g class="{a.fade(2.0)}">'
        + text(x + 26, 518, "+ 661 more players", 15, MUTED, "mono")
        + text(x + 26, 548, "Haaland scores well but costs £15.6m:", 13, MUTED)
        + text(x + 26, 566, "the budget decides whether he fits.", 13, MUTED)
        + "</g>"
    )

    # --- panel 2: the squad puzzle
    x = 416
    out.append(
        f'<g class="{a.fade(2.4)}">{step_badge(x + 36, 66, 2)}'
        + text(x + 62, 72, "Solve the squad puzzle", 22, INK, weight=700)
        + text(x + 26, 108, "A mixed-integer program (HiGHS)", 15, MUTED)
        + text(x + 26, 128, "finds the provably best squad", 15, MUTED)
        + "</g>"
    )
    out.append(
        f'<g class="{a.fade(2.9)}"><rect x="{x + 22}" y="150" width="324" height="70" rx="10" fill="{BG}" stroke="{LINE}"/>'
        + text(x + 38, 176, "MAXIMISE", 12, AMBER, "mono", 700, extra='letter-spacing="2"')
        + text(x + 38, 204, "xPts of the XI + captain ×2", 17, INK, weight=600)
        + "</g>"
    )
    out.append(
        f'<g class="{a.fade(3.3)}">'
        + text(x + 26, 252, "SO THAT", 12, AMBER, "mono", 700, extra='letter-spacing="2"')
        + "</g>"
    )
    rules = [
        "15 players: 2 GK, 5 DEF, 5 MID, 3 FWD",
        "at most 3 from any one club",
        "cost ≤ your selling value + bank",
        "a legal starting XI and formation",
        "exactly k players swapped out",
    ]
    for i, r in enumerate(rules):
        y = 284 + i * 38
        t0 = 3.6 + i * 0.45
        out.append(
            f'<g class="{a.add("opacity:0;transform:scale(.3)", "opacity:1;transform:none", t0, 0.35)}">{check(x + 36, y - 5)}</g>'
        )
        out.append(f'<g class="{a.fade(t0 + 0.1)}">' + text(x + 56, y, r, 15, INK2) + "</g>")
    out.append(
        f'<g class="{a.fade(6.1)}">'
        + text(x + 26, 498, "Solved once per number of transfers:", 14, MUTED)
        + "</g>"
    )
    for k in range(6):
        cx = x + 26 + k * 54
        t0 = 6.4 + k * 0.25
        out.append(
            f'<g class="{a.add("opacity:0;transform:scale(.4)", "opacity:1;transform:none", t0, 0.3)}">'
            f'<rect x="{cx}" y="514" width="46" height="36" rx="18" fill="{GRID}" stroke="{GAIN}"/>'
            + text(cx + 23, 538, f"k={k}", 14, INK, "mono", 700, "middle")
            + "</g>"
        )

    # --- panel 3: how many transfers (GW6, 1 free transfer)
    x = 808
    out.append(
        f'<g class="{a.fade(8.0)}">{step_badge(x + 36, 66, 3)}'
        + text(x + 62, 72, "Pick how many transfers", 22, INK, weight=700)
        + text(x + 26, 108, "Gameweek 6, with 1 free transfer:", 15, MUTED)
        + text(x + 26, 128, "each extra transfer costs 4 points", 15, MUTED)
        + "</g>"
    )
    # legend
    lg = [
        (f'<rect x="0" y="-11" width="14" height="14" rx="3" fill="{GAIN}"/>', "points gained"),
        ('<rect x="0" y="-11" width="14" height="14" rx="3" fill="url(#hatch)"/>', "hit (−4 each)"),
        (
            f'<circle cx="7" cy="-4" r="6" fill="{INK}" stroke="{CARD}" stroke-width="2"/>',
            "net gain",
        ),
        (
            f'<path d="M0 -4h16" stroke="{AMBER}" stroke-width="2" stroke-dasharray="4 3"/>',
            "needed: +0.5 each",
        ),
    ]
    for i, (mark, label) in enumerate(lg):
        lx = x + 26 + (i % 2) * 170
        ly = 160 + (i // 2) * 24
        out.append(
            f'<g transform="translate({lx} {ly})"><g class="{a.fade(8.3)}">{mark}'
            + text(22, 0, label, 13, INK2)
            + "</g></g>"
        )

    gain = [0, 2.20, 4.01, 5.57, 6.72, 7.51]
    hit = [0, 0, 4, 8, 12, 16]
    net = [g - h for g, h in zip(gain, hit, strict=True)]
    y0, s = 322, 9.0
    gx0, gw = x + 40, 52
    out.append(
        f'<line x1="{x + 26}" x2="{x + 346}" y1="{y0}" y2="{y0}" stroke="{FAINT}" stroke-width="1.5"/>'
    )
    out.append(text(x + 22, y0 + 4, "0", 12, MUTED, "mono", anchor="end"))
    for k in range(6):
        cx = gx0 + k * gw
        bw = 26
        bxk = cx + (gw - bw) / 2 - 6
        grp: list[str] = []
        if gain[k] > 0:
            grp.append(
                f'<g class="{a.grow(9.0 + k * 0.18, origin="bottom")}">'
                + vbar(bxk, bw, y0 - 1, y0 - 1 - gain[k] * s, GAIN)
                + "</g>"
            )
        if hit[k] > 0:
            grp.append(
                f'<g class="{a.grow(10.6 + k * 0.18, origin="top")}">'
                + vbar(bxk, bw, y0 + 1, y0 + 1 + hit[k] * s, "url(#hatch)")
                + "</g>"
            )
        # threshold for this k
        if k > 0:
            ty = y0 - 0.5 * k * s
            grp.append(
                f'<g class="{a.fade(12.4, 0.4)}"><path d="M{bxk - 6} {ty}h{bw + 12}" stroke="{AMBER}" '
                f'stroke-width="2" stroke-dasharray="4 3"/></g>'
            )
        ny = y0 - net[k] * s
        grp.append(
            f'<g class="{a.add("opacity:0;transform:scale(0)", "opacity:1;transform:none", 11.8 + k * 0.12, 0.35)}">'
            f'<circle cx="{bxk + bw / 2}" cy="{ny}" r="6.5" fill="{INK}" stroke="{CARD}" stroke-width="2.5"/></g>'
        )
        lab = (
            "0" if k == 0 else ("0.0" if abs(net[k]) < 0.05 else f"{net[k]:+.1f}".replace("-", "−"))
        )
        netrow = (
            f'<g class="{a.fade(12.0 + k * 0.12)}">'
            + text(bxk + bw / 2, 524, lab, 13, INK, "mono", 700, "middle")
            + "</g>"
        )
        body = "".join(grp)
        if k >= 2:  # not worth it: the gain per transfer is below +0.5
            body = f'<g class="{a.dim(13.3)}">{body}</g>'
            netrow = f'<g class="{a.dim(13.3)}">{netrow}</g>'
        out.append(body + netrow)
        out.append(text(bxk + bw / 2, 498, str(k), 15, INK2, "mono", 700, "middle"))
    out.append(text(x + 22, 498, "k", 12, MUTED, "mono", anchor="end"))
    out.append(text(x + 22, 524, "net", 12, MUTED, "mono", anchor="end"))

    # the pick
    wx = gx0 + gw + (gw - 26) / 2 - 6 + 13
    out.append(
        f'<g class="{a.add("opacity:0;transform:scale(.6)", "opacity:1;transform:none", 13.9, 0.4)}">'
        f'<circle cx="{wx}" cy="{y0 - 2.2 * s}" r="13" fill="none" stroke="{LIME}" stroke-width="2.5"/></g>'
    )
    out.append(
        f'<g class="{a.fade(14.3)}"><rect x="{x + 152}" y="200" width="196" height="50" rx="12" fill="{BG}" stroke="{LIME}"/>'
        + text(x + 168, 221, "✓ Make 1 transfer", 16, LIME, weight=700)
        + text(x + 168, 240, "+2.2 xPts, no hit", 13, INK2, "mono")
        + "</g>"
    )
    out.append(
        f'<g class="{a.fade(14.6)}">'
        + text(x + 26, 556, "2 or more: the hits cancel out the gain", 13, MUTED)
        + "</g>"
    )

    return frame(
        W,
        H,
        "How the optimiser picks transfers",
        "Three steps. 1: every player gets expected points (for example B.Fernandes 6.4, "
        "Tarkowski 5.6, Haaland 5.4). 2: a mixed-integer program picks the best 15-man squad "
        "within the FPL rules and budget, once for each number of transfers from 0 to 5. "
        "3: for gameweek 6 with one free transfer, net gains are 0, +2.2, 0.0, -2.4, -5.3 "
        "and -8.5 points for 0 to 5 transfers; only 1 transfer clears the +0.5 per transfer "
        "bar, so it recommends 1 transfer.",
        "\n".join(out),
        a,
    )


# --------------------------------------------------------------------------------------------
# 2. Backtest: average points of each gameweek's top-30 picks.
def backtest() -> str:
    a = Anim(loop=3.2, repeat=False)
    W, H = 1200, 380
    out: list[str] = []
    out.append(text(48, 64, "Average points of each gameweek's top-30 picks", 24, INK, weight=700))
    out.append(
        text(48, 92, "2025-26 season, every gameweek predicted only from data before it", 15, MUTED)
    )
    # legend
    out.append(
        f'<rect x="830" y="52" width="14" height="14" rx="3" fill="{GAIN}"/>'
        + text(852, 64, "this model", 14, INK2)
        + f'<rect x="960" y="52" width="14" height="14" rx="3" fill="{OTHER}"/>'
        + text(982, 64, "the alternative", 14, INK2)
    )
    x0, scale = 400, 116.0  # px per point
    for v in range(0, 6):
        gx = x0 + v * scale
        out.append(
            f'<line x1="{gx}" x2="{gx}" y1="120" y2="330" stroke="{GRID}" stroke-width="1"/>'
        )
        out.append(text(gx, 352, str(v), 13, MUTED, "mono", anchor="middle"))
    groups = [
        ("vs a recent-form baseline", "all gameweeks", "recent form", 4.45, 3.76),
        ("vs FPL's own prediction", "9 gameweeks with ep_next", "FPL ep_next", 4.52, 3.93),
    ]
    for g, (title, sub, other, m, o) in enumerate(groups):
        y = 128 + g * 104
        out.append(text(48, y + 26, title, 17, INK, weight=600))
        out.append(text(48, y + 48, sub, 13, MUTED))
        for j, (label, val, col) in enumerate([("this model", m, GAIN), (other, o, OTHER)]):
            by = y + 8 + j * 36
            out.append(text(x0 - 12, by + 21, label, 13, INK2, "mono", anchor="end"))
            out.append(
                f'<g class="{a.grow(0.3 + g * 0.4 + j * 0.15, 0.9, origin="left")}">'
                + hbar(x0, x0 + val * scale, by, 30, col)
                + "</g>"
            )
            out.append(
                f'<g class="{a.fade(1.1 + g * 0.4 + j * 0.15)}">'
                + text(x0 + val * scale + 10, by + 21, f"{val:.2f}", 15, INK, "mono", 700)
                + "</g>"
            )
        pct = round(100 * (m / o - 1))
        out.append(
            f'<g class="{a.fade(1.8 + g * 0.4)}">'
            + text(1150, y + 48, f"+{pct}%", 26, LIME, "mono", 800, "end")
            + "</g>"
        )
    return frame(
        W,
        H,
        "Backtest: model vs baselines",
        "Average points of each gameweek's top-30 picks in 2025-26: this model 4.45 vs a "
        "recent-form baseline 3.76 (+18%); this model 4.52 vs FPL's ep_next 3.93 on the 9 "
        "gameweeks where ep_next is available (+15%).",
        "\n".join(out),
        a,
    )


# --------------------------------------------------------------------------------------------
# 3. The daily pipeline.
def pipeline() -> str:
    a = Anim(loop=9.0)
    W, H = 1200, 400
    out: list[str] = []

    def node(x: float, y: float, w: float, label: str, sub: str, t: float) -> str:
        flash = a.add(f"stroke:{LINE}", f"stroke:{LIME}", t, 0.25, end=t + 1.0)
        return (
            f'<rect class="{flash}" x="{x}" y="{y}" width="{w}" height="66" rx="14" fill="{CARD}" stroke="{LIME}" stroke-width="2"/>'
            + text(x + w / 2, y + 30, label, 18, INK, "mono", 700, "middle")
            + text(x + w / 2, y + 51, sub, 12, MUTED, anchor="middle")
        )

    def arrow(x1: float, x2: float, y: float) -> str:
        return (
            f'<path d="M{x1} {y}H{x2 - 2}m-6-6l6 6-6 6" fill="none" stroke="{FAINT}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )

    y = 116
    out.append(text(48, 56, "Every morning at 10:00 UK", 22, INK, weight=700))
    out.append(
        text(
            48,
            82,
            "Cloud Scheduler starts a Cloud Workflow that runs four short-lived Cloud Run jobs",
            15,
            MUTED,
        )
    )
    steps = [
        (48, 150, "ingest", "FPL API → Parquet", 0.4),
        (238, 150, "predict", "xPts per player", 1.4),
        (428, 160, "optimise", "best plan per user", 2.4),
    ]
    for x, w, lab, sub, t in steps:
        out.append(node(x, y, w, lab, sub, t))
    out.append(arrow(200, 236, y + 33))
    out.append(arrow(390, 426, y + 33))
    out.append(arrow(590, 628, y + 33))
    # decision
    dx, dy = 700, y + 33
    flash = a.add(f"stroke:{LINE}", f"stroke:{AMBER}", 3.4, 0.25, end=4.4)
    out.append(
        f'<path class="{flash}" d="M{dx - 70} {dy}L{dx} {dy - 46}L{dx + 70} {dy}L{dx} {dy + 46}Z" fill="{CARD}" stroke="{AMBER}" stroke-width="2"/>'
    )
    out.append(text(dx, dy - 4, "deadline", 14, INK, "mono", 700, "middle"))
    out.append(text(dx, dy + 14, "today?", 14, INK, "mono", 700, "middle"))
    # yes -> notify -> phone
    out.append(arrow(772, 818, dy))
    out.append(text(795, dy - 10, "yes", 13, LIME, "mono", 700, "middle"))
    out.append(node(820, y, 140, "notify", "Telegram", 4.4))
    out.append(arrow(962, 996, dy))
    toast = a.fade(5.0, 0.4, end=8.2)
    out.append(
        f'<g class="{toast}"><rect x="998" y="{y + 6}" width="160" height="54" rx="12" fill="{GRID}" stroke="{LIME}" stroke-opacity=".7"/>'
        + text(1014, y + 29, "Gameweek plan", 14, INK, weight=600)
        + text(1014, y + 48, "pitch + transfers", 12, MUTED, "mono")
        + "</g>"
    )
    # no
    out.append(
        f'<path d="M{dx} {dy + 48}V{dy + 84}" stroke="{FAINT}" stroke-width="2" stroke-dasharray="4 4"/>'
    )
    out.append(text(dx + 10, dy + 72, "no", 13, MUTED, "mono", 700))
    out.append(text(dx, dy + 104, "data refreshed, nothing sent", 13, MUTED, anchor="middle"))
    out.append(
        text(dx, dy + 122, "(a test run sends to testers anyway)", 12, FAINT, anchor="middle")
    )

    # train, weekly
    ty = 272
    out.append(
        f'<rect x="238" y="{ty}" width="150" height="58" rx="14" fill="{CARD}" stroke="{LINE}" stroke-dasharray="5 4" stroke-width="2"/>'
    )
    out.append(text(313, ty + 26, "train", 18, INK, "mono", 700, "middle"))
    out.append(text(313, ty + 45, "weekly, Tue 06:00", 12, MUTED, anchor="middle"))
    out.append(
        f'<path d="M313 {ty - 2}V{y + 70}m-6 6l6-6 6 6" fill="none" stroke="{FAINT}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    out.append(text(323, ty - 20, "new model, only if it", 12, MUTED))
    out.append(text(323, ty - 5, "passes the quality gate", 12, MUTED))

    # stores
    out.append(
        f'<rect x="48" y="352" width="540" height="30" rx="8" fill="{GRID}"/>'
        + text(60, 372, "Cloud Storage · Parquet data, predictions, models", 13, INK2, "mono")
    )
    out.append(
        f'<rect x="600" y="352" width="360" height="30" rx="8" fill="{GRID}"/>'
        + text(612, 372, "Firestore · users, sent-once claims", 13, INK2, "mono")
    )

    # travelling pulse along the main line
    pulse_path = f"M48 {y + 33}H700"
    out.append(
        f'<circle r="6" fill="{LIME}"><animateMotion dur="9s" repeatCount="indefinite" keyPoints="0;1;1" '
        f'keyTimes="0;0.38;1" calcMode="linear" path="{pulse_path}"/>'
        f'<animate attributeName="opacity" dur="9s" repeatCount="indefinite" values="1;1;0;0" keyTimes="0;0.38;0.4;1"/></circle>'
    )
    return frame(
        W,
        H,
        "The daily pipeline",
        "Every morning at 10:00 UK, Cloud Scheduler starts a workflow: ingest the FPL API, "
        "predict expected points, optimise each user's plan, then if the deadline is today "
        "notify sends the plan on Telegram; otherwise nothing is sent. Training runs weekly "
        "and only promotes a model that passes a quality gate. Data and models are in Cloud "
        "Storage; users are in Firestore.",
        "\n".join(out),
        a,
    )


def main() -> None:
    for name, draw in [("optimiser", optimiser), ("backtest", backtest), ("pipeline", pipeline)]:
        (OUT / f"{name}.svg").write_text(draw(), encoding="utf-8", newline="\n")
        print(f"wrote docs/images/{name}.svg")


if __name__ == "__main__":
    main()
