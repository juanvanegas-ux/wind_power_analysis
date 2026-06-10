"""Jensen (Park) wake model for a wind farm at La Guajira.

energy_yield.py knocks 8% off the energy for wakes (turbines stealing wind from
each other), but that 8% was an assumption. this script computes it instead, from
the geometry of a turbine layout, the thrust coefficient, and the actual measured
wind (speed and direction) hour by hour.

The Jensen model is the standard first cut. behind a turbine the wake expands
linearly with distance and the wind speed deficit shrinks as it recovers:

    deficit on the centreline at distance x:
        delta0 = (1 - sqrt(1 - Ct)) * (R / (R + k*x))^2

    R is the rotor radius, k is the wake decay rate (~0.075 onshore), Ct is the
    thrust coefficient. a downstream rotor that only partly sits in the wake feels
    delta0 scaled by the overlap area fraction, and several wakes hitting one
    turbine combine as the square root of the sum of squares.

Ct is the link back to the BEM work. the BEM rotors operate around Ct 0.76 to
0.83 (see loads.py), and utility turbines sit in the same range below rated, so a
representative Ct of 0.8 is used here, the wake physics is the same whatever the
rotor size.

The thing this makes obvious is *direction*. La Guajira wind is overwhelmingly
from the east north east, so the layout should lean into that, build the farm wide
and shallow (lots of turbines side by side across the wind, only a few rows deep
along it) so that few turbines ever sit in another one's wake. for the same 16
turbines the computed loss swings from about 9% (wide and shallow) to about 15%
(deep and narrow), purely from the shape of the layout. that swing is the wind
rose turned into money.

A heads up on the headline number: the assumed 8% turns out to be optimistic for a
compact array here. a naive 4x4 square at 7D spacing computes to about 15%, and you
only get down near 8% with generous spacing or the wide shallow layout. the reason
is that this resource spends most of its hours *below* rated (mean 9.3 m/s against
a 12 m/s rated), and wakes only cost you energy below rated, above rated the
turbine is capped anyway. a site that ran mostly above rated would shrug wakes off.

Outputs (saved to ../results):
  - wake_recovery.png   how a single wake recovers with distance (a few Ct)
  - wake_array.png      the layout, and array efficiency vs wind direction + rose
  - wake_spacing.png    computed wake loss vs row spacing (the land vs wake trade)

Run:  python src/wake.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import config
import energy_yield

# --- turbine / wake assumptions ----------------------------------------
D = 90.0            # rotor diameter of a ~2 MW utility turbine [m] (R = 45 m)
R = D / 2.0
K_WAKE = 0.075      # wake decay constant, standard onshore value
CT_OP = 0.80        # representative operating thrust coeff (BEM rotors + utility)

# the baseline 2 MW power curve from energy_yield, so the wake loss is computed
# on the same machine the rest of the repo uses
TURBINE = energy_yield.TURBINES["2.0 MW (baseline)"]

# --- farm layout (a grid, oriented relative to the dominant wind) -------
# default is the sensible wide/shallow layout for a unidirectional wind: only a
# couple of rows deep along the wind, many turbines spread across it.
DOMINANT_FROM_DEG = 70.0   # ENE, where the La Guajira wind comes from
N_ALONG = 2                # rows along (downwind of) the dominant wind
N_ACROSS = 8               # columns across the dominant wind
S_ALONG = 7.0              # downwind row spacing [rotor diameters]
S_ACROSS = 3.0             # crosswind column spacing [rotor diameters]


def _bearing_vector(deg):
    """Unit vector (east, north) pointing along a compass bearing."""
    r = np.radians(deg)
    return np.array([np.sin(r), np.cos(r)])


def layout(n_along=N_ALONG, n_across=N_ACROSS, s_along=S_ALONG,
           s_across=S_ACROSS, dominant_from=DOMINANT_FROM_DEG):
    """Turbine positions [m] in east/north, grid aligned to the dominant wind.

    the 'along' axis points the way the dominant wind blows (from + 180), the
    'across' axis is perpendicular. so rows are stacked downwind and columns sit
    side by side across the wind, which is the sensible orientation here.
    """
    along = _bearing_vector(dominant_from + 180.0)     # wind blows toward this
    across = _bearing_vector(dominant_from + 180.0 - 90.0)
    pos = []
    for i in range(n_along):
        for j in range(n_across):
            p = (i * s_along * D) * along \
                + (j - (n_across - 1) / 2.0) * s_across * D * across
            pos.append(p)
    return np.array(pos)


def _circle_overlap(r1, r2, d):
    """Area where two circles (radii r1, r2, centre distance d) overlap."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return np.pi * min(r1, r2) ** 2
    a1 = r1 ** 2 * np.arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
    a2 = r2 ** 2 * np.arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
    a3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) *
                       (d - r1 + r2) * (d + r1 + r2))
    return a1 + a2 - a3


def pair_deficit(x_down, y_cross, ct=CT_OP, k=K_WAKE):
    """Speed deficit one upstream turbine causes at a downstream one, given the
    downwind distance and the crosswind offset between them."""
    if x_down <= 0:
        return 0.0
    rw = R + k * x_down                          # wake radius here
    delta0 = (1 - np.sqrt(1 - ct)) * (R / rw) ** 2
    frac = _circle_overlap(rw, R, abs(y_cross)) / (np.pi * R ** 2)
    return delta0 * frac


def deficit_by_direction(pos, ct=CT_OP, k=K_WAKE):
    """For every wind direction (1 deg bins), the combined speed deficit at each
    turbine. returns an array shape (360, n_turbines)."""
    n = len(pos)
    out = np.zeros((360, n))
    for theta in range(360):
        blow = _bearing_vector(theta + 180.0)               # wind blows toward
        perp = _bearing_vector(theta + 180.0 - 90.0)
        for w in range(n):
            sq = 0.0
            for u in range(n):
                if u == w:
                    continue
                rel = pos[w] - pos[u]
                x_down = float(rel @ blow)                   # >0 if w downwind
                y_cross = float(rel @ perp)
                d = pair_deficit(x_down, y_cross, ct, k)
                sq += d * d
            out[theta, w] = min(np.sqrt(sq), 0.95)
        # safety, deficits combine by sum of squares
    return out


def array_efficiency_by_direction(deficit):
    """Below rated, power goes with V^3, so the array efficiency at a direction is
    the mean of (1 - deficit)^3 over the turbines. this is the clean direction
    only view (it ignores that above rated wakes cost nothing)."""
    return np.mean((1.0 - deficit) ** 3, axis=1)


def computed_wake_loss(deficit, df):
    """The honest number: run the real measured (speed, direction) record through
    the layout and the actual power curve, and compare the array's energy to the
    same turbines with no wakes. returns the wake loss fraction."""
    v = df["wind_speed_100m_ms"].values
    dirs = df["wind_direction_100m_deg"].values % 360
    bins = np.clip(np.round(dirs).astype(int), 0, 359)

    cut_in, rated, cut_out, rated_power = TURBINE
    free = energy_yield.power_curve(v, *TURBINE)            # no wake, per turbine
    n = deficit.shape[1]

    waked_total = np.zeros_like(v)
    for w in range(n):
        vw = v * (1.0 - deficit[bins, w])
        waked_total += energy_yield.power_curve(vw, *TURBINE)

    free_total = free * n
    return 1.0 - waked_total.sum() / free_total.sum()


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------

def plot_recovery():
    x_over_d = np.linspace(0.1, 12, 200)
    x = x_over_d * D
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for ct, color in [(0.6, "grey"), (CT_OP, config.ACCENT),
                      (0.9, config.HIGHLIGHT)]:
        delta0 = (1 - np.sqrt(1 - ct)) * (R / (R + K_WAKE * x)) ** 2
        lbl = f"Ct = {ct:.2f}" + (" (operating)" if ct == CT_OP else "")
        ax.plot(x_over_d, delta0 * 100, color=color, lw=2, label=lbl)
    for s in (S_ALONG,):
        ax.axvline(s, color="black", ls=":", lw=1)
        ax.text(s + 0.1, ax.get_ylim()[1] * 0.9,
                f"{s:.0f}D row spacing", rotation=90, va="top", fontsize=8)
    ax.set_xlabel("Downwind distance  [rotor diameters]")
    ax.set_ylabel("Centreline speed deficit  [%]")
    ax.set_title("How a single wake recovers (Jensen, k = 0.075)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "wake_recovery.png"), dpi=130)


def plot_array(pos, eff_dir, df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

    # ---- layout ----
    ax = axes[0]
    ax.scatter(pos[:, 0], pos[:, 1], s=60, color=config.ACCENT, zorder=3)
    blow = _bearing_vector(DOMINANT_FROM_DEG + 180.0)
    span = np.ptp(pos[:, 0]) + np.ptp(pos[:, 1])
    c = pos.mean(axis=0)
    ax.annotate("", xy=c + blow * span * 0.28, xytext=c - blow * span * 0.28,
                arrowprops=dict(arrowstyle="->", color=config.HIGHLIGHT, lw=2))
    ax.text(*(c - blow * span * 0.30), "dominant\nwind (ENE)",
            color=config.HIGHLIGHT, fontsize=9, ha="center")
    ax.set_aspect("equal")
    ax.set_xlabel("East  [m]")
    ax.set_ylabel("North  [m]")
    ax.set_title(f"Layout, {N_ALONG}x{N_ACROSS} turbines "
                 f"({S_ACROSS:.0f}D across, {S_ALONG:.0f}D along)")
    ax.grid(alpha=0.3)

    # ---- efficiency vs direction, with the rose ----
    ax = axes[1]
    deg = np.arange(360)
    ax.plot(deg, eff_dir * 100, color=config.ACCENT, lw=2,
            label="array efficiency")
    ax.set_xlabel("Wind direction (from)  [deg]")
    ax.set_ylabel("Array efficiency  [%]", color=config.ACCENT)
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 45))
    ax2 = ax.twinx()
    hrs = df["wind_direction_100m_deg"].values % 360
    ax2.hist(hrs, bins=np.arange(0, 361, 10), color=config.HIGHLIGHT,
             alpha=0.30, label="hours (rose)")
    ax2.set_ylabel("Hours per 10 deg bin", color=config.HIGHLIGHT)
    ax.set_title("Where the wind blows vs where the wakes bite")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "wake_array.png"), dpi=130)


def plot_spacing(df):
    """Computed wake loss as the downwind row spacing changes, for a 4x4 grid
    (enough rows deep for spacing to matter). tighter spacing is cheaper (less
    land and cable) but more wake, this is that trade off."""
    spacings = np.arange(3, 11, 1.0)
    losses = []
    for s in spacings:
        pos = layout(n_along=4, n_across=4, s_along=s)
        d = deficit_by_direction(pos)
        losses.append(computed_wake_loss(d, df) * 100)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(spacings, losses, marker="o", color=config.ACCENT, lw=2,
            label="computed (4x4 grid)")
    ax.axhline(8.0, color=config.HIGHLIGHT, ls="--", lw=1.5,
               label="the 8% assumed in energy_yield.py")
    ax.set_xlabel("Downwind row spacing  [rotor diameters]")
    ax.set_ylabel("Computed wake loss  [%]")
    ax.set_title("Wake loss vs spacing (land/cable vs lost energy)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "wake_spacing.png"), dpi=130)


def summary(df):
    # the recommended layout (wide and shallow), and two foils on the same 16
    # turbines: a naive square, and a deep narrow farm stacked into the wind
    pos = layout()                                          # 2 deep, 8 across
    loss_shallow = computed_wake_loss(deficit_by_direction(pos), df)

    pos_sq = layout(n_along=4, n_across=4)
    loss_square = computed_wake_loss(deficit_by_direction(pos_sq), df)

    pos_deep = layout(n_along=8, n_across=2)
    loss_deep = computed_wake_loss(deficit_by_direction(pos_deep), df)

    print("=" * 72)
    print("Computed wake loss (Jensen) vs the 8% assumed in energy_yield.py")
    print("=" * 72)
    print(f"16 x 2 MW turbines, D = {D:.0f} m, Ct = {CT_OP}, k = {K_WAKE}, "
          f"{S_ALONG:.0f}D along / {S_ACROSS:.0f}D across")
    print("-" * 72)
    print(f"wide & shallow  ({N_ALONG} deep x {N_ACROSS} across, recommended) "
          f": {loss_shallow:>5.1%}")
    print(f"square          (4 x 4)                            "
          f": {loss_square:>5.1%}")
    print(f"deep & narrow   (8 deep x 2 across)                "
          f": {loss_deep:>5.1%}")
    print(f"energy_yield.py assumption                         :  8.0%")
    print("-" * 72)
    print("same 16 turbines, the layout alone moves the loss from ~9% to ~15%.")
    print("the 8% rule of thumb is optimistic for a compact array, you reach it")
    print("only by going wide and shallow or by spacing the rows out (see")
    print("wake_spacing.png). leaning into the one wind direction is real money.")
    return pos, deficit_by_direction(pos)


def main():
    os.makedirs(config.RESULTS, exist_ok=True)
    df = energy_yield.load()
    pos, deficit = summary(df)
    eff_dir = array_efficiency_by_direction(deficit)
    plot_recovery()
    plot_array(pos, eff_dir, df)
    plot_spacing(df)
    print(f"Saved figures to {os.path.abspath(config.RESULTS)}")


if __name__ == "__main__":
    main()
