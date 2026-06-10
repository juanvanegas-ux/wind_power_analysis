"""Rotor thrust and tower loads for the small turbines, using the *thrust*
coefficient Ct(lambda) from the BEM solver.

small_wind.py used the power coefficient Cp to work out energy. but the same BEM
run also exports the thrust coefficient Ct, which had been sitting unused, and Ct
is what sizes the structure, the tower and the foundation have to carry the
aerodynamic push on the rotor, not the power. so this script turns the Ct curves
into actual forces and moments.

The thrust on the rotor is

    F = 0.5 * rho * A * Ct * V^2

(note V^2, not V^3 like power). multiply that by the hub height and you get the
overturning moment at the tower base, which is the number a foundation is
designed around.

The interesting result, and the reason this is worth doing, is what it says about
the two blades. the smart blade is bend twist coupled, the whole point of that
design is that the blade twists under load to shed thrust. and you can see it in
the numbers: at its operating point the smart blade sits at a *lower* Ct than the
comercial one, so it carries less structural load per unit of power it makes
(a lower Ct/Cp ratio). it wins on loads and on energy at the same time, which is
exactly the case the BEM study was trying to make, now in force and moment terms.

Two control strategies again, matching small_wind.py:
  * variable speed (MPPT): the rotor sits at lambda_opt, so Ct is roughly constant
    at Ct(lambda_opt) and thrust grows with V^2 until the machine furls at rated
    and sheds load
  * fixed speed: lambda slides with the wind, so Ct slides along its curve too

Outputs (saved to ../results):
  - swt_thrust_curves.png  rotor thrust vs wind, both blades, both strategies
  - swt_tower_loads.png    base overturning moment vs hub height (cost of height)

Run:  python src/loads.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import config
import small_wind as sw


def ct_column(cp_col):
    """Map a Cp column name to its Ct partner, e.g. Cp_Smart -> Ct_Smart."""
    return cp_col.replace("Cp", "Ct")


def ct_at(cp, lam, col, lam_query):
    """Thrust coefficient at a given tip speed ratio (linear interp)."""
    return float(np.interp(lam_query, lam, cp[col].values))


def thrust(V, R, ct, rho=config.AIR_DENSITY):
    """Rotor thrust [N] for a given thrust coefficient: 0.5*rho*A*Ct*V^2."""
    return 0.5 * rho * sw.swept_area(R) * ct * np.asarray(V, dtype=float) ** 2


def thrust_mppt(V, R, ct_op, rho=config.AIR_DENSITY):
    """Variable speed thrust [N]. the rotor holds lambda_opt so Ct is constant,
    thrust grows as V^2 up to the rated wind, then the turbine furls / regulates
    and we cap the thrust at its rated value (a standard simplification, real
    furling can even pull it down a bit)."""
    V = np.asarray(V, dtype=float)
    f = thrust(V, R, ct_op, rho)
    f_rated = thrust(sw.V_RATED, R, ct_op, rho)
    f = np.minimum(f, f_rated)                       # furl: cap the load
    f[(V < sw.CUT_IN) | (V > sw.CUT_OUT)] = 0.0
    return f


def thrust_fixed(V, R, cp, lam, ct_col, rpm=sw.FIXED_RPM, rho=config.AIR_DENSITY):
    """Fixed speed thrust [N]. constant rpm, so lambda = omega*R/V slides and Ct
    slides along its curve with it."""
    V = np.asarray(V, dtype=float)
    omega = rpm * 2.0 * np.pi / 60.0
    with np.errstate(divide="ignore"):
        lam_local = omega * R / V
    # clamp to the measured lambda range (np.interp default holds the end values)
    ct_local = np.interp(lam_local, lam, cp[ct_col].values)
    f = 0.5 * rho * sw.swept_area(R) * ct_local * V ** 2
    f[(V < sw.CUT_IN) | (V > sw.CUT_OUT)] = 0.0
    return f


def overturning_moment(thrust_n, hub_height):
    """Base overturning moment [N m] = thrust * hub height. the load the
    foundation is designed to resist."""
    return thrust_n * hub_height


def analyse(cp):
    lam = cp["lambda"].values
    print("=" * 80)
    print("Rotor thrust and tower loads (from the BEM Ct curves)")
    print("=" * 80)
    print(f"{'blade':<18}{'Cp_op':>7}{'Ct_op':>7}{'Ct/Cp':>7}"
          f"{'F rated kN':>12}{'base moment kNm':>17}")
    print("-" * 80)

    results = {}
    for name, t in sw.TURBINES.items():
        cmax, lopt = sw.cp_peak(cp, lam, t["col"])
        ct_op = ct_at(cp, lam, ct_column(t["col"]), lopt)
        f_rated = thrust(sw.V_RATED, t["R"], ct_op)        # N at rated, MPPT
        moment = overturning_moment(f_rated, sw.HUB_HEIGHT)
        results[name] = dict(cmax=cmax, lopt=lopt, ct_op=ct_op,
                             ct_cp=ct_op / cmax, f_rated=f_rated, moment=moment)
        print(f"{name:<18}{cmax:>7.3f}{ct_op:>7.3f}{ct_op/cmax:>7.3f}"
              f"{f_rated/1e3:>12.2f}{moment/1e3:>17.1f}")

    print("-" * 80)
    s = results["Smart blade"]["ct_cp"]
    c = results["Comercial blade"]["ct_cp"]
    print(f"the comercial blade carries {(c/s - 1)*100:.0f}% more thrust per unit "
          f"of power (Ct/Cp) than the\nbend twist coupled smart blade, so it needs"
          f" a heavier structure for less energy.")
    return results


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------

def plot_thrust_curves(cp):
    lam = cp["lambda"].values
    V = np.linspace(0, sw.CUT_OUT + 1, 200)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, t in sw.TURBINES.items():
        _, lopt = sw.cp_peak(cp, lam, t["col"])
        ct_op = ct_at(cp, lam, ct_column(t["col"]), lopt)
        ax.plot(V, thrust_mppt(V, t["R"], ct_op) / 1e3, color=t["color"], lw=2,
                label=f"{name}, variable speed (Ct {ct_op:.2f})")
        ax.plot(V, thrust_fixed(V, t["R"], cp, lam, ct_column(t["col"])) / 1e3,
                color=t["color"], lw=1.5, ls="--",
                label=f"{name}, fixed {sw.FIXED_RPM:.0f} rpm")
    ax.axvline(sw.V_RATED, color="black", ls=":", lw=1)
    ax.text(sw.V_RATED + 0.1, ax.get_ylim()[1] * 0.05, "rated / furl",
            rotation=90, fontsize=8)
    ax.set_xlabel("Wind speed at hub  [m/s]")
    ax.set_ylabel("Rotor thrust  [kN]")
    ax.set_title("Rotor thrust from the Ct(lambda) data")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "swt_thrust_curves.png"), dpi=130)


def plot_tower_loads(cp):
    """Base overturning moment vs hub height. the thrust at rated is fixed by the
    rotor, but the moment is thrust times height, so it grows straight line with
    the tower, this is the structural cost that sits against the energy gain of a
    taller tower in small_wind.py."""
    lam = cp["lambda"].values
    heights = np.arange(10, 41, 2.0)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, t in sw.TURBINES.items():
        _, lopt = sw.cp_peak(cp, lam, t["col"])
        ct_op = ct_at(cp, lam, ct_column(t["col"]), lopt)
        f_rated = thrust(sw.V_RATED, t["R"], ct_op)
        moments = overturning_moment(f_rated, heights) / 1e3   # kN m
        ax.plot(heights, moments, marker="o", ms=3, color=t["color"],
                label=f"{name}")
    ax.axvline(sw.HUB_HEIGHT, color="grey", ls=":", lw=1)
    ax.text(sw.HUB_HEIGHT + 0.3, ax.get_ylim()[0], "baseline 24 m",
            rotation=90, va="bottom", fontsize=8)
    ax.set_xlabel("Hub height  [m]")
    ax.set_ylabel("Base overturning moment at rated  [kN m]")
    ax.set_title("The structural cost of a taller tower (foundation sizing)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.RESULTS, "swt_tower_loads.png"), dpi=130)


def main():
    os.makedirs(config.RESULTS, exist_ok=True)
    cp = sw.load_cp()
    analyse(cp)
    plot_thrust_curves(cp)
    plot_tower_loads(cp)
    print(f"Saved figures to {os.path.abspath(config.RESULTS)}")


if __name__ == "__main__":
    main()
