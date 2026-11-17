# Genera las figuras del trabajo (opcional, recomendación 4 del enunciado).
# Uso:  python figuras.py   ->  crea fig1A.pdf, fig1Ap.pdf, ...  en la carpeta actual
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import special as sp

def save(name, xlabel, ylabel, ylim=None, legend=True):
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(alpha=.3)
    if ylim: plt.ylim(*ylim)
    if legend: plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(name + ".pdf"); plt.close()

# ---------- Cap. 1: Bessel ----------
z = np.linspace(0, 15, 800)
plt.figure()
for n in range(5): plt.plot(z, sp.jv(n, z), label=f"$J_{n}$")
plt.plot(sp.jn_zeros(0, 5), np.zeros(5), "ko", label="ceros de $J_0$")
save("fig1A", "$z$", "$J_n(z)$")

z = np.linspace(0.1, 15, 800)
plt.figure()
for n in range(5): plt.plot(z, sp.yv(n, z), label=f"$Y_{n}$")
save("fig1Ap", "$z$", "$Y_n(z)$", ylim=(-2, 1))

x = np.linspace(0, 1, 400)
plt.figure()
for n in range(1, 4): plt.plot(x, sp.jv(0, sp.jn_zeros(0, n)[-1] * x), label=f"$n={n}$")
save("fig1B", r"$\rho/a$", r"$J_0(\alpha_{0n}\rho/a)$")

z = np.linspace(0.1, 15, 800)
plt.figure()
plt.plot(z, sp.j0(z), label=r"Re $H_0^{(1)}=J_0$"); plt.plot(z, sp.y0(z), label=r"Im $H_0^{(1)}=Y_0$")
e = np.sqrt(2 / (np.pi * z)); plt.plot(z, e, "k--", lw=.8, label=r"$\pm\sqrt{2/\pi z}$"); plt.plot(z, -e, "k--", lw=.8)
save("fig1H", "$z$", "$H_0^{(1)}(z)$")

# ejemplo: modos de la membrana (0,1),(1,1),(0,2)
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
r, ph = np.meshgrid(np.linspace(0, 1, 120), np.linspace(0, 2 * np.pi, 200))
X, Y = r * np.cos(ph), r * np.sin(ph)
for ax, (m, n) in zip(axs, [(0, 1), (1, 1), (0, 2)]):
    u = sp.jv(m, sp.jn_zeros(m, n)[-1] * r) * np.cos(m * ph)
    ax.contourf(X, Y, u, 21, cmap="RdBu"); ax.set_aspect("equal"); ax.set_title(f"({m},{n})"); ax.axis("off")
plt.tight_layout(); plt.savefig("fig1E.pdf"); plt.close()

# ---------- Cap. 2: Legendre ----------
x = np.linspace(-1, 1, 500)
plt.figure()
for n in range(5): plt.plot(x, sp.eval_legendre(n, x), label=f"$P_{n}$")
save("fig2A", "$x$", "$P_n(x)$")

th = np.linspace(0, np.pi, 400)
Y2 = {"00": lambda t: 1 / (4 * np.pi) + 0 * t,
      "10": lambda t: 3 / (4 * np.pi) * np.cos(t) ** 2,
      "11": lambda t: 3 / (8 * np.pi) * np.sin(t) ** 2,
      "20": lambda t: 5 / (16 * np.pi) * (3 * np.cos(t) ** 2 - 1) ** 2,
      "21": lambda t: 15 / (8 * np.pi) * np.sin(t) ** 2 * np.cos(t) ** 2,
      "22": lambda t: 15 / (32 * np.pi) * np.sin(t) ** 4}
fig, axs = plt.subplots(2, 3, figsize=(9, 6))
for ax, (k, f) in zip(axs.flat, Y2.items()):
    rr = f(th)
    for s in (1, -1): ax.plot(s * rr * np.sin(th), rr * np.cos(th), "b")
    ax.set_aspect("equal"); ax.set_title(f"$|Y_{k[0]}^{k[1]}|^2$"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("fig2B.pdf"); plt.close()

plt.figure(); c = np.linspace(-1, 1, 200)
plt.plot(c, 3 * c, label=r"$\sigma/\varepsilon_0E_0=3\cos\theta$")
save("fig2E", r"$\cos\theta$", r"$\sigma/(\varepsilon_0E_0)$")

# ---------- Cap. 3: Hermite ----------
x = np.linspace(-3, 3, 600)
plt.figure()
for n in range(5): plt.plot(x, sp.eval_hermite(n, x), label=f"$H_{n}$")
save("fig3A", "$x$", "$H_n(x)$", ylim=(-60, 60))

def psi(n, x): return np.exp(-x**2 / 2) * sp.eval_hermite(n, x) / np.sqrt(2.0**n * sp.factorial(n) * np.sqrt(np.pi))
x = np.linspace(-6, 6, 800)
plt.figure(); plt.plot(x, x**2 / 2, "k", lw=1)
for n in range(5):
    plt.axhline(n + .5, color="gray", lw=.4); plt.plot(x, psi(n, x) + n + .5, label=fr"$\psi_{n}$")
save("fig3B", r"$x$ (en unidades de $\sqrt{\hbar/m\omega}$)", r"$E/\hbar\omega,\ \psi_n$", ylim=(0, 6))

plt.figure(); xx = np.linspace(-6, 6, 800)
for n in (0, 1, 5): plt.plot(xx, psi(n, xx) ** 2, label=fr"$|\psi_{n}|^2$")
cl = np.where(np.abs(xx) < np.sqrt(11), 1 / (np.pi * np.sqrt(np.maximum(11 - xx**2, 1e-9))), np.nan)
plt.plot(xx, cl, "k--", lw=.8, label="clásica $n=5$")
save("fig3E", "$x$", r"$|\psi_n|^2$", ylim=(0, 1))

# ---------- Cap. 4: Laguerre ----------
x = np.linspace(0, 10, 600)
plt.figure()
for n in range(5): plt.plot(x, sp.eval_laguerre(n, x), label=f"$L_{n}$")
save("fig4A", "$x$", "$L_n(x)$", ylim=(-3, 3))

R = {"1s": lambda r: 2 * np.exp(-r),
     "2s": lambda r: (1 / np.sqrt(2)) * (1 - r / 2) * np.exp(-r / 2),
     "2p": lambda r: r * np.exp(-r / 2) / (2 * np.sqrt(6)),
     "3s": lambda r: 2 / (3 * np.sqrt(3)) * (1 - 2 * r / 3 + 2 * r**2 / 27) * np.exp(-r / 3),
     "3p": lambda r: 8 / (27 * np.sqrt(6)) * (1 - r / 6) * r * np.exp(-r / 3)}
r = np.linspace(0, 25, 800)
plt.figure()
for k, f in R.items(): plt.plot(r, r**2 * f(r) ** 2, label=k)
save("fig4B", "$r/a_0$", r"$r^2|R_{nl}|^2\ (a_0^{-1})$")
plt.figure(); r = np.linspace(0, 15, 600)
plt.plot(r, R["2s"](r), label="$R_{20}$"); plt.plot(r, R["2p"](r), label="$R_{21}$"); plt.axhline(0, color="k", lw=.5)
save("fig4E", "$r/a_0$", r"$R_{nl}\ (a_0^{-3/2})$")

# ---------- Cap. 5: Chebyshev ----------
x = np.linspace(-1, 1, 600)
fig, axs = plt.subplots(2, 1, figsize=(6, 7))
for n in range(5):
    axs[0].plot(x, sp.eval_chebyt(n, x), label=f"$T_{n}$"); axs[1].plot(x, sp.eval_chebyu(n, x), label=f"$U_{n}$")
for a, l in zip(axs, ("$T_n(x)$", "$U_n(x)$")): a.set_xlabel("$x$"); a.set_ylabel(l); a.grid(alpha=.3); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig5A.pdf"); plt.close()

plt.figure(); plt.plot(x, np.sqrt(1 - x**2), "k", label=r"$\sqrt{1-x^2}$")
for N in (0, 1, 3):
    s = 2 / np.pi + 0 * x
    for k in range(1, N + 1):
        s = s - (4 / np.pi) * sp.eval_chebyt(2 * k, x) / (4 * k**2 - 1)
    plt.plot(x, s, label=f"hasta $T_{{{2*N}}}$")
save("fig5E", "$x$", r"suma parcial")

# ---------- Cap. 6: Mathieu (scipy usa grados) ----------
eta = np.linspace(0, np.pi, 400); deg = np.degrees(eta)
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for n in range(5):
    for q, ls in ((0, "-"), (0.5, "--")):
        axs[0].plot(eta, sp.mathieu_cem(n, q, deg)[0], ls, label=f"ce{n}, q={q}" if q == 0 else None, color=f"C{n}")
for n in range(1, 5):
    for q, ls in ((0, "-"), (0.5, "--")):
        axs[1].plot(eta, sp.mathieu_sem(n, q, deg)[0], ls, label=f"se{n}, q={q}" if q == 0 else None, color=f"C{n}")
for a in axs: a.set_xlabel(r"$\eta$"); a.grid(alpha=.3); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig6A.pdf"); plt.close()

q = np.linspace(0, 3, 300)
plt.figure(figsize=(6, 5))
for n in range(3):
    lo = sp.mathieu_a(n, q); hi = sp.mathieu_b(n + 1, q)
    plt.fill_between(q, lo, hi, alpha=.25, color="C2")
for n in range(4):
    plt.plot(q, sp.mathieu_a(n, q), "b", lw=1)
    if n >= 1: plt.plot(q, sp.mathieu_b(n, q), "r", lw=1)
plt.plot([], [], "b", label="$a_n(q)$"); plt.plot([], [], "r", label="$b_n(q)$")
save("fig6B", "$q$", "$a$", ylim=(-4, 12))

# trampa de Paul: región de estabilidad simultánea en x e y
q = np.linspace(0, 1, 400)
a0, b1 = sp.mathieu_a(0, q), sp.mathieu_b(1, q)
upper = np.minimum(b1, -a0); lower = -upper
mask = upper > 0
plt.figure()
plt.fill_between(q[mask], lower[mask], upper[mask], alpha=.3, label="región estable")
for y, l in ((b1, r"$\pm b_1$"), (a0, r"$\pm a_0$")):
    plt.plot(q, y, "k", lw=.8); plt.plot(q, -y, "k", lw=.8)
plt.plot([0.706], [0.237], "ro"); plt.plot([0.908], [0], "ro")
save("fig6E", "$q$", "$a$", ylim=(-0.4, 0.4))
print("Figuras generadas.")
