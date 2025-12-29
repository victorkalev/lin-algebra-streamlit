import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Linear Algebra Playground (2D)", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def vec_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def safe_angle_deg(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = vec_norm(a), vec_norm(b)
    if na == 0 or nb == 0:
        return None
    cos = float(np.dot(a, b) / (na * nb))
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))

def plot_vectors(a, b, title="Vectors"):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    s = a + b

    # choose limits
    pts = np.vstack([[[0, 0]], [a], [b], [s]])
    m = float(np.max(np.abs(pts))) if pts.size else 1.0
    lim = max(1.0, m * 1.2)

    fig, ax = plt.subplots()
    ax.axhline(0)
    ax.axvline(0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(title)
    ax.grid(True)

    # plot arrows
    ax.arrow(0, 0, a[0], a[1], length_includes_head=True, head_width=0.15, head_length=0.25)
    ax.text(a[0], a[1], "  a", va="center")

    ax.arrow(0, 0, b[0], b[1], length_includes_head=True, head_width=0.15, head_length=0.25)
    ax.text(b[0], b[1], "  b", va="center")

    ax.arrow(0, 0, s[0], s[1], length_includes_head=True, head_width=0.15, head_length=0.25)
    ax.text(s[0], s[1], "  a+b", va="center")

    return fig

def transform_grid(A: np.ndarray, n=11):
    # grid points
    xs = np.linspace(-1, 1, n)
    ys = np.linspace(-1, 1, n)

    # lines: x fixed, y fixed
    vertical = []
    for x in xs:
        pts = np.stack([np.full_like(ys, x), ys], axis=1)  # (n,2)
        vertical.append(pts)

    horizontal = []
    for y in ys:
        pts = np.stack([xs, np.full_like(xs, y)], axis=1)
        horizontal.append(pts)

    # transform
    Av = [ (A @ pts.T).T for pts in vertical ]
    Ah = [ (A @ pts.T).T for pts in horizontal ]
    return vertical, horizontal, Av, Ah

def plot_transform(A: np.ndarray):
    A = np.array(A, dtype=float)
    detA = float(np.linalg.det(A))

    v, h, Av, Ah = transform_grid(A, n=11)

    # bounds
    all_pts = np.vstack([np.vstack(v), np.vstack(h), np.vstack(Av), np.vstack(Ah)])
    m = float(np.max(np.abs(all_pts)))
    lim = max(1.0, m * 1.2)

    fig, ax = plt.subplots()
    ax.axhline(0)
    ax.axvline(0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True)
    ax.set_title(f"Grid transform (det(A) = {detA:.3f})")

    # original grid (thin)
    for pts in v:
        ax.plot(pts[:,0], pts[:,1], linewidth=1)
    for pts in h:
        ax.plot(pts[:,0], pts[:,1], linewidth=1)

    # transformed grid (thicker)
    for pts in Av:
        ax.plot(pts[:,0], pts[:,1], linewidth=2)
    for pts in Ah:
        ax.plot(pts[:,0], pts[:,1], linewidth=2)

    # basis vectors
    e1 = np.array([1,0], float)
    e2 = np.array([0,1], float)
    Ae1 = A @ e1
    Ae2 = A @ e2
    ax.arrow(0,0, e1[0], e1[1], length_includes_head=True, head_width=0.08, head_length=0.12)
    ax.arrow(0,0, e2[0], e2[1], length_includes_head=True, head_width=0.08, head_length=0.12)
    ax.arrow(0,0, Ae1[0], Ae1[1], length_includes_head=True, head_width=0.10, head_length=0.15)
    ax.arrow(0,0, Ae2[0], Ae2[1], length_includes_head=True, head_width=0.10, head_length=0.15)

    return fig

def plot_lines_and_solution(A, b):
    A = np.array(A, float)
    b = np.array(b, float).reshape(2)
    detA = float(np.linalg.det(A))

    fig, ax = plt.subplots()
    ax.axhline(0)
    ax.axvline(0)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

    # Lines: a11 x + a12 y = b1 and a21 x + a22 y = b2
    a11, a12 = A[0,0], A[0,1]
    a21, a22 = A[1,0], A[1,1]
    b1, b2 = b[0], b[1]

    xs = np.linspace(-10, 10, 400)

    def line_y(a1, a2, bb, xs):
        # a1 x + a2 y = bb => y = (bb - a1 x)/a2
        if abs(a2) < 1e-12:
            return None
        return (bb - a1*xs)/a2

    y1 = line_y(a11, a12, b1, xs)
    y2 = line_y(a21, a22, b2, xs)

    # plot lines (handle vertical lines)
    if y1 is not None:
        ax.plot(xs, y1)
    else:
        # a11 x = b1
        if abs(a11) > 1e-12:
            x0 = b1/a11
            ax.axvline(x0)
    if y2 is not None:
        ax.plot(xs, y2)
    else:
        if abs(a21) > 1e-12:
            x0 = b2/a21
            ax.axvline(x0)

    # Solve if possible
    x_sol = None
    status = ""
    if abs(detA) > 1e-10:
        x_sol = np.linalg.solve(A, b)
        ax.scatter([x_sol[0]], [x_sol[1]], s=60)
        status = f"Unique solution: x = ({x_sol[0]:.4f}, {x_sol[1]:.4f})"
    else:
        # singular
        status = "No unique solution (det(A) ≈ 0). Lines are parallel or identical."

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title(f"2x2 system geometry (det(A) = {detA:.3f})")

    return fig, status, x_sol

def least_squares_fit(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.column_stack([x, np.ones_like(x)])
    # Solve min ||Ax - y|| where x = [m, c]
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    m, c = sol
    return float(m), float(c)

def plot_ls(x, y, m, c):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.scatter(x, y)

    xs = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    ys = m*xs + c
    ax.plot(xs, ys)

    ax.set_title("Least Squares fit: y = m x + c")
    return fig

def eigen_2x2(A):
    w, V = np.linalg.eig(A)
    return w, V

def plot_eigen_iteration(A, v0, steps):
    A = np.array(A, float)
    v = np.array(v0, float)
    if vec_norm(v) == 0:
        v = np.array([1.0, 0.0])

    pts = [v.copy()]
    for _ in range(steps):
        v = A @ v
        if vec_norm(v) == 0:
            break
        v = v / vec_norm(v)  # normalize to focus on direction
        pts.append(v.copy())

    pts = np.array(pts)
    lim = 1.2

    fig, ax = plt.subplots()
    ax.axhline(0)
    ax.axvline(0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(True)
    ax.set_title("Iteration: v_{k+1} = normalize(A v_k)")

    # plot arrows for first and last
    ax.arrow(0,0, pts[0,0], pts[0,1], length_includes_head=True, head_width=0.06, head_length=0.10)
    ax.text(pts[0,0], pts[0,1], "  v0")

    ax.arrow(0,0, pts[-1,0], pts[-1,1], length_includes_head=True, head_width=0.06, head_length=0.10)
    ax.text(pts[-1,0], pts[-1,1], f"  v{len(pts)-1}")

    # show trail points
    ax.plot(pts[:,0], pts[:,1], marker="o")

    return fig, pts

# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("Linear Algebra (2D)")
page = st.sidebar.radio(
    "Choose a topic",
    [
        "Home",
        "Vectors",
        "Matrices as Transforms",
        "Linear Systems (2x2)",
        "Least Squares (Line Fit)",
        "Eigenvectors (2x2)",
    ],
)

# -----------------------------
# Pages
# -----------------------------
if page == "Home":
    st.title("Linear Algebra Playground (2D)")
    st.write(
        "This is a small interactive helper for first-year engineering math: vectors, matrices, systems, least squares, eigenvectors."
    )
    st.markdown(
        """
**How to use this app**
- Move sliders → watch the geometry change
- Read the 3–5 lines of theory
- Try the small tasks and check the numbers

**Exam mindset**
- Know what each object *means* (vector = arrow, matrix = transformation, Ax=b = intersection of constraints)
- Practice quick computations (dot product, determinant, solve 2×2)
"""
    )

elif page == "Vectors":
    st.title("Vectors: dot product, norm, angle")

    col1, col2 = st.columns([1, 2])

    with col1:
        ax = st.slider("a_x", -5.0, 5.0, 2.0, 0.1)
        ay = st.slider("a_y", -5.0, 5.0, 1.0, 0.1)
        bx = st.slider("b_x", -5.0, 5.0, 1.0, 0.1)
        by = st.slider("b_y", -5.0, 5.0, 2.0, 0.1)

        a = np.array([ax, ay])
        b = np.array([bx, by])

        st.markdown("### Key numbers")
        st.write(f"a · b = {float(np.dot(a,b)):.4f}")
        st.write(f"||a|| = {vec_norm(a):.4f}")
        st.write(f"||b|| = {vec_norm(b):.4f}")
        ang = safe_angle_deg(a, b)
        st.write("angle(a,b) = " + (f"{ang:.2f}°" if ang is not None else "undefined (zero vector)"))

        st.markdown("### Remember")
        st.latex(r"a\cdot b = \|a\|\|b\|\cos(\theta)")
        st.caption("Dot product links geometry (angle) and algebra (components).")

    with col2:
        fig = plot_vectors(a, b, title="a, b, and a+b")
        st.pyplot(fig)

elif page == "Matrices as Transforms":
    st.title("Matrices as 2D transformations")

    st.markdown(
        """
A 2×2 matrix acts like a machine: it takes an input vector **x** and outputs **Ax**.
Geometrically, it warps the whole plane: squares become parallelograms, areas scale by **det(A)**.
"""
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        a11 = st.slider("a11", -3.0, 3.0, 1.0, 0.1)
        a12 = st.slider("a12", -3.0, 3.0, 0.5, 0.1)
        a21 = st.slider("a21", -3.0, 3.0, 0.0, 0.1)
        a22 = st.slider("a22", -3.0, 3.0, 1.0, 0.1)
        A = np.array([[a11, a12], [a21, a22]], float)

        detA = float(np.linalg.det(A))
        st.markdown("### Key numbers")
        st.write(f"det(A) = {detA:.4f}")
        st.write("Invertible?" + (" Yes" if abs(detA) > 1e-10 else " No (or nearly)"))

        st.markdown("### Remember")
        st.latex(r"\det(A)\neq 0 \Rightarrow A^{-1}\ \text{exists}")
        st.caption("det(A) scales area. If det(A)=0, the plane collapses onto a line.")

    with col2:
        fig = plot_transform(A)
        st.pyplot(fig)

elif page == "Linear Systems (2x2)":
    st.title("Linear systems: solve Ax = b (2×2)")

    st.markdown(
        """
Each equation is a line in 2D. Solving **Ax=b** means finding the point where the lines intersect.
- One intersection → unique solution
- Parallel lines → no solution
- Same line → infinitely many solutions
"""
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Enter A and b")
        a11 = st.slider("a11 ", -5.0, 5.0, 2.0, 0.1)
        a12 = st.slider("a12 ", -5.0, 5.0, 1.0, 0.1)
        a21 = st.slider("a21 ", -5.0, 5.0, 1.0, 0.1)
        a22 = st.slider("a22 ", -5.0, 5.0, -1.0, 0.1)
        b1  = st.slider("b1", -10.0, 10.0, 3.0, 0.1)
        b2  = st.slider("b2", -10.0, 10.0, 1.0, 0.1)

        A = np.array([[a11, a12], [a21, a22]], float)
        b = np.array([b1, b2], float)

        st.markdown("### Remember")
        st.latex(r"Ax=b")
        st.caption("For 2×2, det(A) ≠ 0 usually means a unique intersection point.")

    with col2:
        fig, status, xsol = plot_lines_and_solution(A, b)
        st.pyplot(fig)
        st.info(status)

        if xsol is not None:
            st.markdown("### Quick check")
            check = A @ xsol
            st.write(f"A x = ({check[0]:.4f}, {check[1]:.4f})  vs  b = ({b[0]:.4f}, {b[1]:.4f})")

elif page == "Least Squares (Line Fit)":
    st.title("Least Squares: best-fit line y = m x + c")

    st.markdown(
        """
When data points don't lie perfectly on a line, least squares finds **m, c** that minimize total squared error.
This is the math behind simple calibration and measurement fitting.
"""
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        n = st.slider("Number of points", 5, 50, 15, 1)
        noise = st.slider("Noise level", 0.0, 5.0, 1.0, 0.1)
        true_m = st.slider("True m (for generated data)", -5.0, 5.0, 1.5, 0.1)
        true_c = st.slider("True c (for generated data)", -5.0, 5.0, -0.5, 0.1)
        seed = st.slider("Random seed", 0, 999, 7, 1)

        rng = np.random.default_rng(seed)
        x = np.linspace(-5, 5, n)
        y = true_m * x + true_c + rng.normal(0, noise, size=n)

        m, c = least_squares_fit(x, y)
        st.markdown("### Fit result")
        st.write(f"Estimated m = {m:.4f}")
        st.write(f"Estimated c = {c:.4f}")

        st.markdown("### Remember")
        st.latex(r"\min_{m,c}\sum_i (mx_i+c-y_i)^2")

    with col2:
        fig = plot_ls(x, y, m, c)
        st.pyplot(fig)

elif page == "Eigenvectors (2x2)":
    st.title("Eigenvectors (2×2): special directions")

    st.markdown(
        """
Eigenvectors are directions that **do not change direction** under a matrix A:
they only get scaled.
"""
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        a11 = st.slider("a11  ", -3.0, 3.0, 1.2, 0.1)
        a12 = st.slider("a12  ", -3.0, 3.0, 0.7, 0.1)
        a21 = st.slider("a21  ", -3.0, 3.0, 0.0, 0.1)
        a22 = st.slider("a22  ", -3.0, 3.0, 0.9, 0.1)
        A = np.array([[a11, a12], [a21, a22]], float)

        v0x = st.slider("v0_x", -1.0, 1.0, 0.6, 0.05)
        v0y = st.slider("v0_y", -1.0, 1.0, 0.2, 0.05)
        steps = st.slider("Iteration steps", 1, 30, 10, 1)

        w, V = eigen_2x2(A)
        st.markdown("### Eigenvalues")
        st.write(w)

        st.markdown("### Eigenvectors (columns)")
        st.write(V)

        st.markdown("### Remember")
        st.latex(r"A v = \lambda v")

    with col2:
        fig, trail = plot_eigen_iteration(A, (v0x, v0y), steps)
        st.pyplot(fig)
        st.caption("The trail shows how the direction evolves under repeated application of A (normalized each step).")
