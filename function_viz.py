#!/usr/bin/env python3
"""
Interactive f(x, y) Visualization with Tangent Plane.

A desktop application that renders a 3D surface from a user-entered f(x, y)
expression. Users can select a point and visualise the tangent plane at that
point, along with its equation. Includes a contour plot with gradient vector
and rotation sliders for the 3D view.

Dependencies: numpy, matplotlib, sympy (+ tkinter, bundled with Python).
"""

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3D projection


# ---------------------------------------------------------------------------
# Colour palette / styling constants
# ---------------------------------------------------------------------------
BG = "#1e1e2e"
BG_SECONDARY = "#282a3a"
FG = "#cdd6f4"
ACCENT = "#89b4fa"
ACCENT2 = "#f38ba8"
ENTRY_BG = "#313244"
ENTRY_FG = "#cdd6f4"
BTN_BG = "#585b70"
BTN_FG = "#cdd6f4"
FONT = ("Segoe UI", 11)
FONT_BOLD = ("Segoe UI", 11, "bold")
FONT_TITLE = ("Segoe UI", 14, "bold")
FONT_MONO = ("Cascadia Code", 11)
FONT_EQ = ("Cascadia Code", 10)

COLORMAPS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "coolwarm",
    "RdYlBu",
    "Spectral",
    "twilight",
    "turbo",
]


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
x_sym, y_sym = sp.symbols("x y")

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def parse_function(expr_str: str):
    """Parse a string into a sympy expression of x, y."""
    expr = parse_expr(
        expr_str,
        local_dict={"x": x_sym, "y": y_sym, "e": sp.E, "pi": sp.pi},
        transformations=TRANSFORMATIONS,
    )
    return expr


def build_evaluator(expr):
    """Return a numpy-ready callable f(x, y) from a sympy expression."""
    return sp.lambdify((x_sym, y_sym), expr, modules=["numpy"])


def tangent_plane_expr(expr, x0_val, y0_val):
    """
    Return (z0, fx0, fy0, plane_expr) for the tangent plane at (x0, y0).

    Tangent plane:  z = z0 + fx0*(x - x0) + fy0*(y - y0)
    """
    fx = sp.diff(expr, x_sym)
    fy = sp.diff(expr, y_sym)

    z0 = float(expr.subs([(x_sym, x0_val), (y_sym, y0_val)]))
    fx0 = float(fx.subs([(x_sym, x0_val), (y_sym, y0_val)]))
    fy0 = float(fy.subs([(x_sym, x0_val), (y_sym, y0_val)]))

    plane = z0 + fx0 * (x_sym - x0_val) + fy0 * (y_sym - y0_val)
    return z0, fx0, fy0, plane


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
class FunctionVizApp:
    """Main application class."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("f(x, y) — Interactive Surface & Tangent Plane Visualiser")
        self.root.configure(bg=BG)
        self.root.minsize(1350, 750)

        # State
        self.expr = None  # current sympy expression
        self.f_np = None  # numpy evaluator
        self.surface_artist = None
        self.tangent_artist = None
        self.point_artist = None
        self.show_surface = tk.BooleanVar(value=True)
        self.show_tangent = tk.BooleanVar(value=True)
        self.show_point = tk.BooleanVar(value=True)
        self.colorbar = None  # contour colorbar handle

        self._build_ui()
        self._initial_plot()

    def _build_ui(self):
        # ── Main Resizable Split (PanedWindow) ──
        style = ttk.Style()
        style.configure("TPanedwindow", background=BG)
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # ── Left pane: Scrollable Container ──
        self.left_container = tk.Frame(self.paned, bg=BG)
        self.paned.add(self.left_container, weight=0)

        self.canvas_left = tk.Canvas(self.left_container, bg=BG, highlightthickness=0)
        self.scrollbar_y = ttk.Scrollbar(
            self.left_container, orient=tk.VERTICAL, command=self.canvas_left.yview
        )
        self.canvas_left.configure(yscrollcommand=self.scrollbar_y.set)

        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame inside canvas for the actual widgets
        self.left_frame = tk.Frame(self.canvas_left, bg=BG)
        self.canvas_window = self.canvas_left.create_window(
            (0, 0), window=self.left_frame, anchor="nw"
        )

        # ── Right pane: Canvas for Plots ──
        self.right_frame = tk.Frame(self.paned, bg=BG_SECONDARY)
        self.paned.add(self.right_frame, weight=1)

        self._build_left_panel()
        self._build_canvas()

        # ── Event Bindings for Scrolling & Resizing ──
        self.left_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas_left.bind("<Configure>", self._on_canvas_configure)
        self._bind_mousewheel(self.canvas_left)
        # Also bind to the frame to ensure scrolling works when hovering over widgets
        self._bind_mousewheel(self.left_frame)

    def _build_left_panel(self):
        pad = dict(padx=12, pady=(6, 2))
        padw = dict(padx=12, pady=2)

        # ── Title ──
        tk.Label(
            self.left_frame,
            text="📐 Function Visualiser",
            font=FONT_TITLE,
            fg=ACCENT,
            bg=BG,
            anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(14, 8))

        sep = ttk.Separator(self.left_frame, orient="horizontal")
        sep.pack(fill=tk.X, padx=12, pady=4)

        # ── Function input ──
        tk.Label(
            self.left_frame, text="f(x, y) =", font=FONT_BOLD, fg=FG, bg=BG, anchor="w"
        ).pack(fill=tk.X, **pad)

        self.func_entry = tk.Entry(
            self.left_frame,
            font=FONT_MONO,
            bg=ENTRY_BG,
            fg=ENTRY_FG,
            insertbackground=FG,
            relief="flat",
            bd=0,
            highlightthickness=2,
            highlightcolor=ACCENT,
            highlightbackground=BTN_BG,
        )
        self.func_entry.insert(0, "sin(x) * cos(y)")
        self.func_entry.pack(fill=tk.X, **padw)
        self.func_entry.bind("<Return>", lambda _: self._on_plot())

        self.plot_btn = tk.Button(
            self.left_frame,
            text="▶  Plot",
            font=FONT_BOLD,
            bg=ACCENT,
            fg="#11111b",
            activebackground="#74c7ec",
            relief="flat",
            cursor="hand2",
            command=self._on_plot,
        )
        self.plot_btn.pack(fill=tk.X, padx=12, pady=(6, 10))

        sep2 = ttk.Separator(self.left_frame, orient="horizontal")
        sep2.pack(fill=tk.X, padx=12, pady=4)

        # ── Range controls ──
        tk.Label(
            self.left_frame, text="Domain", font=FONT_BOLD, fg=FG, bg=BG, anchor="w"
        ).pack(fill=tk.X, **pad)

        range_frame = tk.Frame(self.left_frame, bg=BG)
        range_frame.pack(fill=tk.X, **padw)

        tk.Label(range_frame, text="x ∈", font=FONT, fg=FG, bg=BG).grid(
            row=0, column=0, sticky="w"
        )
        self.x_min_var = tk.DoubleVar(value=-5.0)
        self.x_max_var = tk.DoubleVar(value=5.0)
        tk.Entry(
            range_frame,
            textvariable=self.x_min_var,
            width=5,
            font=FONT_MONO,
            bg=ENTRY_BG,
            fg=ENTRY_FG,
            relief="flat",
            highlightthickness=1,
            highlightcolor=ACCENT,
            highlightbackground=BTN_BG,
        ).grid(row=0, column=1, padx=2)
        tk.Label(range_frame, text="to", font=FONT, fg=FG, bg=BG).grid(row=0, column=2)
        tk.Entry(
            range_frame,
            textvariable=self.x_max_var,
            width=5,
            font=FONT_MONO,
            bg=ENTRY_BG,
            fg=ENTRY_FG,
            relief="flat",
            highlightthickness=1,
            highlightcolor=ACCENT,
            highlightbackground=BTN_BG,
        ).grid(row=0, column=3, padx=2)

        tk.Label(range_frame, text="y ∈", font=FONT, fg=FG, bg=BG).grid(
            row=1, column=0, sticky="w", pady=(4, 0)
        )
        self.y_min_var = tk.DoubleVar(value=-5.0)
        self.y_max_var = tk.DoubleVar(value=5.0)
        tk.Entry(
            range_frame,
            textvariable=self.y_min_var,
            width=5,
            font=FONT_MONO,
            bg=ENTRY_BG,
            fg=ENTRY_FG,
            relief="flat",
            highlightthickness=1,
            highlightcolor=ACCENT,
            highlightbackground=BTN_BG,
        ).grid(row=1, column=1, padx=2, pady=(4, 0))
        tk.Label(range_frame, text="to", font=FONT, fg=FG, bg=BG).grid(
            row=1, column=2, pady=(4, 0)
        )
        tk.Entry(
            range_frame,
            textvariable=self.y_max_var,
            width=5,
            font=FONT_MONO,
            bg=ENTRY_BG,
            fg=ENTRY_FG,
            relief="flat",
            highlightthickness=1,
            highlightcolor=ACCENT,
            highlightbackground=BTN_BG,
        ).grid(row=1, column=3, padx=2, pady=(4, 0))

        # Resolution
        tk.Label(
            self.left_frame, text="Resolution", font=FONT, fg=FG, bg=BG, anchor="w"
        ).pack(fill=tk.X, padx=12, pady=(8, 0))
        self.resolution_var = tk.IntVar(value=80)
        self.res_slider = tk.Scale(
            self.left_frame,
            from_=20,
            to=200,
            orient=tk.HORIZONTAL,
            variable=self.resolution_var,
            bg=BG,
            fg=FG,
            highlightthickness=0,
            troughcolor=ENTRY_BG,
            activebackground=ACCENT,
            font=FONT,
            sliderlength=18,
        )
        self.res_slider.pack(fill=tk.X, padx=12)

        # Colormap
        tk.Label(
            self.left_frame, text="Colormap", font=FONT, fg=FG, bg=BG, anchor="w"
        ).pack(fill=tk.X, padx=12, pady=(6, 0))
        self.cmap_var = tk.StringVar(value="viridis")
        cmap_menu = ttk.Combobox(
            self.left_frame,
            textvariable=self.cmap_var,
            values=COLORMAPS,
            state="readonly",
            font=FONT,
        )
        cmap_menu.pack(fill=tk.X, padx=12, pady=2)

        sep3 = ttk.Separator(self.left_frame, orient="horizontal")
        sep3.pack(fill=tk.X, padx=12, pady=8)

        # ── Tangent point ──
        tk.Label(
            self.left_frame,
            text="Tangent Point",
            font=FONT_BOLD,
            fg=ACCENT2,
            bg=BG,
            anchor="w",
        ).pack(fill=tk.X, **pad)

        tk.Label(self.left_frame, text="x₀", font=FONT, fg=FG, bg=BG, anchor="w").pack(
            fill=tk.X, padx=12, pady=(4, 0)
        )
        self.x0_var = tk.DoubleVar(value=0.0)
        self.x0_slider = tk.Scale(
            self.left_frame,
            from_=-5.0,
            to=5.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.x0_var,
            bg=BG,
            fg=FG,
            highlightthickness=0,
            troughcolor=ENTRY_BG,
            activebackground=ACCENT2,
            font=FONT,
            sliderlength=18,
            command=self._on_tangent_change,
        )
        self.x0_slider.pack(fill=tk.X, padx=12)

        tk.Label(self.left_frame, text="y₀", font=FONT, fg=FG, bg=BG, anchor="w").pack(
            fill=tk.X, padx=12, pady=(2, 0)
        )
        self.y0_var = tk.DoubleVar(value=0.0)
        self.y0_slider = tk.Scale(
            self.left_frame,
            from_=-5.0,
            to=5.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.y0_var,
            bg=BG,
            fg=FG,
            highlightthickness=0,
            troughcolor=ENTRY_BG,
            activebackground=ACCENT2,
            font=FONT,
            sliderlength=18,
            command=self._on_tangent_change,
        )
        self.y0_slider.pack(fill=tk.X, padx=12)

        # Tangent plane size
        tk.Label(
            self.left_frame, text="Plane size", font=FONT, fg=FG, bg=BG, anchor="w"
        ).pack(fill=tk.X, padx=12, pady=(2, 0))
        self.plane_size_var = tk.DoubleVar(value=2.0)
        self.plane_size_slider = tk.Scale(
            self.left_frame,
            from_=0.5,
            to=6.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.plane_size_var,
            bg=BG,
            fg=FG,
            highlightthickness=0,
            troughcolor=ENTRY_BG,
            activebackground=ACCENT2,
            font=FONT,
            sliderlength=18,
            command=self._on_tangent_change,
        )
        self.plane_size_slider.pack(fill=tk.X, padx=12)

        # Toggles
        toggle_frame = tk.Frame(self.left_frame, bg=BG)
        toggle_frame.pack(fill=tk.X, padx=12, pady=(8, 2))
        tk.Checkbutton(
            toggle_frame,
            text="Surface",
            variable=self.show_surface,
            font=FONT,
            fg=FG,
            bg=BG,
            selectcolor=ENTRY_BG,
            activebackground=BG,
            activeforeground=FG,
            command=self._on_toggle,
        ).pack(side=tk.LEFT)
        tk.Checkbutton(
            toggle_frame,
            text="Tangent",
            variable=self.show_tangent,
            font=FONT,
            fg=ACCENT2,
            bg=BG,
            selectcolor=ENTRY_BG,
            activebackground=BG,
            activeforeground=ACCENT2,
            command=self._on_toggle,
        ).pack(side=tk.LEFT, padx=8)
        tk.Checkbutton(
            toggle_frame,
            text="Point",
            variable=self.show_point,
            font=FONT,
            fg=FG,
            bg=BG,
            selectcolor=ENTRY_BG,
            activebackground=BG,
            activeforeground=FG,
            command=self._on_toggle,
        ).pack(side=tk.LEFT)

        sep4 = ttk.Separator(self.left_frame, orient="horizontal")
        sep4.pack(fill=tk.X, padx=12, pady=8)

        # ── Info / equation display (LaTeX rendered) ──
        tk.Label(
            self.left_frame,
            text="Equations",
            font=FONT_BOLD,
            fg=FG,
            bg=BG,
            anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(2, 2))

        self.eq_fig = Figure(figsize=(3.2, 3.2), facecolor=BG_SECONDARY)
        self.eq_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.eq_ax = self.eq_fig.add_subplot(111)
        self.eq_ax.set_axis_off()
        self.eq_ax.set_facecolor(BG_SECONDARY)

        eq_canvas_widget = FigureCanvasTkAgg(self.eq_fig, master=self.left_frame)
        eq_canvas_widget.get_tk_widget().configure(highlightthickness=0, bd=0)
        eq_canvas_widget.get_tk_widget().pack(fill=tk.X, padx=12, pady=2)
        self.eq_canvas = eq_canvas_widget
        self._bind_mousewheel(eq_canvas_widget.get_tk_widget())

        sep5 = ttk.Separator(self.left_frame, orient="horizontal")
        sep5.pack(fill=tk.X, padx=12, pady=8)

        # ── 3D View Rotation ──
        tk.Label(
            self.left_frame,
            text="3D View Rotation",
            font=FONT_BOLD,
            fg=FG,
            bg=BG,
            anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(2, 2))

        tk.Label(
            self.left_frame, text="Elevation", font=FONT, fg=FG, bg=BG, anchor="w"
        ).pack(fill=tk.X, padx=12, pady=(4, 0))
        self.elev_var = tk.DoubleVar(value=30.0)
        self.elev_slider = tk.Scale(
            self.left_frame,
            from_=-90,
            to=90,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.elev_var,
            bg=BG,
            fg=FG,
            highlightthickness=0,
            troughcolor=ENTRY_BG,
            activebackground=ACCENT,
            font=FONT,
            sliderlength=18,
            command=self._on_view_change,
        )
        self.elev_slider.pack(fill=tk.X, padx=12)

        tk.Label(
            self.left_frame, text="Azimuth", font=FONT, fg=FG, bg=BG, anchor="w"
        ).pack(fill=tk.X, padx=12, pady=(2, 0))
        self.azim_var = tk.DoubleVar(value=-60.0)
        self.azim_slider = tk.Scale(
            self.left_frame,
            from_=-180,
            to=180,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.azim_var,
            bg=BG,
            fg=FG,
            highlightthickness=0,
            troughcolor=ENTRY_BG,
            activebackground=ACCENT,
            font=FONT,
            sliderlength=18,
            command=self._on_view_change,
        )
        self.azim_slider.pack(fill=tk.X, padx=12)

    def _build_canvas(self):
        self.fig = Figure(facecolor=BG_SECONDARY)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.30)

        # 3D surface subplot
        self.ax = self.fig.add_subplot(gs[0, 0], projection="3d")
        self._style_3d_axes(self.ax)

        # 2D contour subplot
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self._style_2d_axes(self.ax2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(self.right_frame, bg=BG_SECONDARY)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    def _style_3d_axes(self, ax):
        """Apply dark theme styling to a 3D axes."""
        ax.set_facecolor(BG_SECONDARY)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.tick_params(colors=FG, labelsize=9)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        ax.zaxis.label.set_color(FG)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    def _style_2d_axes(self, ax):
        """Apply dark theme styling to a 2D axes."""
        ax.set_facecolor(BG_SECONDARY)
        ax.tick_params(colors=FG, labelsize=9)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Contour & Gradient", color=FG, fontsize=10)
        for spine in ax.spines.values():
            spine.set_color(FG)

    # ---- Actions -----------------------------------------------------------
    def _initial_plot(self):
        """Plot the default function on startup."""
        self._on_plot()

    def _on_plot(self):
        """Parse the expression and redraw everything."""
        expr_str = self.func_entry.get().strip()
        if not expr_str:
            messagebox.showwarning("Empty input", "Please enter a function f(x, y).")
            return

        try:
            self.expr = parse_function(expr_str)
            self.f_np = build_evaluator(self.expr)
        except Exception as exc:
            messagebox.showerror(
                "Parse error",
                f"Could not parse the expression:\n\n{expr_str}\n\nError: {exc}",
            )
            return

        # Update slider ranges to match domain
        x_lo, x_hi = self.x_min_var.get(), self.x_max_var.get()
        y_lo, y_hi = self.y_min_var.get(), self.y_max_var.get()
        self.x0_slider.configure(from_=x_lo, to=x_hi)
        self.y0_slider.configure(from_=y_lo, to=y_hi)

        # Clamp tangent point into range
        self.x0_var.set(max(x_lo, min(x_hi, self.x0_var.get())))
        self.y0_var.set(max(y_lo, min(y_hi, self.y0_var.get())))

        self._redraw()

    def _redraw(self):
        """Full redraw of the 3D scene and contour plot."""
        self.ax.cla()
        self._style_3d_axes(self.ax)

        # Remove old colorbar if present
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
        self.ax2.cla()
        self._style_2d_axes(self.ax2)

        if self.f_np is None:
            self.canvas.draw_idle()
            return

        x_lo, x_hi = self.x_min_var.get(), self.x_max_var.get()
        y_lo, y_hi = self.y_min_var.get(), self.y_max_var.get()
        n = self.resolution_var.get()

        xs = np.linspace(x_lo, x_hi, n)
        ys = np.linspace(y_lo, y_hi, n)
        X, Y = np.meshgrid(xs, ys)

        try:
            Z = self.f_np(X, Y)
            # Make sure Z is an array of the right shape
            if np.isscalar(Z):
                Z = np.full_like(X, float(Z))
            elif Z.shape != X.shape:
                Z = np.broadcast_to(Z, X.shape).copy()
        except Exception as exc:
            messagebox.showerror(
                "Evaluation error", f"Could not evaluate f(x,y):\n{exc}"
            )
            self.canvas.draw_idle()
            return

        # Surface
        if self.show_surface.get():
            self.ax.plot_surface(
                X,
                Y,
                Z,
                cmap=self.cmap_var.get(),
                alpha=0.85,
                edgecolor="none",
                antialiased=True,
                rstride=1,
                cstride=1,
            )

        # ── Stabilize Axes ──
        self.ax.set_xlim(x_lo, x_hi)
        self.ax.set_ylim(y_lo, y_hi)

        # Calculate Z limits with some padding based on function range
        z_min, z_max = np.min(Z), np.max(Z)
        z_span = z_max - z_min
        if z_span < 1e-9:
            z_span = 1.0
        padding = 0.15 * z_span
        self.ax.set_zlim(z_min - padding, z_max + padding)

        # Tangent plane + point (returns gradient components)
        grad_info = self._draw_tangent()

        # Contour plot
        self._draw_contour(X, Y, Z, grad_info)

        # Apply current view angles
        self.ax.view_init(elev=self.elev_var.get(), azim=self.azim_var.get())

        self.canvas.draw_idle()

    def _draw_tangent(self):
        """Draw tangent plane and point marker. Returns (x0, y0, fx0, fy0) or None."""
        if self.expr is None:
            return None

        x0 = self.x0_var.get()
        y0 = self.y0_var.get()

        try:
            z0, fx0, fy0, plane = tangent_plane_expr(self.expr, x0, y0)
        except Exception:
            self._update_eq_canvas(error=True)
            return None

        # Update LaTeX equation display
        grad_mag = np.sqrt(fx0**2 + fy0**2)
        self._update_eq_canvas(
            z0=z0,
            fx0=fx0,
            fy0=fy0,
            x0=x0,
            y0=y0,
            grad_mag=grad_mag,
        )

        # Draw tangent plane patch
        if self.show_tangent.get():
            half = self.plane_size_var.get()
            tp_n = 20
            txs = np.linspace(x0 - half, x0 + half, tp_n)
            tys = np.linspace(y0 - half, y0 + half, tp_n)
            TX, TY = np.meshgrid(txs, tys)
            TZ = z0 + fx0 * (TX - x0) + fy0 * (TY - y0)
            self.ax.plot_surface(
                TX,
                TY,
                TZ,
                color="#f38ba8",
                alpha=0.45,
                edgecolor="none",
                antialiased=True,
            )

        # Draw point marker
        if self.show_point.get():
            self.ax.scatter(
                [x0],
                [y0],
                [z0],
                color="#f38ba8",
                s=80,
                edgecolors="#11111b",
                linewidths=1.5,
                zorder=10,
                depthshade=False,
            )

        return (x0, y0, fx0, fy0)

    def _draw_contour(self, X, Y, Z, grad_info):
        """Draw the 2D contour plot with gradient vector."""
        cmap = self.cmap_var.get()

        # Filled contours
        cf = self.ax2.contourf(X, Y, Z, levels=25, cmap=cmap, alpha=0.9)
        self.colorbar = self.fig.colorbar(cf, ax=self.ax2, fraction=0.046, pad=0.04)
        self.colorbar.ax.tick_params(colors=FG, labelsize=8)
        self.colorbar.outline.set_edgecolor(FG)

        # Contour lines for clarity
        self.ax2.contour(X, Y, Z, levels=15, colors="#cdd6f488", linewidths=0.5)

        # Gradient arrow at the tangent point
        if grad_info is not None:
            x0, y0, fx0, fy0 = grad_info
            # Point marker
            self.ax2.plot(
                x0,
                y0,
                "o",
                color="#f38ba8",
                markersize=8,
                markeredgecolor="#11111b",
                markeredgewidth=1.5,
                zorder=5,
            )
            # set axes equal
            self.ax2.set_aspect("equal", adjustable="box")
            # Gradient arrow
            grad_mag = np.sqrt(fx0**2 + fy0**2)
            if grad_mag > 1e-8:
                # Scale arrow length relative to domain for visibility
                x_span = X.max() - X.min()
                scale = x_span * 0.15 / grad_mag
                self.ax2.annotate(
                    "",
                    xy=(x0 + fx0 * scale, y0 + fy0 * scale),
                    xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#a6e3a1",
                        lw=2.5,
                        mutation_scale=18,
                    ),
                    zorder=6,
                )
                self.ax2.annotate(
                    "∇f",
                    xy=(x0 + fx0 * scale * 1.1, y0 + fy0 * scale * 1.1),
                    color="#a6e3a1",
                    fontsize=10,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=6,
                )

    def _update_eq_canvas(
        self, z0=0, fx0=0, fy0=0, x0=0, y0=0, grad_mag=0, error=False
    ):
        """Update the LaTeX equation canvas."""
        self.eq_ax.cla()
        self.eq_ax.set_axis_off()

        if error:
            self.eq_ax.text(
                0.05,
                0.5,
                "Error calculating tangent",
                color=ACCENT2,
                fontsize=11,
                transform=self.eq_ax.transAxes,
            )
        else:
            # Original function in LaTeX
            fn_latex = sp.latex(self.expr) if self.expr else "..."
            fn_tex = rf"$f(x, y) = {fn_latex}$"

            # Symbolic Partial Derivatives
            fx_sym = sp.diff(self.expr, x_sym)
            fy_sym = sp.diff(self.expr, y_sym)
            fx_latex = sp.latex(fx_sym)
            fy_latex = sp.latex(fy_sym)
            fx_tex = rf"$\partial_x f(x, y) = {fx_latex}$"
            fy_tex = rf"$\partial_y f(x, y) = {fy_latex}$"

            # Tangent plane equation
            sym_eq_tex = rf"$z_{{\text{{tan}}}} = {fx_latex}|_{{x_0={x0:.2f}, y_0={y0:.2f}}}(x - x_0) + {fy_latex}|_{{x_0={x0:.2f}, y_0={y0:.2f}}}(y - y_0)$"
            eq_tex = rf"$z_{{\text{{tan}}}} = {z0:.2f} + {fx0:.2f}(x - {x0:.2f}) + {fy0:.2f}(y - {y0:.2f})$"

            # Point coordinates
            pt_tex = rf"$P = ({x0:.2f}, {y0:.2f}, {z0:.2f})$"

            # Gradient vector and magnitude
            grad_tex = rf"$\nabla f = ({fx0:.4f}, {fy0:.4f}) \to \|\nabla f\| = {grad_mag:.4f}$"

            full_tex = f"{fn_tex}\n\n{fx_tex}\n{fy_tex}\n\n{sym_eq_tex}\n{eq_tex}\n{pt_tex}\n{grad_tex}"

            self.eq_ax.text(
                0.05,
                0.97,
                full_tex,
                color=ACCENT,
                fontsize=9.5,
                va="top",
                transform=self.eq_ax.transAxes,
                linespacing=2.0,
            )

        self.eq_canvas.draw_idle()

    def _on_frame_configure(self, event):
        """Reset the scroll region to encompass the inner frame."""
        self.canvas_left.configure(scrollregion=self.canvas_left.bbox("all"))

    def _on_canvas_configure(self, event):
        """Update the width of the inner frame to fill the canvas."""
        self.canvas_left.itemconfig(self.canvas_window, width=event.width)

    def _bind_mousewheel(self, widget):
        """Bind mouse wheel events recursively to a widget and all children."""
        widget.bind("<Button-4>", self._on_mousewheel)
        widget.bind("<Button-5>", self._on_mousewheel)
        widget.bind("<MouseWheel>", self._on_mousewheel)
        for child in widget.winfo_children():
            self._bind_mousewheel(child)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        if event.num == 4 or event.delta > 0:
            self.canvas_left.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.canvas_left.yview_scroll(1, "units")

    def _on_tangent_change(self, _=None):
        """Called when x₀, y₀, or plane size sliders change."""
        if self.expr is None:
            return
        self._redraw()

    def _on_view_change(self, _=None):
        """Called when elevation/azimuth sliders change — lightweight update."""
        self.ax.view_init(elev=self.elev_var.get(), azim=self.azim_var.get())
        self.canvas.draw_idle()

    def _on_toggle(self):
        """Called when any visibility toggle changes."""
        self._redraw()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    root.geometry("1500x800")
    app = FunctionVizApp(root)  # noqa: F841
    root.mainloop()


if __name__ == "__main__":
    main()
