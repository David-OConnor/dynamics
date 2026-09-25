//! CMAP: A correction energy on a grid over two dihedral angles, φ and ψ, used by CHARMM (and
//! Amber ff19SB) for protein backbones. We interpolate it with bicubic patches, whose derivatives
//! at the grid points come from periodic cubic splines, as CHARMM and OpenMM do.

use std::f64::consts::{PI, TAU};

/// An energy grid over (φ, ψ), with precomputed bicubic coefficients.
#[derive(Clone, Debug, PartialEq)]
pub struct CmapGrid {
    /// Grid points per dimension.
    size: usize,
    /// kcal/mol. Row-major: `energies[i * size + j]` is at φ = −180° + i Δ, ψ = −180° + j Δ, with
    /// Δ = 360° / size. (CHARMM's convention)
    energies: Vec<f64>,
    /// Per grid cell: a[m][n], the coefficient of t^m u^n, where t and u are the fractional
    /// positions within the cell along φ and ψ.
    coeffs: Vec<[[f64; 4]; 4]>,
}

impl CmapGrid {
    /// `energies` are in kcal/mol, in the layout described on the `energies` field.
    pub fn new(size: usize, energies: Vec<f64>) -> Self {
        assert_eq!(energies.len(), size * size, "CMAP grid must be size × size");
        let h = TAU / size as f64;
        let e = |i: usize, j: usize| energies[(i % size) * size + (j % size)];

        // Derivatives at the grid points, from periodic splines along each axis.
        let mut d_phi = vec![0.; size * size];
        let mut d_psi = vec![0.; size * size];
        let mut d_phi_psi = vec![0.; size * size];

        for j in 0..size {
            let column: Vec<f64> = (0..size).map(|i| e(i, j)).collect();
            for (i, d) in periodic_spline_derivatives(&column, h)
                .into_iter()
                .enumerate()
            {
                d_phi[i * size + j] = d;
            }
        }
        for i in 0..size {
            let row: Vec<f64> = (0..size).map(|j| e(i, j)).collect();
            for (j, d) in periodic_spline_derivatives(&row, h).into_iter().enumerate() {
                d_psi[i * size + j] = d;
            }
            let row: Vec<f64> = (0..size).map(|j| d_phi[i * size + j]).collect();
            for (j, d) in periodic_spline_derivatives(&row, h).into_iter().enumerate() {
                d_phi_psi[i * size + j] = d;
            }
        }

        // Bicubic coefficients per cell: a = L F Lᵀ, with derivatives scaled to unit cells.
        const L: [[f64; 4]; 4] = [
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [-3., 3., -2., -1.],
            [2., -2., 1., 1.],
        ];
        let at = |v: &[f64], i: usize, j: usize| v[(i % size) * size + (j % size)];

        let mut coeffs = Vec::with_capacity(size * size);
        for i in 0..size {
            for j in 0..size {
                let (i1, j1) = (i + 1, j + 1);
                let f = [
                    [
                        e(i, j),
                        e(i, j1),
                        h * at(&d_psi, i, j),
                        h * at(&d_psi, i, j1),
                    ],
                    [
                        e(i1, j),
                        e(i1, j1),
                        h * at(&d_psi, i1, j),
                        h * at(&d_psi, i1, j1),
                    ],
                    [
                        h * at(&d_phi, i, j),
                        h * at(&d_phi, i, j1),
                        h * h * at(&d_phi_psi, i, j),
                        h * h * at(&d_phi_psi, i, j1),
                    ],
                    [
                        h * at(&d_phi, i1, j),
                        h * at(&d_phi, i1, j1),
                        h * h * at(&d_phi_psi, i1, j),
                        h * h * at(&d_phi_psi, i1, j1),
                    ],
                ];

                let mut lf = [[0.; 4]; 4];
                for r in 0..4 {
                    for c in 0..4 {
                        lf[r][c] = (0..4).map(|k| L[r][k] * f[k][c]).sum();
                    }
                }
                let mut a = [[0.; 4]; 4];
                for r in 0..4 {
                    for c in 0..4 {
                        a[r][c] = (0..4).map(|k| lf[r][k] * L[c][k]).sum();
                    }
                }
                coeffs.push(a);
            }
        }

        Self {
            size,
            energies,
            coeffs,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn energies(&self) -> &[f64] {
        &self.energies
    }

    /// Returns (energy, dE/dφ, dE/dψ), in kcal/mol and kcal/mol/rad. Angles are in radians, in
    /// any range.
    pub fn eval(&self, phi: f64, psi: f64) -> (f64, f64, f64) {
        let h = TAU / self.size as f64;
        let cell = |angle: f64| -> (usize, f64) {
            let x = (angle + PI).rem_euclid(TAU) / h;
            let i = (x.floor() as usize).min(self.size - 1);
            (i, x - i as f64)
        };
        let (i, t) = cell(phi);
        let (j, u) = cell(psi);
        let a = &self.coeffs[i * self.size + j];

        let (mut e, mut de_dt, mut de_du) = (0., 0., 0.);
        for m in 0..4 {
            for n in 0..4 {
                let c = a[m][n];
                e += c * t.powi(m as i32) * u.powi(n as i32);
                if m > 0 {
                    de_dt += c * m as f64 * t.powi(m as i32 - 1) * u.powi(n as i32);
                }
                if n > 0 {
                    de_du += c * n as f64 * t.powi(m as i32) * u.powi(n as i32 - 1);
                }
            }
        }

        (e, de_dt / h, de_du / h)
    }
}

/// First derivatives at the knots of a periodic cubic spline through `y`, with spacing `h`.
fn periodic_spline_derivatives(y: &[f64], h: f64) -> Vec<f64> {
    let n = y.len();
    let yi = |i: isize| y[i.rem_euclid(n as isize) as usize];

    // Second derivatives M: M[i-1] + 4 M[i] + M[i+1] = 6 (y[i+1] - 2 y[i] + y[i-1]) / h², cyclic.
    let mut a = vec![vec![0.; n]; n];
    let mut b = vec![0.; n];
    for i in 0..n {
        a[i][(i + n - 1) % n] += 1.;
        a[i][i] += 4.;
        a[i][(i + 1) % n] += 1.;
        let ii = i as isize;
        b[i] = 6. * (yi(ii + 1) - 2. * yi(ii) + yi(ii - 1)) / (h * h);
    }
    let m = solve_dense(a, b);

    (0..n)
        .map(|i| {
            let ii = i as isize;
            (yi(ii + 1) - yi(ii)) / h - h * (2. * m[i] + m[(i + 1) % n]) / 6.
        })
        .collect()
}

/// Solve A x = b with Gaussian elimination and partial pivoting. For the small, well-conditioned
/// systems above.
fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&r0, &r1| a[r0][col].abs().total_cmp(&a[r1][col].abs()))
            .unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        for r in col + 1..n {
            let factor = a[r][col] / a[col][col];
            for c in col..n {
                a[r][c] -= factor * a[col][c];
            }
            b[r] -= factor * b[col];
        }
    }
    let mut x = vec![0.; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|c| a[r][c] * x[c]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    x
}
