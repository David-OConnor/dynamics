//! Relates to zeroing center-of-mass drift adn possibly rotation.

use lin_alg::f32::Vec3;

use crate::MdState;

impl MdState {
    /// Remove center-of-mass drift for "atoms" only (exclude water).
    pub fn zero_linear_momentum_atoms(&mut self) {
        let mut m_sum = 0.0;
        let mut p_sum = Vec3::new_zero(); // Σ m v

        for a in &self.atoms {
            let m = a.mass;
            m_sum += m;
            p_sum += a.vel * m;
        }
        if m_sum <= 0.0 {
            return;
        }

        let v_cm = p_sum / m_sum; // COM velocity
        // Subtract uniformly so Σ m v' = 0
        for a in &mut self.atoms {
            a.vel -= v_cm;
        }
    }

    // todo: Assess if you want this for multi-molecule systems.
    /// Remove rigid-body rotation for "atoms" only (exclude water).
    /// Computes ω from I ω = L about the atoms' COM, then sets v' = v - ω × (r - r_cm).
    pub fn zero_angular_momentum_atoms(&mut self) {
        // COM position for atoms (wrapped is fine; all r use the same frame)
        let mut m_sum = 0.0;
        let mut m_r_sum = Vec3::new_zero();
        for a in &self.atoms {
            let m = a.mass;
            m_sum += m;
            m_r_sum += a.posit * m;
        }
        if m_sum <= 0.0 {
            return;
        }
        let r_cm = m_r_sum / m_sum;

        // Build inertia tensor I and angular momentum L about r_cm
        let mut i_xx = 0.0f32;
        let mut i_xy = 0.0f32;
        let mut i_xz = 0.0f32;
        let mut i_yy = 0.0f32;
        let mut i_yz = 0.0f32;
        let mut i_zz = 0.0f32;
        let mut L = Vec3::new_zero();

        for a in &self.atoms {
            let m = a.mass;
            // r relative to COM (use minimum-image if you prefer: self.cell.min_image(a.posit - r_cm))
            let r = a.posit - r_cm;
            let vxr = r.cross(a.vel);
            L += vxr * m;

            let rx = r.x;
            let ry = r.y;
            let rz = r.z;
            let r2 = rx * rx + ry * ry + rz * rz;
            i_xx += m * (r2 - rx * rx);
            i_yy += m * (r2 - ry * ry);
            i_zz += m * (r2 - rz * rz);
            i_xy -= m * (rx * ry);
            i_xz -= m * (rx * rz);
            i_yz -= m * (ry * rz);
        }

        // Inertia tensor (symmetric)
        let I = [[i_xx, i_xy, i_xz], [i_xy, i_yy, i_yz], [i_xz, i_yz, i_zz]];

        // Solve I * ω = L  (3x3). Add tiny Tikhonov for degeneracy.
        let eps = 1.0e-6f32;
        let Ireg = [
            [I[0][0] + eps, I[0][1], I[0][2]],
            [I[1][0], I[1][1] + eps, I[1][2]],
            [I[2][0], I[2][1], I[2][2] + eps],
        ];

        // Inverse of 3x3 (cofactor / det)
        let det = Ireg[0][0] * (Ireg[1][1] * Ireg[2][2] - Ireg[1][2] * Ireg[2][1])
            - Ireg[0][1] * (Ireg[1][0] * Ireg[2][2] - Ireg[1][2] * Ireg[2][0])
            + Ireg[0][2] * (Ireg[1][0] * Ireg[2][1] - Ireg[1][1] * Ireg[2][0]);

        if det.abs() < 1e-12 {
            return;
        }

        let inv_det = 1.0 / det;
        let inv = [
            [
                (Ireg[1][1] * Ireg[2][2] - Ireg[1][2] * Ireg[2][1]) * inv_det,
                (Ireg[0][2] * Ireg[2][1] - Ireg[0][1] * Ireg[2][2]) * inv_det,
                (Ireg[0][1] * Ireg[1][2] - Ireg[0][2] * Ireg[1][1]) * inv_det,
            ],
            [
                (Ireg[1][2] * Ireg[2][0] - Ireg[1][0] * Ireg[2][2]) * inv_det,
                (Ireg[0][0] * Ireg[2][2] - Ireg[0][2] * Ireg[2][0]) * inv_det,
                (Ireg[0][2] * Ireg[1][0] - Ireg[0][0] * Ireg[1][2]) * inv_det,
            ],
            [
                (Ireg[1][0] * Ireg[2][1] - Ireg[1][1] * Ireg[2][0]) * inv_det,
                (Ireg[0][1] * Ireg[2][0] - Ireg[0][0] * Ireg[2][1]) * inv_det,
                (Ireg[0][0] * Ireg[1][1] - Ireg[0][1] * Ireg[1][0]) * inv_det,
            ],
        ];

        let omega = Vec3 {
            x: inv[0][0] * L.x + inv[0][1] * L.y + inv[0][2] * L.z,
            y: inv[1][0] * L.x + inv[1][1] * L.y + inv[1][2] * L.z,
            z: inv[2][0] * L.x + inv[2][1] * L.y + inv[2][2] * L.z,
        };

        // If ω is tiny, nothing to do
        if !omega.x.is_finite() || !omega.y.is_finite() || !omega.z.is_finite() {
            return;
        }
        if omega.magnitude_squared() < 1e-16 {
            return;
        }

        // v' = v - ω × (r - r_cm)
        for a in &mut self.atoms {
            let r = a.posit - r_cm;
            a.vel -= omega.cross(r);
        }

        // Clean up any translation introduced by roundoff
        self.zero_linear_momentum_atoms();
    }
}
