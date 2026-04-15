//! Zero center of mass linear drift and rotation for all atoms in the system.

use lin_alg::{f32::Vec3, f64::Vec3 as Vec3F64};

use crate::{
    MdState,
    solvent::{H_MASS, MASS_WATER_MOL, O_MASS},
};

const EPS: f64 = 1e-6;

impl MdState {
    fn dynamic_mass_and_momentum(&self) -> (f64, Vec3F64) {
        let mut mass_sum = 0.0;
        let mut p_sum = Vec3F64::new_zero();

        for a in &self.atoms {
            if a.static_ {
                continue;
            }
            mass_sum += a.mass as f64;
            let p: Vec3F64 = (a.vel * a.mass).into();
            p_sum += p;
        }

        for w in &self.water {
            mass_sum += MASS_WATER_MOL as f64;

            let p_o: Vec3F64 = (w.o.vel * O_MASS).into();
            let p_h0: Vec3F64 = (w.h0.vel * H_MASS).into();
            let p_h1: Vec3F64 = (w.h1.vel * H_MASS).into();

            p_sum += p_o + p_h0 + p_h1;
        }

        (mass_sum, p_sum)
    }

    fn shift_dynamic_positions(&mut self, displacement: Vec3) {
        for a in &mut self.atoms {
            if a.static_ {
                continue;
            }
            a.posit += displacement;
        }

        for w in &mut self.water {
            w.o.posit += displacement;
            w.h0.posit += displacement;
            w.h1.posit += displacement;
            w.m.posit += displacement;
        }
    }

    /// Remove center-of-mass drift. This can help stabilize system energy.
    /// We perform the sums here as f64.
    pub fn zero_linear_momentum(&mut self) {
        let (mass_sum, p_sum) = self.dynamic_mass_and_momentum();

        if mass_sum <= EPS {
            return;
        }

        let vel_com: Vec3 = (p_sum / mass_sum).into();

        // Subtract uniformly so Σ m v' = 0
        for a in &mut self.atoms {
            if a.static_ {
                continue;
            }
            a.vel -= vel_com;
        }

        // I don't think we need to use SHAKE/RATTLE here, as the velocity
        // change is uniform for the solvent atoms in a given mol.
        for w in &mut self.water {
            w.o.vel -= vel_com;
            w.h0.vel -= vel_com;
            w.h1.vel -= vel_com;
        }
    }

    /// GROMACS-style linear acceleration correction: use the current COM velocity
    /// to estimate the displacement accumulated over the removal interval under
    /// nearly constant COM acceleration, shift positions back uniformly, then
    /// remove the translational COM velocity.
    pub(crate) fn zero_linear_momentum_acceleration_corrected(&mut self, interval_dt: f32) {
        if interval_dt <= 0.0 {
            self.zero_linear_momentum();
            return;
        }

        let (mass_sum, p_sum) = self.dynamic_mass_and_momentum();
        if mass_sum <= EPS {
            return;
        }

        let vel_com: Vec3 = (p_sum / mass_sum).into();

        // If the COM acceleration has been nearly constant since the previous
        // removal event, the displacement is 0.5 * a * T^2 = 0.5 * v_end * T.
        self.shift_dynamic_positions(-(vel_com * (0.5 * interval_dt)));
        self.zero_linear_momentum();
    }

    // todo: Assess if you want this for multi-molecule systems.
    /// Remove rigid-body rotation.
    /// Computes ω from I ω = L about the atoms' COM, then sets v' = v - ω × (r - r_cm).
    pub fn zero_angular_momentum(&mut self) {
        let mut mass_sum = 0.0;
        let mut m_r_sum = Vec3F64::new_zero();

        for a in &self.atoms {
            if a.static_ {
                continue;
            }
            mass_sum += a.mass as f64;

            let m_r: Vec3F64 = (a.posit * a.mass).into();
            m_r_sum += m_r;
        }

        for w in &self.water {
            mass_sum += MASS_WATER_MOL as f64;

            let mr_o: Vec3F64 = (w.o.posit * O_MASS).into();
            let mr_h0: Vec3F64 = (w.h0.posit * H_MASS).into();
            let mr_h1: Vec3F64 = (w.h1.posit * H_MASS).into();

            m_r_sum += mr_o + mr_h0 + mr_h1;
        }

        if mass_sum <= EPS {
            return;
        }
        let rot_com: Vec3 = (m_r_sum / mass_sum).into();

        // Build inertia tensor I and angular momentum L about r_cm
        let mut i_xx = 0.0;
        let mut i_xy = 0.0;
        let mut i_xz = 0.0;
        let mut i_yy = 0.0;
        let mut i_yz = 0.0;
        let mut i_zz = 0.0;

        let mut L = Vec3::new_zero();

        for a in &self.atoms {
            if a.static_ {
                continue;
            }
            let m = a.mass;

            let r = a.posit - rot_com;
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

        for w in &self.water {
            for a in [&w.o, &w.h0, &w.h1] {
                let m = a.mass;

                let r = a.posit - rot_com;
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
            if a.static_ {
                continue;
            }
            let r = a.posit - rot_com;
            a.vel -= omega.cross(r);
        }

        // todo: Do we need to shake/rattle here? likely.
        for w in &mut self.water {
            let r_o = w.o.posit - rot_com;
            let r_h0 = w.h0.posit - rot_com;
            let r_h1 = w.h1.posit - rot_com;

            w.o.vel -= omega.cross(r_o);
            w.h0.vel -= omega.cross(r_h0);
            w.h1.vel -= omega.cross(r_h1);
        }

        // Clean up any translation introduced by roundoff
        self.zero_linear_momentum();
    }
}

#[cfg(test)]
mod tests {
    use lin_alg::f32::Vec3;

    use crate::{AtomDynamics, MdState};

    fn atom(posit: Vec3, vel: Vec3, mass: f32, static_: bool) -> AtomDynamics {
        AtomDynamics {
            posit,
            vel,
            mass,
            static_,
            ..Default::default()
        }
    }

    #[test]
    fn zero_linear_momentum_ignores_static_atoms() {
        let mut state = MdState {
            atoms: vec![
                atom(
                    Vec3::new(10., 10., 10.),
                    Vec3::new(100., -20., 3.),
                    100.0,
                    true,
                ),
                atom(Vec3::new(0., 0., 0.), Vec3::new(4., 0., 0.), 2.0, false),
                atom(Vec3::new(1., 0., 0.), Vec3::new(-2., 0., 0.), 4.0, false),
            ],
            ..Default::default()
        };

        let static_vel_before = state.atoms[0].vel;
        state.zero_linear_momentum();

        assert_eq!(state.atoms[0].vel, static_vel_before);

        let p_sum = state
            .atoms
            .iter()
            .filter(|a| !a.static_)
            .fold(Vec3::new_zero(), |acc, a| acc + a.vel * a.mass);

        assert!(
            p_sum.magnitude() < 1e-5,
            "dynamic momentum not zeroed: {p_sum:?}"
        );
    }

    #[test]
    fn zero_angular_momentum_leaves_static_atoms_unchanged() {
        let mut state = MdState {
            atoms: vec![
                atom(
                    Vec3::new(20., 20., 20.),
                    Vec3::new(50., -10., 5.),
                    50.0,
                    true,
                ),
                atom(Vec3::new(-1., 0., 0.), Vec3::new(0., -1., 0.), 1.0, false),
                atom(Vec3::new(1., 0., 0.), Vec3::new(0., 1., 0.), 1.0, false),
                atom(Vec3::new(0., 1., 0.), Vec3::new(-1., 0., 0.), 1.0, false),
            ],
            ..Default::default()
        };

        let static_vel_before = state.atoms[0].vel;
        state.zero_angular_momentum();

        assert_eq!(state.atoms[0].vel, static_vel_before);

        let dynamic_speed_sq: f32 = state
            .atoms
            .iter()
            .filter(|a| !a.static_)
            .map(|a| a.vel.magnitude_squared())
            .sum();

        assert!(
            dynamic_speed_sq < 1e-4,
            "dynamic angular motion was not removed: {dynamic_speed_sq}"
        );
    }

    #[test]
    fn acceleration_correction_shifts_dynamic_positions_only() {
        let mut state = MdState {
            atoms: vec![
                atom(Vec3::new(50., 0., 0.), Vec3::new(9., 0., 0.), 10.0, true),
                atom(Vec3::new(1., 0., 0.), Vec3::new(2., 0., 0.), 2.0, false),
                atom(Vec3::new(3., 0., 0.), Vec3::new(2., 0., 0.), 2.0, false),
            ],
            ..Default::default()
        };

        state.zero_linear_momentum_acceleration_corrected(4.0);

        assert_eq!(state.atoms[0].posit, Vec3::new(50., 0., 0.));
        assert_eq!(state.atoms[1].posit, Vec3::new(-3., 0., 0.));
        assert_eq!(state.atoms[2].posit, Vec3::new(-1., 0., 0.));

        let p_sum = state
            .atoms
            .iter()
            .filter(|a| !a.static_)
            .fold(Vec3::new_zero(), |acc, a| acc + a.vel * a.mass);

        assert!(
            p_sum.magnitude() < 1e-5,
            "dynamic momentum not zeroed: {p_sum:?}"
        );
    }
}
