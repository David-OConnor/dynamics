//! Bonded forces. These assume immutable covalent bonds, and operate using spring-like mechanisms,
//! restoring bond lengths, angles between 3 atoms, linear dihedral angles, and dihedral angles
//! among four atoms in a hub configuration (Improper dihedrals). Bonds to hydrogen are treated
//! differently: They have rigid lengths, which is good enough, and allows for a larger timestep.

use crate::{MdState, SHAKE_MAX_IT, SHAKE_TOL, bonded_forces, split2_mut, split3_mut, split4_mut};

const EPS_SHAKE_RATTLE: f32 = 1.0e-8;

impl MdState {
    pub fn apply_bond_stretching_forces(&mut self) {
        for (indices, params) in &self.force_field_params.bond_stretching {
            let (a_0, a_1) = split2_mut(&mut self.atoms, indices.0, indices.1);

            let (f, energy) = bonded_forces::f_bond_stretching(a_0.posit, a_1.posit, params);

            // We divide by mass in `step`.
            a_0.accel += f;
            a_1.accel -= f;

            self.potential_energy += energy as f64;
        }
    }

    /// This maintains bond angles between sets of three atoms as they should be from hybridization.
    /// It reflects this hybridization, steric clashes, and partial double-bond character. This
    /// identifies deviations from the ideal angle, calculates restoring torque, and applies forces
    /// based on this to push the atoms back into their ideal positions in the molecule.
    ///
    /// Valence angles, which are the angle formed by two adjacent bonds ba et bc
    /// in a same molecule; a valence angle tends to maintain constant the anglê
    /// abc. A valence angle is thus concerned by the positions of three atoms.
    pub fn apply_angle_bending_forces(&mut self) {
        for (indices, params) in &self.force_field_params.angle {
            let (a_0, a_1, a_2) = split3_mut(&mut self.atoms, indices.0, indices.1, indices.2);

            let ((f_0, f_1, f_2), energy) =
                bonded_forces::f_angle_bending(a_0.posit, a_1.posit, a_2.posit, params);

            // We divide by mass in `step`.
            a_0.accel += f_0;
            a_1.accel += f_1;
            a_2.accel += f_2;

            self.potential_energy += energy as f64;
        }
    }

    /// This maintains dihedral angles. (i.e. the angle between four atoms in a sequence). This models
    /// effects such as σ-bond overlap (e.g. staggered conformations), π-conjugation, which locks certain
    /// dihedrals near 0 or τ, and steric hindrance. (Bulky groups clashing).
    ///
    /// This applies both "proper" linear dihedral angles, and "improper", hub-and-spoke dihedrals. These
    /// two angles are calculated in the same way, but the covalent-bond arrangement of the 4 atoms differs.
    pub fn apply_dihedral_forces(&mut self, improper: bool) {
        let dihedrals = if improper {
            &self.force_field_params.improper
        } else {
            &self.force_field_params.dihedral
        };

        for (indices, params) in dihedrals {
            // Split the four atoms mutably without aliasing
            let (a_0, a_1, a_2, a_3) =
                split4_mut(&mut self.atoms, indices.0, indices.1, indices.2, indices.3);

            let ((f_0, f_1, f_2, f3), energy) = bonded_forces::f_dihedral(
                a_0.posit, a_1.posit, a_2.posit, a_3.posit, params, improper,
            );

            // We divide by mass in `step`.
            a_0.accel += f_0;
            a_1.accel += f_1;
            a_2.accel += f_2;
            a_3.accel += f3;

            self.potential_energy += energy as f64;
        }
    }

    /// Part of our SHAKE + RATTLE algorithms for fixed hydrogens.
    pub fn shake_hydrogens(&mut self) {
        for _ in 0..SHAKE_MAX_IT {
            let mut max_corr: f32 = 0.0;

            for (indices, (r0_sq, inv_mass)) in &self.force_field_params.bond_rigid_constraints {
                let (ai, aj) = split2_mut(&mut self.atoms, indices.0, indices.1);

                let diff = aj.posit - ai.posit;
                let dist_sq = diff.magnitude_squared();

                // println!("Diff: {:.3}", dist_sq);

                // λ = (r² − r₀²) / (2·inv_m · r_ij·r_ij)
                let lambda = (dist_sq - r0_sq) / (2.0 * inv_mass * dist_sq.max(EPS_SHAKE_RATTLE));
                let corr = diff * lambda; // vector correction

                ai.posit += corr / ai.mass;
                aj.posit -= corr / aj.mass;

                max_corr = max_corr.max(corr.magnitude());
            }

            // Converged
            if max_corr < SHAKE_TOL {
                break;
            }
        }
    }

    /// Part of our SHAKE + RATTLE algorithms for fixed hydrogens.
    pub fn rattle_hydrogens(&mut self, mol_i: usize) {
        // RATTLE on velocities so that d/dt(|r|²)=0  ⇒  v_ij · r_ij = 0
        for (indices, (_r0_sq, inv_mass)) in &self.force_field_params.bond_rigid_constraints {
            // println!("RATTLE: {:?}", (i, j, _r0));
            let (ai, aj) = split2_mut(&mut self.atoms, indices.0, indices.1);

            let r_meas = aj.posit - ai.posit;
            let v_meas = aj.vel - ai.vel;
            let r_sq = r_meas.magnitude_squared();

            // λ' = (v_ij·r_ij) / (inv_m · r_ij·r_ij)
            let lambda_p = v_meas.dot(r_meas) / (inv_mass * r_sq.max(EPS_SHAKE_RATTLE));
            let corr_v = r_meas * lambda_p;

            ai.vel += corr_v / ai.mass;
            aj.vel -= corr_v / aj.mass;
        }
    }
}
