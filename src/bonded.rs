//! Bonded forces. These assume immutable covalent bonds, and operate using spring-like mechanisms,
//! restoring bond lengths, angles between 3 atoms, linear dihedral angles, and dihedral angles
//! among four atoms in a hub configuration (Improper dihedrals). Bonds to hydrogen are treated
//! differently: They have rigid lengths, which is good enough, and allows for a larger timestep.

use lin_alg::f32::Vec3;

use crate::{MdState, bonded_forces, split2_mut, split3_mut, split4_mut};

const EPS_SHAKE_RATTLE: f32 = 1.0e-8;

// SHAKE tolerances for fixed hydrogens. These SHAKE constraints are for fixed hydrogens.
// The tolerance controls how close we get
// to the target value; lower values are more precise, but require more iterations. `SHAKE_MAX_ITER`
// constrains the number of iterations.
const SHAKE_TOL: f32 = 1.0e-4; // Å
const SHAKE_MAX_IT: usize = 100;

impl MdState {
    pub(crate) fn apply_bonded_forces(&mut self) {
        self.apply_bond_stretching_forces();
        self.apply_angle_bending_forces();
        self.apply_dihedral_forces(false);
        self.apply_dihedral_forces(true);
    }

    pub(crate) fn apply_bond_stretching_forces(&mut self) {
        for (indices, params) in &self.force_field_params.bond_stretching {
            let (a_0, a_1) = split2_mut(&mut self.atoms, indices.0, indices.1);

            let (f, energy) =
                bonded_forces::f_bond_stretching(a_0.posit, a_1.posit, params, &self.cell);

            // We divide by mass in `step`.
            a_0.force += f;
            a_1.force -= f;

            // Local virial: Σ r_i · F_i.

            // todo: You already calc min_image and the diff in bonded_forces; consolidate.
            let r_virial = self.cell.min_image(a_1.posit - a_0.posit);
            let virial = r_virial.dot(f); // f is force on atom 0 due to atom 1
            self.barostat.virial.bonded += virial as f64;

            self.potential_energy += energy as f64;
            self.potential_energy_bonded += energy as f64;
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
                bonded_forces::f_angle_bending(a_0.posit, a_1.posit, a_2.posit, params, &self.cell);

            // We divide by mass in `step`.
            a_0.force += f_0;
            a_1.force += f_1;
            a_2.force += f_2;

            // todo: As above, we already calculate this min-image diffs in the force fn above.
            // Use the middle atom (a_1) as the reference.
            // Compute minimum-image displacement vectors for the two bonds in the angle.
            let r10 = self.cell.min_image(a_0.posit - a_1.posit); // vector from 1 -> 0
            let r12 = self.cell.min_image(a_2.posit - a_1.posit); // vector from 1 -> 2

            // Local virial for this 3-body term, using relative coordinates to atom 1.
            let virial = r10.dot(f_0) + r12.dot(f_2);
            // (the a_1 term would be 0 * f_1, so it's omitted)

            self.barostat.virial.bonded += virial as f64;

            self.potential_energy += energy as f64;
            self.potential_energy_bonded += energy as f64;
        }
    }

    /// This maintains dihedral angles. (i.e. the angle between four atoms in a sequence). This models
    /// effects such as σ-bond overlap (e.g. staggered conformations), π-conjugation, which locks certain
    /// dihedrals near 0 or τ, and steric hindrance. (Bulky groups clashing).
    ///
    /// This applies both "proper" linear dihedral angles, and "improper", hub-and-spoke dihedrals. These
    /// two angles are calculated in the same way, but the covalent-bond arrangement of the 4 atoms differs.
    pub(crate) fn apply_dihedral_forces(&mut self, improper: bool) {
        let dihedrals = if improper {
            &self.force_field_params.improper
        } else {
            &self.force_field_params.dihedral
        };

        for (indices, params) in dihedrals {
            // Split the four atoms mutably without aliasing
            let (a_0, a_1, a_2, a_3) =
                split4_mut(&mut self.atoms, indices.0, indices.1, indices.2, indices.3);

            let ((f_0, f_1, f_2, f_3), energy) = bonded_forces::f_dihedral(
                a_0.posit, a_1.posit, a_2.posit, a_3.posit, params, &self.cell,
            );

            // We divide by mass in `step`.
            a_0.force += f_0;
            a_1.force += f_1;
            a_2.force += f_2;
            a_3.force += f_3;

            // let virial =
            //     a_0.posit.dot(f_0) + a_1.posit.dot(f_1) + a_2.posit.dot(f_2) + a_3.posit.dot(f_3);
            //
            // self.barostat.virial.bonded += virial as f64;

            // Reference atom: a_1
            let r10 = self.cell.min_image(a_0.posit - a_1.posit);
            let r12 = self.cell.min_image(a_2.posit - a_1.posit);
            let r13 = self.cell.min_image(a_3.posit - a_1.posit);

            let virial = r10.dot(f_0) + r12.dot(f_2) + r13.dot(f_3);

            self.barostat.virial.bonded += virial as f64;

            self.potential_energy += energy as f64;
            self.potential_energy_bonded += energy as f64;
        }
    }

    // todo: Energy from constrained H!?

    /// This makes the position constraint difference 0; run after drifting positions.
    /// Part of our SHAKE + RATTLE algorithms for fixed hydrogens.
    pub(crate) fn shake_hydrogens(&mut self, dt: f32) {
        // For computing the virial. Store unconstrained positions for atoms involved in constraints
        // (For performance in production, you might want to cache this vector in MdState)
        let mut unconstrained_pos = vec![Vec3::new_zero(); self.atoms.len()];
        for (indices, _) in &self.force_field_params.bond_rigid_constraints {
            unconstrained_pos[indices.0] = self.atoms[indices.0].posit;
            unconstrained_pos[indices.1] = self.atoms[indices.1].posit;
        }

        for _ in 0..SHAKE_MAX_IT {
            let mut max_corr: f32 = 0.0;

            for (indices, (r0_sq, inv_mass)) in &self.force_field_params.bond_rigid_constraints {
                let (ai, aj) = split2_mut(&mut self.atoms, indices.0, indices.1);

                let diff = aj.posit - ai.posit;
                let dist_sq = diff.magnitude_squared();

                // λ = (r² − r₀²) / (2·inv_m · r_ij·r_ij)
                let lambda = (dist_sq - r0_sq) / (2.0 * inv_mass * dist_sq.max(EPS_SHAKE_RATTLE));
                let corr = diff * lambda; // vector correction

                if !ai.static_ {
                    ai.posit += corr / ai.mass;
                }

                if !aj.static_ {
                    aj.posit -= corr / aj.mass;
                }

                max_corr = max_corr.max(corr.magnitude());
            }

            // Converged
            if max_corr < SHAKE_TOL {
                break;
            }
        }

        // todo: Re-evaluate this once the baro and virial work correctly with flexible hydrogens.
        // Calculate constraint virial from the total displacement
        let inv_dt2 = 1.0 / (dt * dt);

        for (indices, _) in &self.force_field_params.bond_rigid_constraints {
            let i = indices.0;
            let j = indices.1;

            let ri_new = self.atoms[i].posit;
            let rj_new = self.atoms[j].posit;

            let dri = ri_new - unconstrained_pos[i];
            let drj = rj_new - unconstrained_pos[j];

            // Approx constraint forces
            let fi_c = dri * (self.atoms[i].mass * inv_dt2);
            // fj_c would be drj * (m_j/dt^2), but we only need one consistently.

            // Use minimum-image bond vector
            let rij = self.cell.min_image(rj_new - ri_new);

            // Pair virial contribution (scalar)
            self.barostat.virial.constraints += rij.dot(fi_c) as f64;
        }
    }

    /// This makes the velocity constraint difference 0; run after updating velocities.
    /// Part of our SHAKE + RATTLE algorithms for fixed hydrogens.
    pub(crate) fn rattle_hydrogens(&mut self, dt: f32) {
        // RATTLE on velocities so that d/dt(|r|²)=0  ⇒  v_ij · r_ij = 0
        for (indices, (_r0_sq, inv_mass)) in &self.force_field_params.bond_rigid_constraints {
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
