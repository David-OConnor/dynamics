//! Integration of potentials that evaluate the complete system.

use lin_alg::f32::Vec3;

use crate::{
    AtomDynamics, CHARGE_UNIT_SCALER, ComputationDevice, KCAL_TO_NATIVE, MdConfig, MdState,
    MolDynamics, ParamError, SimBox, params::FfParamSet,
};

impl MdState {
    pub(crate) fn new_system_potential(
        dev: &ComputationDevice,
        cfg: &MdConfig,
        mols: &[MolDynamics],
        set: &FfParamSet,
    ) -> Result<(Self, Vec<MolDynamics>), ParamError> {
        let backend = set.backend.system_potential().unwrap();
        backend.validate_config(cfg)?;
        let mut result = Self {
            cfg: cfg.clone(),
            ..Self::default()
        };

        for mol in mols {
            if mol.bonded_only {
                return Err(ParamError::new(
                    "Whole-system potentials cannot omit interactions for selected atoms",
                ));
            }

            if mol
                .atom_posits
                .as_ref()
                .is_some_and(|p| p.len() != mol.atoms.len())
                || mol
                    .atom_init_velocities
                    .as_ref()
                    .is_some_and(|v| v.len() != mol.atoms.len())
            {
                return Err(ParamError::new(
                    "Coordinates or velocities do not match the atom count",
                ));
            }

            let start = result.atoms.len();
            result.mol_start_indices.push(start);
            let adjacency = crate::util::build_adjacency_list(&mol.atoms, &mol.bonds)?;
            result.adjacency_list.extend(
                adjacency
                    .into_iter()
                    .map(|neighbors| neighbors.into_iter().map(|i| i + start).collect()),
            );

            for (i, atom) in mol.atoms.iter().enumerate() {
                let mass = atom.element.atomic_weight();
                if !mass.is_finite() || mass <= 0. {
                    return Err(ParamError::new("Invalid atom mass"));
                }

                result.atoms.push(AtomDynamics {
                    serial_number: atom.serial_number,
                    static_: mol.static_,
                    element: atom.element,
                    force_field_type: format!(
                        "{}:{}",
                        set.backend.name(),
                        atom.element.to_letter()
                    ),
                    posit: mol.atom_posits.as_ref().map_or(atom.posit, |p| p[i]).into(),
                    vel: mol
                        .atom_init_velocities
                        .as_ref()
                        .map_or(Vec3::new_zero(), |v| v[i]),
                    mass,
                    partial_charge: atom.partial_charge.unwrap_or(0.) * CHARGE_UNIT_SCALER,
                    ..Default::default()
                });
                result.mass_accel_factor.push(KCAL_TO_NATIVE / mass);
            }
        }
        if result.atoms.is_empty() {
            return Err(ParamError::new(
                "No atoms selected for the system potential",
            ));
        }

        let positions: Vec<_> = result.atoms.iter().map(|a| a.posit).collect();
        result.cell = SimBox::from_solute_atoms(&positions, &cfg.sim_box);
        result.solute_atom_count = result.atoms.len();
        result.thermo_dof = result.dof_for_thermo();
        result.system_potential = Some(backend.create(&result.atoms, &result.cell, mols)?);
        result.apply_all_forces(dev, &None);
        result.check_force_error()?;

        if let Some(iterations) = cfg.max_init_relaxation_iters {
            result.minimize_energy(dev, iterations, None);
            result.check_force_error()?;
            result.reset_f_acc_pe_virial();
            result.apply_all_forces(dev, &None);
            result.check_force_error()?;
        }
        for (atom, &factor) in result.atoms.iter_mut().zip(&result.mass_accel_factor) {
            atom.accel = atom.force * factor;
        }

        Ok((result, Vec::new()))
    }

    /// A backend error stops integration until the state is rebuilt.
    pub fn check_force_error(&self) -> Result<(), ParamError> {
        match &self.force_error {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    /// Current backend charges in elementary-charge units.
    pub fn atom_charge(&self, index: usize) -> Option<f32> {
        self.atoms
            .get(index)
            .map(|a| a.partial_charge / CHARGE_UNIT_SCALER)
    }

    pub(crate) fn apply_system_potential(&mut self, external: &Option<Vec<Vec3>>) {
        if self.force_error.is_some() {
            return;
        }

        if self.alchemical.mol_idx.is_some() {
            self.force_error = Some(ParamError::new(
                "Alchemical scaling is unavailable for this system potential",
            ));
            self.potential_energy = f64::NAN;

            return;
        }

        let positions: Vec<_> = self.atoms.iter().map(|a| a.posit).collect();
        let result = self.system_potential.as_mut().unwrap().evaluate(&positions);
        match result {
            Ok(result) => {
                if result.forces.len() != self.atoms.len()
                    || result.charges.as_ref().is_some_and(|q| {
                        q.len() != self.atoms.len() || q.iter().any(|v| !v.is_finite())
                    })
                    || !result.energy.is_finite()
                    || result
                        .forces
                        .iter()
                        .flat_map(|f| f.to_arr())
                        .any(|v| !v.is_finite())
                {
                    self.force_error = Some(ParamError::new("Invalid system-potential result"));
                    self.potential_energy = f64::NAN;

                    return;
                }

                for (atom, force) in self.atoms.iter_mut().zip(result.forces) {
                    atom.force = force;
                }
                if let Some(charges) = result.charges {
                    for (atom, charge) in self.atoms.iter_mut().zip(charges) {
                        atom.partial_charge = charge * CHARGE_UNIT_SCALER;
                    }
                }
                if let Some(external) = external {
                    if external.len() != self.atoms.len() {
                        self.force_error = Some(ParamError::new(
                            "External forces do not match the atom count",
                        ));

                        return;
                    }

                    for (atom, force) in self.atoms.iter_mut().zip(external) {
                        atom.force += *force;
                    }
                }
                self.potential_energy = result.energy;
                self.potential_components = result.components;
                // Whole-system terms need not have a classical bonded/nonbonded split.
                self.potential_energy_bonded = f64::NAN;
                self.potential_energy_nonbonded = f64::NAN;
                self.potential_energy_between_mols =
                    vec![f64::NAN; self.mol_start_indices.len().pow(2)];
            }
            Err(error) => {
                self.force_error = Some(error);
                self.potential_energy = f64::NAN;
            }
        }
    }
}
