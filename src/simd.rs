//! Packing, unpacking, etc.

use std::array::from_fn;

use itertools::Itertools;
use lin_alg::f32::{Vec3x8, Vec3x16, f32x8, f32x16};

use crate::{AtomDynamicsx8, AtomDynamicsx16, MdState};

impl MdState {
    pub fn pack_atoms(&mut self) {
        if is_x86_feature_detected!("avx512f") {
            self.atoms_x16 = Vec::new();

            for chunk in self.atoms.chunks(16) {
                let serial_number = from_fn(|i| chunk.get(i).map(|a| a.serial_number).unwrap_or(0));
                let static_ = from_fn(|i| chunk.get(i).map(|a| a.static_).unwrap_or(false));
                let bonded_only = from_fn(|i| chunk.get(i).map(|a| a.bonded_only).unwrap_or(false));
                let force_field_type = from_fn(|i| {
                    chunk
                        .get(i)
                        .map(|a| a.force_field_type.clone())
                        .unwrap_or_default()
                });
                let element = from_fn(|i| chunk.get(i).map(|a| a.element).unwrap_or_default());

                let px = from_fn(|i| chunk.get(i).map(|a| a.posit.x).unwrap_or(0.0));
                let py = from_fn(|i| chunk.get(i).map(|a| a.posit.y).unwrap_or(0.0));
                let pz = from_fn(|i| chunk.get(i).map(|a| a.posit.z).unwrap_or(0.0));

                let vx = from_fn(|i| chunk.get(i).map(|a| a.vel.x).unwrap_or(0.0));
                let vy = from_fn(|i| chunk.get(i).map(|a| a.vel.y).unwrap_or(0.0));
                let vz = from_fn(|i| chunk.get(i).map(|a| a.vel.z).unwrap_or(0.0));

                let ax = from_fn(|i| chunk.get(i).map(|a| a.accel.x).unwrap_or(0.0));
                let ay = from_fn(|i| chunk.get(i).map(|a| a.accel.y).unwrap_or(0.0));
                let az = from_fn(|i| chunk.get(i).map(|a| a.accel.z).unwrap_or(0.0));

                let mass = from_fn(|i| chunk.get(i).map(|a| a.mass).unwrap_or(0.0));
                let partial_charge =
                    from_fn(|i| chunk.get(i).map(|a| a.partial_charge).unwrap_or(0.0));
                let lj_sigma = from_fn(|i| chunk.get(i).map(|a| a.lj_sigma).unwrap_or(0.0));
                let lj_eps = from_fn(|i| chunk.get(i).map(|a| a.lj_eps).unwrap_or(0.0));

                self.atoms_x16.push(AtomDynamicsx16 {
                    serial_number,
                    static_,
                    bonded_only,
                    force_field_type,
                    element,
                    posit: Vec3x16 {
                        x: f32x16::from_array(px),
                        y: f32x16::from_array(py),
                        z: f32x16::from_array(pz),
                    },
                    vel: Vec3x16 {
                        x: f32x16::from_array(vx),
                        y: f32x16::from_array(vy),
                        z: f32x16::from_array(vz),
                    },
                    accel: Vec3x16 {
                        x: f32x16::from_array(ax),
                        y: f32x16::from_array(ay),
                        z: f32x16::from_array(az),
                    },
                    mass: f32x16::from_array(mass),
                    partial_charge: f32x16::from_array(partial_charge),
                    lj_sigma: f32x16::from_array(lj_sigma),
                    lj_eps: f32x16::from_array(lj_eps),
                });
            }
        } else {
            self.atoms_x8 = Vec::new();

            for chunk in self.atoms.chunks(8) {
                let serial_number = from_fn(|i| chunk.get(i).map(|a| a.serial_number).unwrap_or(0));
                let static_ = from_fn(|i| chunk.get(i).map(|a| a.static_).unwrap_or(false));
                let bonded_only = from_fn(|i| chunk.get(i).map(|a| a.bonded_only).unwrap_or(false));
                let force_field_type = from_fn(|i| {
                    chunk
                        .get(i)
                        .map(|a| a.force_field_type.clone())
                        .unwrap_or_default()
                });
                let element = from_fn(|i| chunk.get(i).map(|a| a.element).unwrap_or_default());

                let px = from_fn(|i| chunk.get(i).map(|a| a.posit.x).unwrap_or(0.0));
                let py = from_fn(|i| chunk.get(i).map(|a| a.posit.y).unwrap_or(0.0));
                let pz = from_fn(|i| chunk.get(i).map(|a| a.posit.z).unwrap_or(0.0));

                let vx = from_fn(|i| chunk.get(i).map(|a| a.vel.x).unwrap_or(0.0));
                let vy = from_fn(|i| chunk.get(i).map(|a| a.vel.y).unwrap_or(0.0));
                let vz = from_fn(|i| chunk.get(i).map(|a| a.vel.z).unwrap_or(0.0));

                let ax = from_fn(|i| chunk.get(i).map(|a| a.accel.x).unwrap_or(0.0));
                let ay = from_fn(|i| chunk.get(i).map(|a| a.accel.y).unwrap_or(0.0));
                let az = from_fn(|i| chunk.get(i).map(|a| a.accel.z).unwrap_or(0.0));

                let mass = from_fn(|i| chunk.get(i).map(|a| a.mass).unwrap_or(0.0));
                let partial_charge =
                    from_fn(|i| chunk.get(i).map(|a| a.partial_charge).unwrap_or(0.0));
                let lj_sigma = from_fn(|i| chunk.get(i).map(|a| a.lj_sigma).unwrap_or(0.0));
                let lj_eps = from_fn(|i| chunk.get(i).map(|a| a.lj_eps).unwrap_or(0.0));

                self.atoms_x8.push(AtomDynamicsx8 {
                    serial_number,
                    static_,
                    bonded_only,
                    force_field_type,
                    element,
                    posit: Vec3x8 {
                        x: f32x8::from_array(px),
                        y: f32x8::from_array(py),
                        z: f32x8::from_array(pz),
                    },
                    vel: Vec3x8 {
                        x: f32x8::from_array(vx),
                        y: f32x8::from_array(vy),
                        z: f32x8::from_array(vz),
                    },
                    accel: Vec3x8 {
                        x: f32x8::from_array(ax),
                        y: f32x8::from_array(ay),
                        z: f32x8::from_array(az),
                    },
                    mass: f32x8::from_array(mass),
                    partial_charge: f32x8::from_array(partial_charge),
                    lj_sigma: f32x8::from_array(lj_sigma),
                    lj_eps: f32x8::from_array(lj_eps),
                });
            }
        }
    }
}
