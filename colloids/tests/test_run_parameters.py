from dataclasses import fields
import os
from openmm import unit
import pytest
from colloids.abstracts import Parameters
from colloids.run_parameters import RunParameters


class TestQuantity(object):
    @pytest.mark.parametrize("openmm_quantity",
                             [1.0 * unit.meter,
                              2.0 * (unit.nano * unit.meter),
                              -1.0 * ((unit.nano * unit.meter) ** 2),
                              1.0 / unit.second,
                              -1.0 * ((unit.pico * unit.second) ** -1),
                              -2.0 / ((unit.pico * unit.second) ** 2),
                              (3.0 * (unit.milli * unit.volt) * (unit.micro * unit.ampere)
                               / (unit.angstrom ** 2 * (unit.nano * unit.second))),
                              (-3.0 * (unit.mega * unit.angstrom) / (unit.milli * unit.second)),
                              (3.0 * (unit.mega * unit.angstrom) / ((unit.milli * unit.second) * unit.volt)),
                              (12.0 * (unit.mega * unit.angstrom) / (unit.milli * unit.second) * unit.volt)])
    def test_quantity(self, openmm_quantity):
        new_openmm_quantity = Parameters._Quantity(openmm_quantity).to_openmm_quantity()
        # Using new_openmm_quantity == openmm_quantity directly can fail because openmm uses floating conversion factors
        # in this equality comparison that can lead to small differences in the values.
        assert (new_openmm_quantity.value_in_unit(openmm_quantity.unit)
                == pytest.approx(openmm_quantity.value_in_unit(openmm_quantity.unit), rel=1e-12, abs=1e-12))


class TestRunParameters(object):
    @pytest.fixture
    def parameters(self):
        return RunParameters(initial_configuration="first_frame.xyz")

    @pytest.fixture(autouse=True)
    def change_test_dir(self, request, monkeypatch):
        # Change the working directory to the directory of the test file.
        # See https://stackoverflow.com/questions/62044541/change-pytest-working-directory-to-test-case-directory
        monkeypatch.chdir(request.fspath.dirname)

    @pytest.fixture
    def yaml_file(self, parameters):
        parameters.to_yaml("test.yaml")
        yield "test.yaml"
        os.remove("test.yaml")

    @pytest.fixture
    def yaml_parameters(self, yaml_file):
        return RunParameters.from_yaml(yaml_file)

    def test_run_parameters(self, parameters, yaml_parameters):
        # Because we cannot compare openmm quantities directly (see above), we have to compare all fields explicitly.
        # When new fields are added to the RunParameters dataclass, this test must be updated accordingly.
        assert len(fields(parameters)) == len(fields(yaml_parameters)) == 45
        assert parameters.initial_configuration == yaml_parameters.initial_configuration
        assert len(parameters.masses) == len(yaml_parameters.masses)
        assert len(parameters.radii) == len(yaml_parameters.radii)
        assert len(parameters.surface_potentials) == len(yaml_parameters.surface_potentials)
        for t in parameters.masses:
            assert t in yaml_parameters.masses
            assert (parameters.masses[t].value_in_unit(parameters.masses[t].unit)
                    == pytest.approx(yaml_parameters.masses[t].value_in_unit(parameters.masses[t].unit), rel=1e-12,
                                     abs=1e-12))
        for t in parameters.radii:
            assert t in yaml_parameters.radii
            assert (parameters.radii[t].value_in_unit(parameters.radii[t].unit)
                    == pytest.approx(yaml_parameters.radii[t].value_in_unit(parameters.radii[t].unit), rel=1e-12,
                                     abs=1e-12))
        for t in parameters.surface_potentials:
            assert t in yaml_parameters.surface_potentials
            assert (parameters.surface_potentials[t].value_in_unit(parameters.surface_potentials[t].unit)
                    == pytest.approx(
                        yaml_parameters.surface_potentials[t].value_in_unit(parameters.surface_potentials[t].unit),
                        rel=1e-12, abs=1e-12))

        assert parameters.platform_name == yaml_parameters.platform_name
        assert (parameters.potential_temperature.value_in_unit(parameters.potential_temperature.unit)
                == pytest.approx(
                    yaml_parameters.potential_temperature.value_in_unit(parameters.potential_temperature.unit),
                    rel=1e-12, abs=1e-12))
        assert (parameters.brush_density.value_in_unit(parameters.brush_density.unit)
                == pytest.approx(yaml_parameters.brush_density.value_in_unit(parameters.brush_density.unit), rel=1e-12,
                                 abs=1e-12))
        assert (parameters.brush_length.value_in_unit(parameters.brush_length.unit)
                == pytest.approx(yaml_parameters.brush_length.value_in_unit(parameters.brush_length.unit), rel=1e-12,
                                 abs=1e-12))
        assert (parameters.debye_length.value_in_unit(parameters.debye_length.unit)
                == pytest.approx(yaml_parameters.debye_length.value_in_unit(parameters.debye_length.unit), rel=1e-12,
                                 abs=1e-12))
        assert parameters.dielectric_constant == yaml_parameters.dielectric_constant
        assert parameters.use_log == yaml_parameters.use_log
        assert parameters.use_tabulated == yaml_parameters.use_tabulated
        assert parameters.integrator == yaml_parameters.integrator
        assert parameters.integrator_parameters == yaml_parameters.integrator_parameters
        assert parameters.velocity_seed == yaml_parameters.velocity_seed
        assert parameters.run_steps == yaml_parameters.run_steps
        assert parameters.state_data_interval == yaml_parameters.state_data_interval
        assert parameters.state_data_filename == yaml_parameters.state_data_filename
        assert parameters.trajectory_interval == yaml_parameters.trajectory_interval
        assert parameters.trajectory_filename == yaml_parameters.trajectory_filename
        assert parameters.checkpoint_interval == yaml_parameters.checkpoint_interval
        assert parameters.checkpoint_filename == yaml_parameters.checkpoint_filename
        assert parameters.minimize_energy_initially == yaml_parameters.minimize_energy_initially
        assert parameters.final_configuration_gsd_filename == yaml_parameters.final_configuration_gsd_filename
        assert parameters.final_configuration_xyz_filename == yaml_parameters.final_configuration_xyz_filename
        assert parameters.cutoff_factor == yaml_parameters.cutoff_factor
        assert all(pw == yw for pw, yw in zip(parameters.wall_directions, yaml_parameters.wall_directions))
        assert parameters.alpha == yaml_parameters.alpha
        assert parameters.epsilon == yaml_parameters.epsilon
        assert parameters.use_depletion == yaml_parameters.use_depletion
        assert parameters.depletion_phi == yaml_parameters.depletion_phi
        assert parameters.depletant_radius == yaml_parameters.depletant_radius
        assert parameters.use_gravity == yaml_parameters.use_gravity
        assert parameters.gravitational_acceleration == yaml_parameters.gravitational_acceleration
        assert parameters.water_density == yaml_parameters.water_density
        assert parameters.particle_density == yaml_parameters.particle_density
        assert parameters.update_reporter == yaml_parameters.update_reporter
        assert parameters.update_reporter_parameters == yaml_parameters.update_reporter_parameters
        assert parameters.substrate_type == yaml_parameters.substrate_type
        assert parameters.substrate_particle_type == yaml_parameters.substrate_particle_type
        assert parameters.use_snowman == yaml_parameters.use_snowman
        assert parameters.snowman_seed == yaml_parameters.snowman_seed
        assert parameters.snowman_bond_types == yaml_parameters.snowman_bond_types
        assert parameters.snowman_distances == yaml_parameters.snowman_distances


if __name__ == '__main__':
    pytest.main([__file__])
