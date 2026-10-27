from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

PROJECT_ROOT = Path(__file__).resolve().parents[0]
PYPSA_EUR = PROJECT_ROOT / "pypsa-eur"

network_path = PYPSA_EUR / "results/test-elec/networks/solved_2050.nc"
n = pypsa.Network(network_path)


############################################################
#Scenario 3 : N-1 contingency : 1 Line off
############################################################

n_master = n.copy()
max_normal_thermal_rating = 100
thermal_rating =n.lines.s_nom * n.lines.s_max_pu
n_lines = n.lines.index
min_voltage_violation_pu = 0.90
max_voltage_violation_pu = 1.05


n.optimize()

# transfer the optimized dispatch into the setpoints needed for subsequent power-flow calculations
n.optimize.fix_optimal_dispatch()

#Copying a solved network with an attached solver model is not supported.  delete the model first using `n.model.solver_model = None`.
n.model.solver_model = None

n_opt = n.copy()

for line in n_lines:
    #turning a line off
    n.remove("Line", line)

    # optimizw security constrained: force the linear optimization to respect N-1 line capacities before running the AC power flow validation
    # This ensures that no branch is overloaded even given the branch outages.
    n.optimize.optimize_security_constrained()
    n.optimize.fix_optimal_dispatch()

    #recalculate subnetwork topology after removing a line
    n.determine_network_topology()

    convergence_dict = {}
    for sn in n.sub_networks.obj:

        print(sn.components.buses.static.carrier.unique())
        #skip isolated subnetworks
        if len(sn.components.buses.static) == 1:
            continue

        pf= sn.pf(distribute_slack = True, slack_weights = sn.n.generators.p_nom)

        iterations, errors, converged = pf
        print("all converged" , converged.all())


        convergence = pd.DataFrame({"iterations" : iterations, "errors": errors })
        print(convergence.loc['2013-03-01', "errors"])
        convergence_dict[sn] = convergence


    s0 = np.hypot(n.lines_t.p0, n.lines_t.q0)
    s1 = np.hypot(n.lines_t.p1, n.lines_t.q1)

    smax = np.maximum(s0, s1)
    line_loading = smax.div(thermal_rating, axis=1)*100

    grid_states=[]

    for timestamp in n.snapshots:
        bus_voltage = n.buses_t.v_mag_pu.loc[timestamp]

        min_bus_voltage = bus_voltage.min()
        max_bus_voltage = bus_voltage.max()

        weakest_bus = bus_voltage.idxmin()

        line_load = line_loading.loc[timestamp]

        max_loading_after_outage = line_loading.loc[timestamp].max()

        max_loaded_line = line_loading.loc[timestamp].idxmax()

        number_overloaded_lines = line_load[line_load> max_normal_thermal_rating].count()

        number_voltage_violations = bus_voltage[(bus_voltage < min_voltage_violation_pu)|(bus_voltage > max_voltage_violation_pu)].count()

        grid_states.append({
            "timestamp": timestamp,
            "min_bus_voltage": min_bus_voltage,
            "max_bus_voltage": max_bus_voltage,
            "weakest_bus": weakest_bus,
            "max_loading_after_outage": max_loading_after_outage,
            "max_loaded_line": max_loaded_line,
            "number_overloaded_lines": number_overloaded_lines,
            "number_voltage_violations": number_voltage_violations,
            "convergence_errors" : [convergence_dict[sn].loc[timestamp,"errors"] for sn in n.sub_networks.obj if len(sn.components.buses.static) > 1][0],
            "convergence_iterations" : [convergence_dict[sn].loc[timestamp, "iterations"] for sn in n.sub_networks.obj if len(sn.components.buses.static) > 1][0],
        })

    n1_grid_states = pd.DataFrame(grid_states)
    n1_grid_states.to_csv(f"n_1_contingency_line_{line}_off.csv", index= False)

    n = n_opt.copy()




