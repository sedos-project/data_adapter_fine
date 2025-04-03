import os
import utils
import datetime

if os.getcwd().endswith('data_adapter_fine'):
    os.environ["COLLECTIONS_DIR"] = "./collections/"
    os.environ["STRUCTURES_DIR"] = "./structures"
else:
    os.environ["COLLECTIONS_DIR"] = "../collections/"
    os.environ["STRUCTURES_DIR"] = "../structures"

from data_adapter_fine import DataAdapter
from data_adapter_fine import postprocessing

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

veh_class = 'mcar'

da = DataAdapter(
    url="https://databus.openenergyplatform.org/johannes_beh/collections/sedos_fine_mcar",
    downloadData=False,
    structure_name="structure",
    process_sheet="sys_" + veh_class,
    helper_sheet="helper_sys_" + veh_class,
    veh_class=veh_class,
)


da.esM.updateComponent('x2x_storage_hydrogen_lohc_1', {'opexPerChargeOperation': 0, 'opexPerCapacity': 0})
da.esM.updateComponent('x2x_storage_hydrogen_new_1', {'opexPerChargeOperation': 0, 'opexPerCapacity': 0})
da.esM.updateComponent('x2x_storage_hydrogen_retrofit_1', {'opexPerChargeOperation': 0, 'opexPerCapacity': 0})
da.esM.updateComponent('x2x_g2p_h2_fuel_cell_1_ag', {'opexPerOperation': 0.082})


utils.check_coefficents(da.esM)

da.optimize(numberOfTypicalPeriods=24)
results = da.export_results()

# create plots
postprocessing.plot_bev_operation(da, ip=2063)
postprocessing.plot_bev_costs_revenue(da)
postprocessing.plot_tac(da)
postprocessing.plot_capacities(da, veh_class, results)
postprocessing.plot_pri_energy(da, veh_class)
postprocessing.plot_operation_stacked(da, results)
postprocessing.plot_co2(da, results)

