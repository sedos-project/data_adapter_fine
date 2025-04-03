import os

os.environ["COLLECTIONS_DIR"] = "../collections/"
os.environ["STRUCTURES_DIR"] = "../structures"

from data_adapter_fine import DataAdapter
from data_adapter_fine import postprocessing


veh_class = 'mcar'

da = DataAdapter(
    url="https://databus.openenergyplatform.org/johannes_beh/collections/sedos_fine_mcar",
    downloadData=False,
    structure_name="structure",
    process_sheet="sys_" + veh_class,
    helper_sheet="helper_sys_" + veh_class,
    veh_class=veh_class,
    scenario_name='f_tra_tokio'
)


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

