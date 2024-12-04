import os
from fine.IOManagement.standardIO import writeOptimizationOutputToExcel
from fine.IOManagement.xarrayIO import writeEnergySystemModelToNetCDF, readNetCDFtoEnergySystemModel

os.environ["COLLECTIONS_DIR"] = "../collections/"
os.environ["STRUCTURES_DIR"] = "../structures"

from data_adapter_fine import DataAdapter

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# da = DataAdapter(
#     url="https://databus.openenergyplatform.org/johannes_beh/collections/sedos_fine",
#     downloadData=False,
#     structure_name="structure_fine",
#     process_sheet="tra_road_mcar",
#     helper_sheet="helper_tra_road_mcar",
# )

da = DataAdapter(
   url="https://databus.openenergyplatform.org/johannes_beh/collections/sedos_fine",
   downloadData=False,
   structure_name="structure_fine",
   process_sheet="tra_road_mcar_bev",
   helper_sheet="helper_buffer",
)


# da = DataAdapter(
#     url="https://databus.openenergyplatform.org/johannes_beh/collections/sedos_fine_x2x",
#     # downloadData=False,
#     structure_name="tra_v2",
#     process_sheet="x2x",
#     helper_sheet="helper_dummy",
# )

da.optimize()
da.export_results()
writeOptimizationOutputToExcel(da.esM, "output/fine_sedos_mcar", optSumOutputLevel=2, optValOutputLevel=1)
writeEnergySystemModelToNetCDF(da.esM, "output/fine_sedos_mcar.nc")
print('test')
# from data_adapter import databus
# from modelBuilder import *
# import pathlib
# import FINE as fn
#
#
# df1 = preprocessing.get_process("hack-a-thon", 'modex_tech_wind_turbine_onshore', 'hack-a-thon_links')
# df2 = preprocessing.get_process("hack-a-thon", 'modex_demand', 'hack-a-thon_links')
#
# collections_folder = pathlib.Path(__file__).parent / "collections"
# url = "https://energy.databus.dbpedia.org/felixmaur/collections/hack-a-thon/"
# # databus.download_collection(url,collections_folder)
#
# collection_dict = structure.get_energy_structure('hack-a-thon_by_file_name')
#
# esM_dict = create_energy_system_dict()
#
# esM = build_model(esM_dict)

