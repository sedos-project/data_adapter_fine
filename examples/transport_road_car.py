import os

os.environ["COLLECTIONS_DIR"] = "../collections/"
os.environ["STRUCTURES_DIR"] = "../structures"

from data_adapter.preprocessing import Adapter
from data_adapter.structure import Structure

from data_adapter_fine import DataAdapter


da = DataAdapter(
    url="https://databus.openenergyplatform.org/johannes_beh/collections/johannes_sedos_emob",
    downloadData=False
)
da.get_data_from_collection(structure_name="tra", process_sheet="Processes_tra_road_car")

da.build_esM()


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

