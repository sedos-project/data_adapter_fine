from data_adapter.preprocessing import Adapter
from data_adapter.structure import Structure
from data_adapter import databus
import pathlib

structure = Structure(
    "SEDOS_Modellstruktur",
    process_sheet="Processes_steel_industry"
)
adapter = Adapter(
    "steel_industry_test",
    structure=structure,
)


collections_folder = pathlib.Path(__file__).parent / "collections"
# url = "https://databus.openenergyplatform.org/felixmaur/collections/steel_industry_test"
url = "https://databus.openenergyplatform.org/johannes_beh/collections/johannes_sedos_emob"
# url = "https://databus.openenergyplatform.org/nailend/collections/emob_test_collection_1"
databus.download_collection(url, collections_folder)
adapter.get_process('exo_steel')

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

