import dataclasses

import fine as fn
from data_adapter import databus
from data_adapter.preprocessing import Adapter
from data_adapter.structure import Structure


@dataclasses.dataclass
class DataAdapter:
    def __init__(
            self,
            url: str,
            downloadData: bool = True,
    ):
        if downloadData:
            databus.download_collection(url)
        self.collection_name = url.split('/')[-1]

    def get_data_from_collection(
            self,
            structure_name: str,
            process_sheet: str,
    ):
        self.structure = Structure(
            structure_name,
            process_sheet=process_sheet
        )
        self.adapter = Adapter(
            self.collection_name,
            structure=self.structure,
        )

        self.commodities = set()
        for process in self.structure.processes.values():
            process_commodities = process['inputs'] + process['outputs']
            for commodities in process_commodities:
                if isinstance(commodities, list):
                    self.commodities.update(commodities)
                else:
                    self.commodities.add(commodities)

        self.commodityUnitDict = {commodity: 'kW' for commodity in self.commodities}
        print(list(self.structure.processes.keys())[0:2])
        self.processes = {
            process: self.adapter.get_process(process)
            for process in list(self.structure.processes.keys())[0:2]
        }

    def build_esM(self):
        self.esM = fn.EnergySystemModel(
            locations={'Germany'},
            commodities=self.commodities,
            numberOfTimeSteps=8760,
            commodityUnitsDict=self.commodityUnitDict,
            hoursPerTimeStep=1,
            numberOfInvestmentPeriods=7,
            investmentPeriodInterval=7,
            costUnit="1e6 Euro",
            lengthUnit="km",
            verboseLogLevel=0,
        )
        print('test')