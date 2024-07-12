import dataclasses
import pandas as pd
import numpy as np
import math

import fine as fn
from data_adapter import databus
from data_adapter.preprocessing import Adapter
from data_adapter.structure import Structure


@dataclasses.dataclass
class DataAdapter:
    def __init__(
            self,
            url: str,
            structure_name: str,
            process_sheet: str,
            downloadData: bool = True,
    ):
        if downloadData:
            databus.download_collection(url)
        self.collection_name = url.split('/')[-1]

        self.structure = Structure(
            structure_name,
            process_sheet=process_sheet
        )
        self.adapter = Adapter(
            self.collection_name,
            structure=self.structure,
        )
        self.commodities = self.get_commodities()
        self.commodityUnitDict = {commodity: 'kW' for commodity in self.commodities}
        self.esM = self.init_esM()
        self.processes = {
            process: self.adapter.get_process(process)
            for process in list(self.structure.processes.keys())
        }
        self.add_processes_to_esM()
        print('test')

    def add_processes_to_esM(self):
        for process in self.processes.items():
            inputs, outputs = process[1].inputs, process[1].outputs
            if len(inputs) == 0:
                raise NotImplementedError()
            elif len(outputs) == 0:
                raise NotImplementedError()
            elif len(inputs) == 1 and inputs == outputs:
                raise NotImplementedError()
            else:
                constructor = ConversionConstructor(process, self.esM)

            self.esM.add(constructor.comp)
            #fine_args = self.get_fine_args_from_process(process)

    def get_fine_args_from_process(self, process):

        param_mapping = {
            'cost_fix_tra': 'opexPerCapacity',
            'cost_inv_tra': 'investPerCapacity',
            'cost_var_tra': 'opexPerOperation',
        }
        def interpolate_df(df):
            for year in pd.Index(self.esM.investmentPeriodNames).difference(scalars.index):
                df.loc[year] = np.nan
            df = df.sort_index().astype('float64')
            df = df.interpolate(method='index')
            return df.loc[self.esM.investmentPeriodNames]

        fine_args = {
            'name': process[0],
            'emissionFactors': {},
        }

        scalars = process[1].scalars.set_index('year')
        scalars = scalars.replace('global_emission_factors.gasoline', 2) # TODO remove when global parameters are set
        scalars = scalars.replace('global_emission_factors.bioethanol', 2) # TODO remove when global parameters are set
        scalars = scalars.replace('global_scalars.wacc', 0.05) # TODO remove when global parameters are set

        commodities_raw = process[1].inputs + process[1].outputs
        input_commodities = [
            commodity
            for item in process[1].inputs
            for commodity in (item if isinstance(item, list) else [item])
        ]

        ccf_columns = [col for col in scalars if col.startswith('conversion_factor')]
        ccf_df = interpolate_df(scalars[ccf_columns])
        ccf_df.rename(columns=lambda x: x[18:], inplace=True)
        ccf_df[input_commodities] = ccf_df[input_commodities] * -1

        commodity_groups = [
            group
            for group in commodities_raw
            if isinstance(group, list)
        ]
        commodities_no_group = [
            commod
            for commod in commodities_raw
            if not isinstance(commod, list)
            if not commod.startswith('emi')
        ]
        fine_args['commodityConversionFactors'] = ccf_df[commodities_no_group].to_dict('index')
        for idx, group in enumerate(commodity_groups):
            for ip in self.esM.investmentPeriodNames:
                fine_args['commodityConversionFactors'][ip]['group' + str(idx)] = ccf_df[group].loc[ip].to_dict()

        ef_columns = [col for col in scalars if col.startswith('ef')]
        ef_df = interpolate_df(scalars[ef_columns])
        ef_df.rename(columns=lambda x: x[3:], inplace=True)

        for col in ef_df.columns:
            emi_commodity = 'emi' + col.split('_emi')[1]
            if col.split('_emi')[0] in commodities_no_group:
                for ip in self.esM.investmentPeriodNames:
                    if emi_commodity not in fine_args['commodityConversionFactors'][ip].keys():
                        fine_args['commodityConversionFactors'][ip][emi_commodity] = 0
                    fine_args['commodityConversionFactors'][ip][emi_commodity] += (
                            ef_df[col] * abs(ccf_df[col.split('_emi')[0]])
                    ).loc[ip]
            else:
                if not (ef_df[col] == ef_df[col].iloc[0]).all():
                    raise ValueError("Emission factors must be equal for all investment periods. "
                                     f"Please check values for {fine_args['name']}.")
                if emi_commodity in fine_args['emissionFactors'].keys():
                    fine_args['emissionFactors'][emi_commodity][col.split('_emi')[0]] = ef_df[col].iloc[0]
                else:
                    fine_args['emissionFactors'][emi_commodity] = {col.split('_emi')[0]: ef_df[col].iloc[0]}


        return fine_args


    def get_commodities(self):
        commodities = set()
        for process in self.structure.processes.values():
            process_commodities = process['inputs'] + process['outputs']
            for commod in process_commodities:
                if isinstance(commod, list):
                    commodities.update(commod)
                else:
                    commodities.add(commod)
        return commodities

    def init_esM(self):
        esM = fn.EnergySystemModel(
            locations={'Germany'},
            commodities=self.commodities,
            numberOfTimeSteps=8760,
            commodityUnitsDict=self.commodityUnitDict,
            hoursPerTimeStep=1,
            startYear=2021,
            numberOfInvestmentPeriods=8,
            investmentPeriodInterval=7,
            costUnit="1e6 Euro",
            lengthUnit="km",
            verboseLogLevel=0,
        )

        return esM

class ComponentConstructor:
    def __init__(self, process, esM):

        self.param_mapping = {
            'cost_fix_tra': 'opexPerCapacity',
            'cost_inv_tra': 'investPerCapacity',
            'cost_var_tra': 'opexPerOperation',
        }

        self.fine_args = {
            'name': process[0],
        }

        self.scalars = process[1].scalars.set_index('year')

        cost_columns = [col for col in self.scalars if col in self.param_mapping.keys()]
        cost_df = self.interpolate_df(self.scalars[cost_columns], esM)
        cost_df.rename(columns=self.param_mapping, inplace=True)
        self.fine_args = self.fine_args | cost_df.to_dict('dict')
        self.fine_args['economicLifetime'] = math.floor(self.scalars['lifetime'].mean())

    def interpolate_df(self, df, esM):
        for year in pd.Index(esM.investmentPeriodNames).difference(self.scalars.index):
            df.loc[year] = np.nan
        df = df.sort_index().astype('float64')
        df = df.interpolate(method='index')
        return df.loc[esM.investmentPeriodNames]


class ConversionConstructor(ComponentConstructor):
    def __init__(self, process, esM):
        ComponentConstructor.__init__(self, process, esM)
        self.fine_args['emissionFactors'] = {}

        self.scalars = self.scalars.replace('global_emission_factors.gasoline', 2)  # TODO remove when global parameters are set
        self.scalars = self.scalars.replace('global_emission_factors.bioethanol', 2)  # TODO remove when global parameters are set
        self.scalars = self.scalars.replace('global_scalars.wacc', 0.05)  # TODO remove when global parameters are set

        commodities_raw = process[1].inputs + process[1].outputs
        input_commodities = [
            commodity
            for item in process[1].inputs
            for commodity in (item if isinstance(item, list) else [item])
        ]

        ccf_columns = [col for col in self.scalars if col.startswith('conversion_factor')]
        ccf_df = self.interpolate_df(self.scalars[ccf_columns], esM)
        ccf_df.rename(columns=lambda x: x[18:], inplace=True)
        ccf_df[input_commodities] = ccf_df[input_commodities] * -1

        commodity_groups = [
            group
            for group in commodities_raw
            if isinstance(group, list)
        ]
        commodities_no_group = [
            commod
            for commod in commodities_raw
            if not isinstance(commod, list)
            if not commod.startswith('emi')
        ]
        self.fine_args['commodityConversionFactors'] = ccf_df[commodities_no_group].to_dict('index')
        for idx, group in enumerate(commodity_groups):
            for ip in esM.investmentPeriodNames:
                self.fine_args['commodityConversionFactors'][ip]['group' + str(idx)] = ccf_df[group].loc[ip].to_dict()

        ef_columns = [col for col in self.scalars if col.startswith('ef')]
        ef_df = self.interpolate_df(self.scalars[ef_columns], esM)
        ef_df.rename(columns=lambda x: x[3:], inplace=True)

        for col in ef_df.columns:
            emi_commodity = 'emi' + col.split('_emi')[1]
            if col.split('_emi')[0] in commodities_no_group:
                for ip in esM.investmentPeriodNames:
                    if emi_commodity not in self.fine_args['commodityConversionFactors'][ip].keys():
                        self.fine_args['commodityConversionFactors'][ip][emi_commodity] = 0
                    self.fine_args['commodityConversionFactors'][ip][emi_commodity] += (
                            ef_df[col] * abs(ccf_df[col.split('_emi')[0]])
                    ).loc[ip]
            else:
                if not (ef_df[col] == ef_df[col].iloc[0]).all():
                    raise ValueError("Emission factors must be equal for all investment periods. "
                                     f"Please check values for {self.fine_args['name']}.")
                if emi_commodity in self.fine_args['emissionFactors'].keys():
                    self.fine_args['emissionFactors'][emi_commodity][col.split('_emi')[0]] = ef_df[col].iloc[0]
                else:
                    self.fine_args['emissionFactors'][emi_commodity] = {col.split('_emi')[0]: ef_df[col].iloc[0]}

        self.comp = fn.Conversion(
            esM=esM,
            physicalUnit='kW', # TODO change
            **self.fine_args
        )

        print('test')
