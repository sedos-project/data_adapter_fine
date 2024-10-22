import dataclasses
import logging
import math
import numpy as np
import pandas as pd


import fine as fn
from data_adapter import databus
from data_adapter.preprocessing import Adapter
from data_adapter.structure import Structure

import utils

@dataclasses.dataclass
class DataAdapter:
    def __init__(
            self,
            url: str,
            structure_name: str,
            process_sheet: str,
            helper_sheet: str,
            downloadData: bool = True,
    ):
        if downloadData:
            databus.download_collection(url)
        self.collection_name = url.split('/')[-1]

        self.structure = Structure(
            structure_name,
            process_sheet=process_sheet,
            helper_sheet=helper_sheet,
        )
        self.adapter = Adapter(
            self.collection_name,
            structure=self.structure,
        )
        self.commodities = self.get_commodities()
        self.commodityUnitDict = {commodity: 'missing' for commodity in self.commodities}
        self.esM = self.init_esM()

        # self.processes = {}
        # process_error_list = []
        # for process in list(self.structure.processes.keys()):
        #     try:
        #         self.processes[process] = self.adapter.get_process(process)
        #     except:
        #         print(f"Error while loading {process}")
        #         process_error_list.append(process)


        self.processes = {
            process: self.adapter.get_process(process)
            for process in list(self.structure.processes.keys())
        }
        self.add_processes_to_esM()
        print('test')

    def add_processes_to_esM(self):
        for process in self.processes.items():

            if process[0] in [
                'x2x_p2gas_sabm_1',
                'x2x_x2liquid_oref_0',
                'x2x_p2gas_biom_1',
                'x2x_other_cng_compression',
                'x2x_other_lng_liquefication',
                'x2x_delivery_methane_pipeline_0',
                'x2x_storage_methane_0', #TODO Hendrik
                # 'x2x_g2p_pemfc_ls_1', #TODO Hendrik fragen: Warum wird scalars nicht geladen? self.adapter.get_process('x2x_g2p_pemfc_ls_1') falsche version
                # 'x2x_g2p_sofc_ls_1', #TODO Hendrik fragen: Warum wird scalars nicht geladen? self.adapter.get_process('x2x_g2p_pemfc_ls_1') falsche version
                'x2x_x2gas_mpyr_1', #TODO Hendrik fragen: capacity_p_abs_new_max fehlt, df beschnitten
            ]: #TODO remove
                continue

            inputs, outputs = process[1].inputs, process[1].outputs
            if len(inputs) == 0 or len(outputs) == 0:
                constructor = SourceSinkConstructor(process, self.esM)
            elif len(inputs) == 1 and inputs == outputs:
                constructor = StorageConstructor(process, self.esM)
            else:
                constructor = ConversionConstructor(process, self.esM)

            if len(constructor.col_set) > 0:
                raise ValueError(f"Unused columns in {process[0]}: {constructor.col_set}")
            for comp in constructor.comps:
                self.esM.add(comp)
            print(f'Added {process[0]} to esM')

    def get_commodities(self):
        commodities = set(utils.helper_commodities)
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
        self.fine_args = {
            'name': process[0],
        }
        self.comps = []

        self.scalars = process[1].scalars.set_index('year')
        self.scalars = utils.drop_unused_columns(self.scalars)
        self.scalars = self.scalars.replace('global_scalars.wacc', 0.05)
        self.col_set = set(self.scalars.columns)

        param_columns = [col for col in self.scalars if col in utils.param_mapping.keys()]
        param_df = self.interpolate_df(self.scalars[param_columns], esM)
        if 'capacity_e_max'in param_df.columns: #TODO: adapt for storage
            param_df['capacity_e_max'] = param_df['capacity_e_max']/8760
        param_df.rename(columns=utils.param_mapping, inplace=True)
        param_df = self.set_tech_availability(param_df, esM)
        if param_df.columns.duplicated().any():
            raise ValueError(f"Duplicate columns in {process[0]} scalars.")
        self.fine_args = self.fine_args | param_df.to_dict('dict')
        if 'lifetime' in self.scalars.columns:
            self.fine_args['economicLifetime'] = math.floor(self.scalars['lifetime'].mean())
        else:
            self.fine_args['economicLifetime'] = esM.investmentPeriodInterval
        if 'wacc' in self.scalars.columns:
            self.fine_args['interestRate'] = self.scalars['wacc'].mean()

        if 'capacityFix' in param_df.columns and '_0' in process[0]:
            interval = esM.investmentPeriodInterval
            rounded_lifetime = math.floor(self.fine_args['economicLifetime'] / interval) * interval
            stock_commissioning = param_df['capacityFix'].copy()
            stock_commissioning.index = stock_commissioning.index - rounded_lifetime
            stock_commissioning = stock_commissioning[stock_commissioning.index < esM.startYear]
            self.fine_args['stockCommissioning'] = stock_commissioning.to_dict()
            for param in ['investPerCapacity', 'opexPerCapacity']:
                if param in self.fine_args.keys():
                    for stock_ip in stock_commissioning.index:
                        self.fine_args[param][stock_ip] = self.fine_args[param][esM.startYear]
            del self.fine_args['capacityFix']
        self.col_set = self.col_set.difference(param_columns + ['lifetime', 'wacc'])



    def set_tech_availability(self, param_df, esM):
        if param_df.isna().any().any():
            self.fine_args['commissioningFix'] = {
                ip: 0 if ip in param_df[param_df.isna().any(axis=1)].index.tolist()
                else None
                for ip in esM.investmentPeriodNames
            }
        return param_df.fillna(0)

    def interpolate_df(self, df, esM):
        for year in pd.Index(esM.investmentPeriodNames).difference(self.scalars.index):
            df.loc[year] = np.nan
        df = df.sort_index().astype('float64')
        df = df.interpolate(method='index')
        return df.loc[esM.investmentPeriodNames]


class ConversionConstructor(ComponentConstructor):
    def __init__(self, process, esM):
        ComponentConstructor.__init__(self, process, esM)

        # self.scalars = self.scalars.replace('global_emission_factors.gasoline', 2)  # TODO remove when global parameters are set
        # self.scalars = self.scalars.replace('global_emission_factors.bioethanol', 2)  # TODO remove when global parameters are set
        # self.scalars = self.scalars.replace('global_scalars.wacc', 0.05)  # TODO remove when global parameters are set

        commodities_raw = process[1].inputs + process[1].outputs
        input_commodities = [
            commodity
            for item in process[1].inputs
            for commodity in (item if isinstance(item, list) else [item])
        ]
        output_commodities = [
            commodity
            for item in process[1].outputs
            for commodity in (item if isinstance(item, list) else [item])
        ]
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

        ccf_columns = [col for col in self.scalars if col.startswith('conversion_factor')]
        self.col_set = self.col_set.difference(ccf_columns)
        ccf_df = self.interpolate_df(self.scalars[ccf_columns], esM)
        ccf_df.rename(columns=lambda x: x[18:], inplace=True)
        for commodity in input_commodities + output_commodities:
            if commodity not in ccf_df.columns and not commodity.startswith('emi'):
                ccf_df[commodity] = 1
                if commodity in commodities_no_group:
                    if self.fine_args['name'] in ['x2x_x2liquid_oref_1']:  # TODO: remove
                        print('remove') # TODO: remove
                    else: # TODO: remove
                        raise ValueError(f"Conversion factor for {commodity} not found in {self.fine_args['name']}.")
                logging.warning(f"Conversion factor for {commodity} not found in {self.fine_args['name']}. "
                                f"Default value 1 is used.")
        ccf_df[input_commodities] = ccf_df[input_commodities] * -1

        self.fine_args['commodityConversionFactors'] = ccf_df[commodities_no_group].to_dict('index')
        for idx, group in enumerate(commodity_groups):
            for ip in esM.investmentPeriodNames:
                self.fine_args['commodityConversionFactors'][ip]['group' + str(idx)] = ccf_df[group].loc[ip].to_dict()

        ef_columns = [col for col in self.scalars if col.startswith('ef')]
        if len(ef_columns) > 0:
            ef_df = self.interpolate_df(self.scalars[ef_columns], esM)
            ef_df.rename(columns=lambda x: x[3:], inplace=True)

            if len(commodity_groups) > 0:
                self.fine_args['emissionFactors'] = {}
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
            self.col_set = self.col_set.difference(ef_columns)

        if any(col.startswith('flow_share') for col in self.scalars.columns):
            self.fine_args['flowShares'] = self.get_flow_shares(esM)

        self.comps.append(
            fn.Conversion(
                esM=esM,
                physicalUnit='missing',#TODO: change
                **self.fine_args
            )
        )

    def get_flow_shares(self, esM):
        fs_columns = [col for col in self.scalars if col.startswith('flow_share')]
        fs_df = self.interpolate_df(self.scalars[fs_columns], esM)
        if (fs_df < 1).all().all():
            factor = 1
        else:
            factor = 100
            logging.warning("Flow shares should be given in percent. Values are divided by 100.")

        flow_shares = dict.fromkeys(esM.investmentPeriodNames)
        for ip in esM.investmentPeriodNames:
            flow_shares[ip] = {'min': {}, 'max': {}}
            for col in fs_df.columns:
                if 'min' in col:
                    flow_shares[ip]['min'][col[15:]] = fs_df[col].loc[ip] / factor
                elif 'max' in col:
                    flow_shares[ip]['max'][col[15:]] = fs_df[col].loc[ip] / factor
                else:
                    raise ValueError(f"Flow share column {col} must contain 'min' or 'max'.")#TODO: check for fix values
        self.col_set = self.col_set.difference(fs_columns)
        return flow_shares

class SourceSinkConstructor(ComponentConstructor):
    def __init__(self, process, esM):
        ComponentConstructor.__init__(self, process, esM)

        if len(process[1].inputs) == 0:
            if len(process[1].outputs) == 1:
                self.comps.append(
                    fn.Source(
                        esM=esM,
                        commodity=process[1].outputs[0],
                        hasCapacityVariable=True,
                        **self.fine_args
                    )
                )
            elif len(process[1].outputs) > 1:
                # special case for multi output source processes
                self.comps.append(
                    fn.Source(
                        esM=esM,
                        commodity=process[0] + "_helper",
                        hasCapacityVariable=True,
                        **self.fine_args
                    )
                )
                ccf = {process[0] + "_helper": -1}
                ef_columns = [col for col in self.scalars if col.startswith('ef')]
                self.col_set = self.col_set.difference(ef_columns)
                for col in ef_columns:
                    emission_factors = process[1].scalars[col].unique()
                    if len(emission_factors) != 1:
                        raise ValueError(f"Emission factors for {process[0]} must be equal for all investment periods.")
                    ccf['emi' + col.split('_emi')[1]] = emission_factors[0]

                self.comps.append(
                    fn.Conversion(
                        esM=esM,
                        name=process[0] + "_conv_helper",
                        physicalUnit='missing',
                        commodityConversionFactors=ccf,
                        hasCapacityVariable=False,
                    )
                )
            else:
                raise ValueError(f"Process {process[0]} has no inputs and no outputs.")
        elif len(process[1].outputs) == 0:
            self.comps.append(
                fn.Sink(
                    esM=esM,
                    commodity=process[1].inputs[0],
                    hasCapacityVariable=False,
                    **self.fine_args
                )
            )

class StorageConstructor(ComponentConstructor):
    def __init__(self, process, esM):
        ComponentConstructor.__init__(self, process, esM)

        for param in utils.storage_param_mapping.items():
            if param[0] in self.scalars.columns:
                if (self.scalars[param[0]] < 1).all():
                    factor = 1
                else:
                    factor = 100
                    logging.warning("Storage efficiencies should be given in percent. Values are divided by 100.")
                self.fine_args[param[1]] = self.scalars[param[0]].mean() / factor
                self.col_set = self.col_set.difference([param[0]])

        for ep_ratio in ['sto_ep_ratio_binding', 'sto_ep_ratio_optional']:
            if ep_ratio in self.scalars.columns:
                if 'chargeRate' in self.fine_args.keys():
                    raise ValueError(f"Storage process {process[0]} has binding and optional charge rate.")
                self.fine_args['chargeRate'] = 1 / self.scalars[ep_ratio].mean()
                self.fine_args['dischargeRate'] = 1 / self.scalars[ep_ratio].mean()
                self.col_set = self.col_set.difference([ep_ratio])

        if 'opexPerOperation' in self.fine_args.keys():
            self.fine_args['opexPerChargeOperation'] = self.fine_args['opexPerOperation']
            del self.fine_args['opexPerOperation']

        self.comps.append(
            fn.Storage(
                esM=esM,
                commodity=process[1].inputs[0],
                hasCapacityVariable=True,
                **self.fine_args
            )
        )

