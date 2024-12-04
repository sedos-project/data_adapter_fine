import dataclasses
import logging
import math
import numpy as np
import pandas as pd
import pyomo.environ as pyomo


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
        self.scenario = 'transport_test'
        self.bev_constraint_data = {}

        self.structure = Structure(
            structure_name,
            process_sheet=process_sheet,
            helper_sheet=helper_sheet,
        )
        self.adapter = Adapter(
            self.collection_name,
            structure=self.structure,
            units=list(utils.standard_units.values()),
        )
        self.commodities = self.get_commodities()
        self.commodityUnitDict = {
            commodity: utils.get_commodity_unit(commodity)
            for commodity in self.commodities
        }
        self.esM = self.init_esM()

        self.esM.add(
            fn.Source(
                esM=self.esM,
                name='slack source',
                commodity='exo_road_mcar_pkm',
                hasCapacityVariable=False,
                opexPerOperation=1000,
            )
        )

        # self.esM.add(
        #     fn.Source(
        #         esM=self.esM,
        #         name='slack source gasoline',
        #         commodity='sec_gasoline',
        #         hasCapacityVariable=False,
        #     )
        # )

        self.processes = {
            process: self.adapter.get_process(process)
            for process in list(self.structure.processes.keys())
            if process != 'x2x_x2liquid_oref_1' #TODO: remove
        }
        # false_units = {item for item in units.items() if item[1] not in self.units}
        # if len(false_units) > 0:
        #     logging.warning(f"The following units do not match the specified adapter units. '{false_units}'")
        self.add_processes_to_esM()
        print('All processes added to esM')

    def add_processes_to_esM(self):
        for process in self.processes.items():
            for unit in process[1].units.items():
                if unit[1] not in utils.standard_units:
                    logging.warning(f"Unit {unit} for process {process[0]} not found in standard units.")

            inputs, outputs = process[1].inputs, process[1].outputs
            if len(inputs) == 0 or len(outputs) == 0:
                constructor = SourceSinkConstructor(process, self.esM)
            elif len(inputs) == 1 and inputs == outputs:
                constructor = StorageConstructor(process, self.esM)
            else:
                op_timeseries = None
                if process[0].startswith('tra_') and any(output.startswith('exo_road') for output in outputs):
                    tra_ts_names = [output for output in outputs if output.startswith('exo_road')]
                    if len(tra_ts_names) == 1:
                        raw_ts = self.processes['helper_sink_' + tra_ts_names[0]].timeseries['demand_timeseries']
                        tra_timeseries_dict = {
                            year: raw_ts[raw_ts.index.year == year].squeeze().values
                            for year in raw_ts.index.year.unique()
                        }
                        op_timeseries = pd.DataFrame(tra_timeseries_dict)
                    else:
                        raise ValueError(f"Multiple exo demands are specified in {process[0]}.")

                constructor = ConversionConstructor(process, self.esM, self.adapter, op_timeseries)

            if process[0].startswith('tra_road_') and process[0].endswith('1') and 'bev_pass_engine' in process[0]:
                self.get_data_for_bev_constraints(constructor)
                constructor.col_set = constructor.col_set.difference(['share_tra_charge_mode'])

            if len(constructor.col_set) > 0:
                raise ValueError(f"Unused columns in {process[0]}: {constructor.col_set}")
            for comp in constructor.comps:
                self.esM.add(comp)
            print(f'Added {process[0]} to esM')
            for key, value in constructor.fine_args.items():
                print(f'{key}: {value}')

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
            costUnit=utils.standard_units['cost'],
            lengthUnit="km",
            verboseLogLevel=0,
        )

        return esM

    def get_data_for_bev_constraints(self, constructor):
        name_split = constructor.fine_args['name'].split('_')
        if name_split[2] not in self.bev_constraint_data.keys():
            self.bev_constraint_data[name_split[2]] = pd.DataFrame()
        self.bev_constraint_data[name_split[2]]['_'.join(name_split[6:8])] = constructor.interpolate_df(
            constructor.scalars['share_tra_charge_mode'],
            self.esM
        )

    def add_co2_sink(self):
        self.esM.add(
            fn.Sink(
                esM=self.esM,
                name='co2_sink',
                commodity='co2_equivalent',
                hasCapacityVariable=False,
            )
        )
        emi_co2_dict = {commod: -1 for commod in self.commodities if commod.startswith('emi_co2')}
        emi_ch4_dict = {commod: -1 / 28 for commod in self.commodities if commod.startswith('emi_ch4')}
        emi_n2o_dict = {commod: -1 / 265 for commod in self.commodities if commod.startswith('emi_n2o')}
        emi_dict = {**emi_co2_dict, **emi_ch4_dict, **emi_n2o_dict}
        self.esM.add(
            fn.Conversion(
                esM=self.esM,
                name='co2_conversion_helper',
                physicalUnit=self.esM.commodityUnitsDict['co2_equivalent'],
                commodityConversionFactors={
                    'co2_equivalent': 1,
                    'sectoral_emi': emi_dict,
                },
                hasCapacityVariable=False,
            )
        )
        print('Added co2 sink and conversion to esM')

    def add_slack_sinks(self):
        sink_commodities = [
            comp.commodity
            for comp in self.esM.componentModelingDict['SourceSinkModel'].componentsDict.values()
            if isinstance(comp, fn.Sink)
        ]
        for commodity in self.commodities.difference(sink_commodities):
            if commodity in utils.slack_sink_opex.keys():
                opex = utils.slack_sink_opex[commodity]
            else:
                opex = 1
            self.esM.add(
                fn.Sink(
                    esM=self.esM,
                    name='slack_sink_'+commodity,
                    commodity=commodity,
                    hasCapacityVariable=False,
                    # opexPerOperation=opex,
                )
            )
        print('Added slack sinks to esM')

    def check_slacks(self, exclusion_list = []):
        for ip in self.esM.investmentPeriodNames:
            results = self.esM.getOptimizationSummary('SourceSinkModel', ip=ip)
            slack_results = results.loc[
                                results.index.get_level_values(0).str.startswith('slack_sink')
                            ].loc[:, 'operation', :]
            index = slack_results[slack_results['Germany'] != 0].index
            for comp_name in {ind[0] for ind in index}:
                if comp_name not in exclusion_list:
                    logging.warning(f"Slack sink {comp_name} has non-zero operation in {ip}.")

    def declare_bev_constraints(self):
        for car_class in self.bev_constraint_data.keys():
            self.bev_constraint_data[car_class] = (
                self.bev_constraint_data[car_class][['flex_uni', 'flex_bi']].div(
                    self.bev_constraint_data[car_class]['infl_uni'], axis=0)
            )
        commisVar = self.esM.pyM.commis_conv
        loc = list(self.esM.locations)[0]

        def bev_commissioning_constraint(pyM, car_class, bev_type, ip):
            infl_name = f'tra_road_{car_class}_bev_pass_engine_infl_uni_1'
            flex_name = f'tra_road_{car_class}_bev_pass_engine_flex_{bev_type}_1'

            return (
                    commisVar[loc, flex_name, ip]
                    == commisVar[loc, infl_name, ip]
                    * self.bev_constraint_data[car_class]['flex_' + bev_type].loc[self.esM.investmentPeriodNames[ip]]
            )

        self.esM.pyM.bev_commissioning_constraint = pyomo.Constraint(
            self.bev_constraint_data.keys(),
            ['uni', 'bi'],
            self.esM.investmentPeriods,
            rule=bev_commissioning_constraint
        )

    def optimize(self):
        self.add_co2_sink()
        self.add_slack_sinks()
        self.esM.aggregateTemporally(numberOfTypicalPeriods=1, numberOfTimeStepsPerPeriod=24)
        self.esM.declareOptimizationProblem(timeSeriesAggregation=True)
        self.declare_bev_constraints()
        self.esM.optimize(
            declaresOptimizationProblem=False,
            timeSeriesAggregation=True,
            solver="gurobi"
        )
        self.check_slacks(
            exclusion_list=[
                'slack_sink_sec_kerosene_fos_orig',
                'slack_sink_sec_refinery_gas',
                'slack_sink_sec_heat_low'
            ]
        )

    def export_results(self):
        results_df = pd.DataFrame()
        if any(
                comp.flexibleConversion
                for comp in self.esM.componentModelingDict['ConversionModel'].componentsDict.values()
        ):
            self.op_flex_opt = {
                ip: fn.utils.formatOptimizationOutput(
                    self.esM.pyM.op_flex_conv.get_values(),
                    "operationVariables",
                    "1dim",
                    self.esM.investmentPeriodNames.index(ip),
                    self.esM.periodsOrder[self.esM.investmentPeriodNames.index(ip)],
                    esM=self.esM,
                ).sum(axis=1)
                for ip in self.esM.investmentPeriodNames
            }
        for model_name, model in self.esM.componentModelingDict.items():
            model_results = model._optSummary.copy()
            for comp_name in model.componentsDict.keys():
                if 'slack' not in comp_name:
                    continue
                comp_df = self.get_comp_results(
                    comp_name,
                    {ip: df.loc[comp_name] for ip, df in model_results.items()}
                )
                results_df = pd.concat([results_df, comp_df], ignore_index=True)
        results_df = results_df.reindex(
            columns=['scenario', 'process', 'new', 'parameter', 'sector', 'category', 'specification',
                     'year', 'output_groups', 'input_groups', 'unit', 'value']
        )
        results_df.to_csv('output/test_results.csv')#, index=False)

    def get_comp_results(self, comp_name, model_results):
        comp_df = pd.DataFrame()
        comp_results_base = {
            'scenario': [self.scenario],
            'process': [comp_name],
        }
        comp = self.esM.getComponent(comp_name)
        comp_name_split = comp_name.split('_')
        if len(comp_name_split) > 1:
            comp_results_base['sector'] = [comp_name_split[0]]
            comp_results_base['category'] = [comp_name_split[1]]
            comp_results_base['specification'] = [comp_name_split[2:-1]]
        if comp_name[-1] == '1':
            comp_results_base['new'] = [True]
        elif comp_name[-1] == '0':
            comp_results_base['new'] = [False]
        else:
            comp_results_base['new'] = [None]
        for ip in self.esM.investmentPeriodNames:
            comp_results = comp_results_base.copy()
            comp_results['year'] = [ip]
            params = {
                'capacity_inst': 'capacity',
                'capacity_new': 'commissioning',
                'costs_investment': 'capexCap',
                'costs_fixed': 'opexCap',
                'costs_variable': 'opexOp',
            }
            for param_sedos, param_fine in params.items():
                comp_results['parameter'] = [param_sedos]
                if (
                        param_sedos.startswith('capacity')
                        and comp_name.startswith('tra_')
                        and any(output.startswith('exo_road_') for output in self.processes[comp_name].outputs)
                ):
                    comp_results['unit'] = [utils.standard_units['vehicles']]
                else:
                    comp_results['unit'] = [model_results[ip].loc[param_fine].index.values[0][1:-1]]
                comp_results['value'] = [model_results[ip].loc[param_fine].values[0][0]]
                comp_df = pd.concat([comp_df, pd.DataFrame(comp_results)])
            comp_results['parameter'] = ['flow_volume']
            comp_results['value'] = [model_results[ip].loc['operation'].values[0][0]]
            if isinstance(comp, fn.Source):
                comp_results['output_groups'] = [comp.commodity]
                comp_results['unit'] = self.esM.commodityUnitsDict[comp.commodity]
                comp_df = pd.concat([comp_df, pd.DataFrame(comp_results)])
            elif isinstance(comp, fn.Sink):
                comp_results['input_groups'] = [comp.commodity]
                comp_results['unit'] = self.esM.commodityUnitsDict[comp.commodity]
                comp_df = pd.concat([comp_df, pd.DataFrame(comp_results)])
            elif isinstance(comp, fn.Conversion):
                for commod, value in comp.commodityConversionFactors[ip].items():
                    comp_results_flow = comp_results.copy()
                    if isinstance(value, dict):
                        group = commod
                        for commod, ccf in value.items():
                            comp_results_flow = comp_results.copy()
                            comp_results_flow['value'] = (
                                    self.op_flex_opt[ip].loc[comp_name, group, commod].values[0] * abs(ccf)
                            )
                            if ccf > 0:
                                comp_results_flow['output_groups'] = [commod]
                            elif ccf < 0:
                                comp_results_flow['input_groups'] = [commod]
                            comp_results_flow['unit'] = self.esM.commodityUnitsDict[commod]
                            comp_df = pd.concat([comp_df, pd.DataFrame(comp_results_flow)])
                        continue
                    elif value > 0:
                        comp_results_flow['output_groups'] = [commod]
                    elif value < 0:
                        comp_results_flow['input_groups'] = [commod]
                    comp_results_flow['unit'] = self.esM.commodityUnitsDict[commod]
                    comp_results_flow['value'] = [comp_results_flow['value'][0] * abs(value)]
                    comp_df = pd.concat([comp_df, pd.DataFrame(comp_results_flow)])

        return comp_df


class ComponentConstructor:
    def __init__(self, process, esM, adapter=None):
        self.fine_args = {
            'name': process[0],
        }
        self.comps = []

        self.scalars = process[1].scalars.set_index('year')
        self.scalars = utils.drop_unused_columns(self.scalars)
        self.scalars = self.scalars.round(5)
        # self.scalars = self.scalars.replace('global_scalars.wacc', 0.05) #TODO: remove
        self.col_set = set(self.scalars.columns)
        if not process[1].timeseries.empty:
            self.timeseries = {}
            raw_ts = process[1].timeseries
            for col in raw_ts.columns:
                ts_dict = {
                    year: raw_ts[raw_ts.index.year == year][col].squeeze().values
                    for year in raw_ts.index.year.unique()
                }
                self.timeseries[col[0]] = self.interpolate_df(pd.DataFrame(ts_dict).T, esM).T
                self.col_set.add(col[0])
        else:
            self.timeseries = None


        param_columns = [col for col in self.scalars if col in utils.param_mapping.keys()]
        if len(param_columns) > 0:
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
                logging.warning(f"Missing lifetime for {process[0]}. Default value 7 is used.")#TODO: change into error?
                self.fine_args['economicLifetime'] = esM.investmentPeriodInterval
                if self.fine_args['name'] in ['x2x_delivery_methane_pipeline_0']:  # TODO: remove
                    self.fine_args['economicLifetime'] = 56
            if 'wacc' in self.scalars.columns:
                self.fine_args['interestRate'] = self.scalars['wacc'].astype(float).mean() / 100 #TODO: remove .astype(float)?

            if process[0].startswith('tra_road') and "motorc" not in process[0]:
                self.fine_args['floorTechnicalLifetime'] = False

            if process[0].endswith('_0'):
                self.fine_args['commissioningFix'] = dict.fromkeys(esM.investmentPeriodNames, 0)

            if 'capacityFix' in param_df.columns and '_0' in process[0]:
                interval = esM.investmentPeriodInterval
                if 'floorTechnicalLifetime' in self.fine_args.keys():
                    rounded_lifetime = math.ceil(self.fine_args['economicLifetime'] / interval) * interval
                else:
                    rounded_lifetime = math.floor(self.fine_args['economicLifetime'] / interval) * interval
                stock_commissioning = param_df['capacityFix'].copy()
                stock_commissioning.index = stock_commissioning.index - rounded_lifetime
                stock_commissioning = stock_commissioning.diff().abs().dropna()
                if esM.startYear in stock_commissioning.index:
                    self.fine_args['commissioningFix'][esM.startYear] = float(stock_commissioning.loc[esM.startYear])
                stock_commissioning = stock_commissioning[stock_commissioning.index < esM.startYear]
                self.fine_args['stockCommissioning'] = stock_commissioning.to_dict()
                if len(param_df['capacityFix'][param_df['capacityFix']>0]) > rounded_lifetime / interval:
                    if self.fine_args['name'] in ['x2x_delivery_methane_pipeline_0']: #TODO: remove
                        logging.warning(f"Stock commissioning for {process[0]} exceeds economic lifetime.")
                    else:
                        raise ValueError(f"Stock commissioning for {process[0]} exceeds economic lifetime.")
                for param in ['investPerCapacity', 'opexPerCapacity']:
                    if param in self.fine_args.keys():
                        for stock_ip in stock_commissioning.index:
                            self.fine_args[param][stock_ip] = self.fine_args[param][esM.startYear]
                del self.fine_args['capacityFix']
            self.col_set = self.col_set.difference(param_columns + ['wacc'])

        if self.timeseries is not None:
            ts_columns = [col for col in self.timeseries if col in utils.timeseries_mapping.keys()]
            for col in ts_columns:
                self.fine_args[utils.timeseries_mapping[col]] = self.timeseries[col].to_dict(orient='series')
                self.col_set = self.col_set.difference(ts_columns)

        if process[0].startswith('tra_road_') and process[0].endswith('1') and 'bev' in process[0]:
            name_split = process[0].split('_')
            self.fine_args['linkedQuantityID'] = '_'.join(name_split[1:3] + name_split[6:8])



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
    def __init__(self, process, esM, adapter, op_timeseries=None):
        ComponentConstructor.__init__(self, process, esM, adapter)
        if op_timeseries is not None:
            op_timeseries = self.interpolate_df(op_timeseries.T, esM).T
            op_timeseries = op_timeseries / op_timeseries.sum() #TODO: remove when timeseries is updated
            self.fine_args['operationRateMax'] = op_timeseries.to_dict(orient='series')


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

        self.fine_args['physicalUnit'] = 'MW'
        if (
            self.fine_args['name'].startswith('tra_')
            and any(output.startswith('exo_road') for output in process[1].outputs)
        ):
            self.fine_args['commodityConversionFactors'] = utils.calc_tra_ccf(
                self.fine_args['commodityConversionFactors'],
                self.interpolate_df(self.scalars, esM)
            )
            self.fine_args['physicalUnit'] = utils.standard_units['transport_pass_demand']
            # Hardcoded scaling to prevent numerical issues #TODO: move or remove
            # for arg in ['operationRateMax', 'investPerCapacity', 'opexPerCapacity', 'opexPerOperation', 'capacityMax', 'capacityMin', 'commissioningFix', 'stockCommissioning']:
            #     if arg == 'operationRateMax':
            #         factor = 1000
            #     else:
            #         factor = 1 / 1000
            #     if arg in self.fine_args.keys():
            #         for ip in esM.investmentPeriodNames:
            #             self.fine_args[arg][ip] = self.fine_args[arg][ip] * factor
            self.col_set = self.col_set.difference({'market_share_range', 'mileage', 'occupancy_rate', 'tonnage'})
            if process[0].endswith('_0'):
                self.col_set = self.col_set.difference({'share_tra_charge_mode'})

        self.comps.append(
            fn.Conversion(
                esM=esM,
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
        self.col_set = self.col_set.difference(self.check_ccf_columns())

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
                        physicalUnit='MW', #TODO change
                        commodityConversionFactors=ccf,
                        hasCapacityVariable=False,
                    )
                )
            else:
                raise ValueError(f"Process {process[0]} has no inputs and no outputs.")
        elif len(process[1].outputs) == 0:
            if len(process[1].inputs) > 1:
                raise ValueError(f"Process {process[0]} has to many inputs. Only one input per sink is allowed.")
            if 'demand_timeseries' in self.timeseries.keys():
                ts = self.timeseries['demand_timeseries'] / self.timeseries['demand_timeseries'].sum() #TODO: remove when timeseries is updated/ normalized
                self.fine_args['operationRateFix'] = (
                        ts * self.interpolate_df(self.scalars['demand_annual'], esM)
                ).to_dict(orient='series')
                self.col_set = self.col_set.difference(['demand_annual', 'demand_timeseries'])
            self.comps.append(
                fn.Sink(
                    esM=esM,
                    commodity=process[1].inputs[0],
                    hasCapacityVariable=False,
                    **self.fine_args
                )
            )

    def check_ccf_columns(self):
        ccf_cols = []
        for col in self.col_set:
            if col.startswith('conversion_factor'):
                if (self.scalars[col] == 1).all():
                    ccf_cols.append(col)
                else:
                    raise ValueError(f"Conversion factors for Source/ Sink must be equal to 1. "
                                     f"Error in {self.fine_args['name']} ")
        return ccf_cols


class StorageConstructor(ComponentConstructor):
    def __init__(self, process, esM):
        ComponentConstructor.__init__(self, process, esM)

        for param in utils.storage_param_mapping.items():
            if param[0] in self.scalars.columns:
                self.fine_args[param[1]] = self.scalars[param[0]].mean() / 100
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

        for ts in utils.storage_timeseries_mapping.items():
            if self.timeseries is not None and ts[0] in self.timeseries.keys():
                self.fine_args[ts[1]] = self.timeseries[ts[0]].to_dict(orient='series')
                self.col_set = self.col_set.difference([ts[0]])

        self.comps.append(
            fn.Storage(
                esM=esM,
                commodity=process[1].inputs[0],
                hasCapacityVariable=True,
                **self.fine_args
            )
        )

