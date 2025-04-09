import os
import dataclasses
import logging
import numpy as np
import pandas as pd
import pyomo.environ as pyomo
import re
from scipy.interpolate import interp1d

import fine as fn
from data_adapter import databus
from data_adapter.preprocessing import Adapter
from data_adapter.structure import Structure

from data_adapter_fine import utils
from data_adapter_fine.constructor import *

@dataclasses.dataclass
class DataAdapter:
    def __init__(
            self,
            url: str,
            structure_name: str,
            process_sheet: str,
            helper_sheet: str,
            scenario_name: str,
            downloadData: bool = True,
            veh_class: str = '',
    ):
        self.op_flex_opt = None
        if downloadData:
            databus.download_collection(url)
        self.collection_name = url.split('/')[-1]
        self.scenario = scenario_name
        self.bev_constraint_data = {}
        self.constructor_dict = {}

        self.market_source_df = pd.DataFrame()
        self.market_sink_df = pd.DataFrame()

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

        if veh_class != '':
            self.esM.add(
                fn.Source(
                    esM=self.esM,
                    name='slack_source_' + veh_class + '_demand',
                    commodity='exo_road_' + veh_class + '_pkm',
                    hasCapacityVariable=False,
                    opexPerOperation=10000,
                )
            )

        self.add_electricity_market()

        self.processes = {
            process: self.adapter.get_process(process)
            for process in list(self.structure.processes.keys())
        }

        self.add_processes_to_esM()
        print('All processes added to esM')

    def add_electricity_market(self):
        electricity_costs = pd.read_csv(
            os.environ["COLLECTIONS_DIR"] + 'electricity_data/DE00_Mgl Cost.csv', usecols=['2035','2040','2050']
        ).T
        electricity_costs.index = electricity_costs.index.astype(int)
        for year in pd.Index(self.esM.investmentPeriodNames).difference(electricity_costs.index):
            electricity_costs.loc[year] = np.nan

        electricity_costs_sum = electricity_costs.sum(axis=1)
        electricity_costs_sum = electricity_costs_sum.replace(0,  np.nan).sort_index()#.interpolate(method='index', limit_area='inside')
        f = interp1d(
            electricity_costs_sum.dropna().index.values,
            electricity_costs_sum.dropna().values,
            fill_value='extrapolate'
        )
        electricity_costs_sum.loc[electricity_costs_sum.index] = f(electricity_costs_sum.index)
        for ip in self.esM.investmentPeriodNames:
            if ip <= 2035:
                electricity_costs.loc[ip] = (
                        electricity_costs.loc[2035]
                        / electricity_costs_sum[2035]
                        * electricity_costs_sum[ip]
                )
            elif ip < 2050:
                electricity_costs.loc[ip] = (
                        electricity_costs.loc[2040]
                        / electricity_costs_sum[2040]
                        * electricity_costs_sum[ip]
                )
            else:
                electricity_costs.loc[ip] = (
                        electricity_costs.loc[2050]
                        / electricity_costs_sum[2050]
                        * electricity_costs_sum[ip]
                )

        electricity_costs = electricity_costs / 1000
        electricity_costs_with_grid = electricity_costs + 0.082
        electricity_costs = electricity_costs.loc[self.esM.investmentPeriodNames].T.to_dict(orient='series')
        electricity_costs_with_grid = electricity_costs_with_grid.loc[self.esM.investmentPeriodNames].T.to_dict(orient='series')
        self.esM.add(
            fn.Source(
                esM=self.esM,
                name='electricity_market_source',
                commodity='sec_elec',
                hasCapacityVariable=False,
                commodityCostTimeSeries=electricity_costs_with_grid,
                interestRate=0.02,
            )
        )
        self.esM.add(
            fn.Sink(
                esM=self.esM,
                name='electricity_market_sink',
                commodity='sec_elec',
                hasCapacityVariable=False,
                commodityRevenueTimeSeries=electricity_costs,
                interestRate=0.02,
                operationRateMax=100,
            )
        )


    def add_processes_to_esM(self):
        for process in self.processes.items():
            for unit in process[1].units.items():
                if unit[1] not in utils.standard_units.values():
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

                # check if process has a deviating main commodity (other than first output)
                # the necessary dats is extracted from the Parameter Input Output sheet of the model structure file
                if process[0] in self.structure.parameters.keys():
                    # check if inputs and outputs are the same for each item of the processes parameters
                    parameters = self.structure.parameters[process[0]]
                    if any(
                            param['inputs'] != list(parameters.values())[0]['inputs']
                            or param['outputs'] != list(parameters.values())[0]['outputs']
                            for param in parameters.values()
                    ):
                        raise ValueError(f"Commodities of {process[0]} are not the same in Parameter Input Output sheet.")
                    inputs = list(parameters.values())[0]['inputs']
                    outputs = list(parameters.values())[0]['outputs']

                    if len(inputs) > 1 or len(outputs) > 1:
                        raise ValueError(f"Multiple inputs or outputs are defined for {process[0]} "
                                         f"in the Parameter_Input-Output sheet.")

                    if outputs != [''] and inputs == ['']:
                        main_commodity = outputs[0]
                    elif outputs == [''] and inputs != ['']:
                        main_commodity = inputs[0]
                    else:
                        raise ValueError(f"Main commodity for {process[0]} could not be determined.")
                else:
                    main_commodity = None

                constructor = ConversionConstructor(process, self.esM, op_timeseries, main_commodity)

            if process[0].startswith('tra_road_') and process[0].endswith('1') and re.search('bev_.*_engine', process[0]):
                self.get_data_for_bev_constraints(constructor)
                constructor.col_set = constructor.col_set.difference(['share_tra_charge_mode'])

            if len(constructor.col_set) > 0:
                raise ValueError(f"Unused columns in {process[0]}: {constructor.col_set}")
            for comp in constructor.comps:
                self.esM.add(comp)
            self.constructor_dict[process[0]] = constructor
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
        # calculate emission budget
        # emission reduction targets for Germany according to Umweltbundesamt
        # https://www.umweltbundesamt.de/daten/klima/treibhausgasminderungsziele-deutschlands#projektionsdaten-2024
        # assuming net-zero emissions in 2045
        emission_budget = pd.Series({2021: 144.4, 2030: 82, 2045: 0})
        # scale emission budget to mcars (in 2021 63.38 Mt co2 emissions according to stock)
        emission_budget = emission_budget * 63.38 / 144.4
        for year in range(2028, 2077, 7):
            emission_budget[year] = np.nan
        emission_budget = emission_budget.interpolate(method='index').drop([2030, 2045])

        balance_limit = {
            ip: pd.DataFrame(
                index=["emissions"],
                columns=["Germany", "lowerBound"],
                data=[[-emission_budget[ip], True]]
            )
            for ip in emission_budget.index
        }

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
            balanceLimit=balance_limit,
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
                balanceLimitID='emissions',
            )
        )
        #include co2 source to ensure operation of dac components
        self.esM.add(
            fn.Source(
                esM=self.esM,
                name='co2_source',
                commodity='co2_equivalent',
                hasCapacityVariable=False,
                balanceLimitID='emissions',
            )
        )
        self.esM.add(
            fn.Sink(
                esM=self.esM,
                name='co2_stored_sink',
                commodity='emi_co2_stored',
                hasCapacityVariable=False,
            )
        )

        emi_co2_dict = {
            commod: -1 for commod in self.commodities
            if commod.startswith('emi_co2')
            and not 'neg' in commod
            and not commod in ['emi_co2_stored', 'emi_co2_reusable']
        }
        # scale unit for CH4 emissions in the x2x and tra sector by 1e-3 for smaller coefficients
        emi_ch4_dict = {commod: -1 / 28 * 1e3 for commod in self.commodities if commod.startswith('emi_ch4')}
        # scale unit for N2O emissions in the x2x and tra sector by 1e-6 for smaller coefficients
        emi_n2o_dict = {commod: -1 / 265 * 1e6 for commod in self.commodities if commod.startswith('emi_n2o')}
        emi_dict = {**emi_co2_dict, **emi_ch4_dict, **emi_n2o_dict}

        self.esM.add(
            fn.Conversion(
                esM=self.esM,
                name='co2_conversion_helper',
                physicalUnit=self.esM.commodityUnitsDict['co2_equivalent'],
                commodityConversionFactors={
                    ip: {'co2_equivalent': 1, 'sectoral_emi': emi_dict}
                    for ip in self.esM.investmentPeriodNames
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
            if 'emi' in commodity:
                continue
            if commodity in utils.slack_sink_opex.keys():
                opex = utils.slack_sink_opex[commodity]
            else:
                opex = 10
            self.esM.add(
                fn.Sink(
                    esM=self.esM,
                    name='slack_sink_'+commodity,
                    commodity=commodity,
                    hasCapacityVariable=False,
                    opexPerOperation=opex,
                )
            )
            self.esM.add(
                fn.Source(
                    esM=self.esM,
                    name='slack_source_'+commodity,
                    commodity=commodity,
                    hasCapacityVariable=False,
                    opexPerOperation=100000,
                )
            )

        print('Added slack sinks to esM')

    def check_slacks(self, exclusion_list=[]):
        for ip in self.esM.investmentPeriodNames:
            results = self.esM.getOptimizationSummary('SourceSinkModel', ip=ip)
            slack_results = results.loc[
                                results.index.get_level_values(0).str.startswith('slack_')
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
        commisVar = self.esM.pyM.cap_conv
        loc = list(self.esM.locations)[0]

        def bev_capacity_constraint(pyM, car_class, bev_type, ip):
            infl_name = f'tra_road_{car_class}_bev_pass_engine_infl_uni_1'
            flex_name = f'tra_road_{car_class}_bev_pass_engine_flex_{bev_type}_1'

            return (
                    commisVar[loc, flex_name, ip]
                    == commisVar[loc, infl_name, ip]
                    * self.bev_constraint_data[car_class]['flex_' + bev_type].loc[self.esM.investmentPeriodNames[ip]]
            )

        self.esM.pyM.bev_capacity_constraint = pyomo.Constraint(
            self.bev_constraint_data.keys(),
            ['uni', 'bi'],
            self.esM.investmentPeriods,
            rule=bev_capacity_constraint
        )

        opVarStor = self.esM.pyM.chargeOp_stor
        opVarConv = self.esM.pyM.op_conv

        def bev_battery_constraint(pyM, car_class, bev_type, ip, p, t):
            if bev_type.endswith('0') or bev_type.endswith('1'):
                new = bev_type[-1]
                bev_type = bev_type[:-2]
            else:
                new = '1'
            batt_name = f'tra_road_{car_class}_bev_pass_battery_{bev_type}_{new}'
            wallbox_name = f'tra_road_{car_class}_bev_pass_wallbox_{bev_type}_g2v_{new}'
            charge_efficiency = self.esM.getComponent(batt_name).chargeEfficiency
            wallbox_ccf = self.esM.getComponent(wallbox_name).processedCommodityConversionFactors[ip]
            wallbox_efficiency = abs(wallbox_ccf[f'sec_elec_{car_class}_{bev_type}'] / wallbox_ccf['sec_elec'])

            return (
                    opVarStor[loc, batt_name, ip, p, t] * charge_efficiency
                    >= opVarConv[loc, wallbox_name, ip, p, t] * wallbox_efficiency * 0.999
            )

        self.esM.pyM.bev_battery_constraint = pyomo.Constraint(
            self.bev_constraint_data.keys(),
            ['infl_uni_0', 'infl_uni_1', 'flex_uni', 'flex_bi'],
            self.esM.pyM.timeSet,
            rule=bev_battery_constraint
        )

    def optimize(self, numberOfTypicalPeriods):
        self.add_co2_sink()
        self.add_slack_sinks()
        optimization_specs = "IntFeasTol=1e-3 NumericFocus=1 BarHomogeneous=1 ScaleFlag=2"
        self.esM.aggregateTemporally(numberOfTypicalPeriods=numberOfTypicalPeriods, segmentation=False)
        self.esM.declareOptimizationProblem(timeSeriesAggregation=True)
        self.declare_bev_constraints()
        self.esM.optimize(
            declaresOptimizationProblem=False,
            timeSeriesAggregation=True,
            optimizationSpecs=optimization_specs,
            solver="gurobi"
        )


    def export_results(self):
        results_df = pd.DataFrame()
        if self.esM.pyM.op_flex_conv.get_values() != {}:
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
                if 'slack_' in comp_name:
                    continue
                comp_df = self.get_comp_results(
                    comp_name,
                    {ip: df.loc[comp_name] for ip, df in model_results.items()}
                )
                results_df = pd.concat([results_df, comp_df], ignore_index=True)
        results_df = results_df.reindex(
            columns=['scenario', 'process', 'parameter', 'sector', 'category', 'specification', 'new',
                     'groups', 'input_groups', 'output_groups', 'year', 'unit', 'value']
        )
        results_df.loc[
            (results_df['unit'] == utils.standard_units['power']) & (results_df['parameter'] == 'flow_volume'), 'unit'
        ] = utils.standard_units['energy']
        if os.getcwd().endswith('data_adapter_fine'):
            results_df.to_csv('examples/output/'+self.scenario+'.csv', index=True, sep=';', index_label='id')
        else:
            results_df.to_csv('output/'+self.scenario+'.csv', index=True, sep=';', index_label='id')

        return results_df

    def get_comp_results(self, comp_name, model_results):
        # set descriptive data for component
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
            comp_results_base['new'] = 1
        elif comp_name[-1] == '0':
            comp_results_base['new'] = 0
        else:
            comp_results_base['new'] = 0 #TODO check components and define value

        params = {
            'capacity_inst': 'capacity',
            'capacity_new': 'commissioning',
            'costs_investment': 'capexCap',
            'costs_fixed': 'opexCap',
            'costs_variable': 'opexOp',
        }

        # set capacity, commissioning, and cost results for each investment period
        for ip in self.esM.investmentPeriodNames:
            comp_results = comp_results_base.copy()
            comp_results['year'] = [ip]
            for param_sedos, param_fine in params.items():
                comp_results['parameter'] = [param_sedos]
                if (
                        param_sedos.startswith('capacity')
                        and comp_name.startswith('tra_')
                        and any(output.startswith('exo_road_') for output in self.processes[comp_name].outputs)
                ):
                    comp_results['unit'] = [utils.standard_units['vehicles']]
                    comp_results['value'] = [model_results[ip].loc[param_fine].values[0][0]]
                elif param_sedos == 'costs_variable' and isinstance(comp, fn.Storage):
                    comp_results['unit'] = [model_results[ip].loc['opexCharge'].index.values[0][1:-1]]
                    comp_results['value'] = [model_results[ip].loc[['opexCharge', 'opexDischarge']].sum().sum()]
                else:
                    comp_results['unit'] = [model_results[ip].loc[param_fine].index.values[0][1:-1]]
                    comp_results['value'] = [model_results[ip].loc[param_fine].values[0][0]]
                comp_df = pd.concat([comp_df, pd.DataFrame(comp_results)])

            # set flow volume results for utilized commodities
            if isinstance(comp, fn.Storage):
                continue
            comp_results['parameter'] = ['flow_volume']
            comp_results['value'] = [model_results[ip].loc['operation'].values[0][0]]
            if isinstance(comp, fn.Source):
                comp_results['output_groups'] = [[comp.commodity]]
                comp_results['unit'] = self.esM.commodityUnitsDict[comp.commodity]
                comp_df = pd.concat([comp_df, pd.DataFrame(comp_results)])
            elif isinstance(comp, fn.Sink):
                comp_results['input_groups'] = [[comp.commodity]]
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
                                comp_results_flow['output_groups'] = [[commod]]
                            elif ccf < 0:
                                comp_results_flow['input_groups'] = [[commod]]
                            comp_results_flow['unit'] = self.esM.commodityUnitsDict[commod]
                            comp_df = pd.concat([comp_df, pd.DataFrame(comp_results_flow)])
                        continue
                    elif value > 0:
                        comp_results_flow['output_groups'] = [[commod]]
                    elif value < 0:
                        comp_results_flow['input_groups'] = [[commod]]
                    comp_results_flow['unit'] = self.esM.commodityUnitsDict[commod]
                    comp_results_flow['value'] = [comp_results_flow['value'][0] * abs(value)]
                    comp_df = pd.concat([comp_df, pd.DataFrame(comp_results_flow)])
                if len(comp_df[comp_df['parameter'] == 'flow_volume']['unit'].unique()) == 1:
                    comp_results_flow['output_groups'] = [['losses']]
                    comp_results_flow['input_groups'] = [None]
                    comp_results_flow['value'] = (
                        comp_df[
                            (comp_df['parameter'] == 'flow_volume')
                            & (comp_df['year'] == ip)
                            & (comp_df['input_groups'].notna())
                        ]['value'].sum()
                        - comp_df[
                            (comp_df['parameter'] == 'flow_volume')
                            & (comp_df['year'] == ip)
                            & (comp_df['output_groups'].notna())
                        ]['value'].sum()
                    )
                    comp_df = pd.concat([comp_df, pd.DataFrame(comp_results_flow)])
        comp_df['value'] = comp_df['value'].fillna(0)

        return comp_df

    def calc_electricity_market_data(self):
        for ip in self.esM.investmentPeriodNames:
            market_source = self.esM.getComponent('electricity_market_source').aggregatedCommodityCostTimeSeries[
                self.esM.investmentPeriodNames.index(ip)
            ]
            market_source = market_source.unstack(level=-1)
            market_source.columns = market_source.columns.droplevel()

            market_sink = self.esM.getComponent('electricity_market_sink').aggregatedCommodityRevenueTimeSeries[
                self.esM.investmentPeriodNames.index(ip)
            ]
            market_sink = market_sink.unstack(level=-1)
            market_sink.columns = market_sink.columns.droplevel()

            market_source_full = []
            market_sink_full = []
            for p in self.esM.periodsOrder[self.esM.investmentPeriodNames.index(ip)]:
                market_source_full.append(market_source.loc[p])
                market_sink_full.append(market_sink.loc[p])
            market_source_ts = pd.concat(market_source_full, axis=0, ignore_index=True)
            market_sink_ts = pd.concat(market_sink_full, axis=0, ignore_index=True)

            self.market_source_df[ip] = market_source_ts
            self.market_sink_df[ip] = market_sink_ts

