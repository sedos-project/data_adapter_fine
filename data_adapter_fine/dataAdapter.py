import os
import dataclasses
import logging
import numpy as np
import pandas as pd
import pyomo.environ as pyomo


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
            downloadData: bool = True,
    ):
        self.op_flex_opt = None
        if downloadData:
            databus.download_collection(url)
        self.collection_name = url.split('/')[-1]
        self.scenario = 'f_tra_tokio'
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
                name='slack_source_mcar_demand',
                commodity='exo_road_mcar_pkm',
                hasCapacityVariable=False,
                opexPerOperation=1000000,
            )
        )

        self.esM.add(
            fn.Source(
                esM=self.esM,
                name='source_electricity',
                commodity='sec_elec',
                hasCapacityVariable=False,
                opexPerOperation=1,
            )
        )

        self.processes = {
            process: self.adapter.get_process(process)
            for process in list(self.structure.processes.keys())
            if process not in ['tra_road_mcar_ice_pass_methanol_1', 'x2x_import_uran',
                               'x2x_import_deuterium', 'x2x_x2liquid_ft_1', 'x2x_g2p_h2_fuel_cell_1_ag',
                               'x2x_x2gas_sr_syngas_psa_0', 'x2x_x2gas_sr_syngas_psa_1', 'x2x_delivery_naphtha'] #TODO remove
        }

        self.add_processes_to_esM()
        print('All processes added to esM')

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

            if process[0].startswith('tra_road_') and process[0].endswith('1') and 'bev_pass_engine' in process[0]:
                self.get_data_for_bev_constraints(constructor)
                constructor.col_set = constructor.col_set.difference(['share_tra_charge_mode'])

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
                opex = 0.0001
            self.esM.add(
                fn.Sink(
                    esM=self.esM,
                    name='slack_sink_'+commodity,
                    commodity=commodity,
                    hasCapacityVariable=False,
                    opexPerOperation=opex,
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
        # self.esM.aggregateTemporally(numberOfTypicalPeriods=1, numberOfTimeStepsPerPeriod=1)
        self.esM.aggregateTemporally(numberOfTypicalPeriods=1, segmentation=False)
        self.esM.declareOptimizationProblem(timeSeriesAggregation=True)
        # self.esM.declareOptimizationProblem(timeSeriesAggregation=False)
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
                comp_df = self.get_comp_results(
                    comp_name,
                    {ip: df.loc[comp_name] for ip, df in model_results.items()}
                )
                results_df = pd.concat([results_df, comp_df], ignore_index=True)
        results_df = results_df.reindex(
            columns=['scenario', 'year', 'process', 'parameter', 'sector', 'category', 'specification', 'groups',
                     'new', 'input_groups', 'output_groups', 'unit', 'value']
        )
        results_df.loc[
            (results_df['unit'] == utils.standard_units['power']) & (results_df['parameter'] == 'flow_volume'), 'unit'
        ] = utils.standard_units['energy']
        if os.getcwd().endswith('data_adapter_fine'):
            results_df.to_csv('examples/output/test_results.csv', index=False, sep=';')
        else:
            results_df.to_csv('output/test_results.csv', index=False, sep=';')

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
            # 'costs_variable': 'opexOp',
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
                    comp_results_flow['input_groups'] = [[None]]
                    comp_results_flow['value'] = comp_df[
                            (comp_df['parameter'] == 'flow_volume')
                            & (comp_df['year'] == ip)
                        ]['value'].sum()
                    comp_df = pd.concat([comp_df, pd.DataFrame(comp_results_flow)])
        comp_df['value'] = comp_df['value'].fillna(0)

        return comp_df

