import logging
import math
import numpy as np
import pandas as pd

import fine as fn

from data_adapter_fine import utils

class ComponentConstructor:
    def __init__(self, process, esM):
        self.fine_args = {
            'name': process[0],
        }
        self.comps = []

        self.scalars = process[1].scalars.set_index('year')
        self.scalars = utils.drop_unused_columns(self.scalars)
        self.scalars.index = self.scalars.index.astype(int)
        if process[0].endswith('_0') and 'capacity_tra_max' in self.scalars.columns and 'capacity_tra_min' in self.scalars.columns:
            self.scalars = self.scalars.drop(columns=['capacity_tra_max', 'capacity_tra_min'])

        # unpack lists in scalars
        if self.scalars.applymap(lambda x: isinstance(x, list)).any().any():
            self.scalars = self.scalars.applymap(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
            self.scalars = self.scalars.applymap(lambda x: np.nan if isinstance(x, list) and len(x) == 0 else x)
        self.col_set = set(self.scalars.columns)
        # cb_coefficient is not used in fine
        self.col_set = self.col_set.difference({'cb_coefficient'})
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
            if 'capacity_e_max' in param_df.columns:
                if not isinstance(self, StorageConstructor):
                    raise ValueError(f"Capacity_e_max is only allowed for storage processes. "
                                     f"Please check {process[0]}.")
                param_df['capacity_e_max'] = param_df['capacity_e_max']/8760
            param_df.rename(columns=utils.param_mapping, inplace=True)
            param_df = self.set_tech_availability(param_df, esM)
            if param_df.columns.duplicated().any():
                raise ValueError(f"Duplicate columns in {process[0]} scalars.")
            if 'yearlyFullLoadHoursMax' in param_df.columns:
                param_df['yearlyFullLoadHoursMax'] = param_df['yearlyFullLoadHoursMax'] / 100 * 8760
            self.fine_args = self.fine_args | param_df.to_dict('dict')

            if 'lifetime' in self.scalars.columns:
                self.fine_args['economicLifetime'] = math.floor(self.scalars['lifetime'].mean())
            else:
                logging.warning(f"Missing lifetime for {process[0]}. Default value 7 is used.")
                self.fine_args['economicLifetime'] = esM.investmentPeriodInterval
                if self.fine_args['name'] in ['x2x_delivery_methane_pipeline_0']:
                    self.fine_args['economicLifetime'] = 56
            if 'wacc' in self.scalars.columns:
                self.fine_args['interestRate'] = self.scalars['wacc'].astype(float).mean() / 100
            else:
                self.fine_args['interestRate'] = 0.02

            if utils.check_floor_lifetime(process[0]):
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
                stock_commissioning = stock_commissioning.round(8)
                self.fine_args['stockCommissioning'] = stock_commissioning.to_dict()
                if len(param_df['capacityFix'][param_df['capacityFix']>0]) > rounded_lifetime / interval:
                    if self.fine_args['name'] in ['x2x_delivery_methane_pipeline_0', 'x2x_x2gas_sr_syngas_0', 'x2x_storage_methane_0']:
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
            if 'engine_flex' in process[0]:
                del self.fine_args['capacityMin']
                del self.fine_args['capacityMax']


    def set_tech_availability(self, param_df, esM):
        if param_df.loc[:, param_df.columns != 'commissioningMax'].isna().any().any():
            self.fine_args['commissioningFix'] = {
                ip: 0 if ip in param_df[param_df.isna().any(axis=1)].index.tolist()
                else None
                for ip in esM.investmentPeriodNames
            }
            param_df = param_df.loc[:, param_df.columns != 'commissioningMax'].fillna(0)
        return param_df.replace(np.nan, None)

    def interpolate_df(self, df, esM):
        for year in pd.Index(esM.investmentPeriodNames).difference(self.scalars.index):
            df.loc[year] = np.nan
        df = df.sort_index().astype('float64')
        df = df.interpolate(method='index', limit_area='inside')
        return df.loc[esM.investmentPeriodNames]


class ConversionConstructor(ComponentConstructor):
    def __init__(self, process, esM, op_timeseries=None, main_commodity=None):
        ComponentConstructor.__init__(self, process, esM)
        if op_timeseries is not None:
            op_timeseries = self.interpolate_df(op_timeseries.T, esM).T
            if (op_timeseries.sum() == 1).any():
                raise ValueError("Operation timeseries sum must be 1.")
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
            if (
                    not commod.startswith('emi')
                    or commod in ['emi_co2_reusable', 'emi_co2_stored', 'emi_co2_neg_air_dacc']
            )
        ]
        commodities_emissions = [
            commod
            for commod in commodities_raw
            if not isinstance(commod, list)
            if commod not in commodities_no_group
        ]

        ccf_columns = [col for col in self.scalars if col.startswith('conversion_factor')]
        self.col_set = self.col_set.difference(ccf_columns)
        ccf_df = self.interpolate_df(self.scalars[ccf_columns], esM)
        ccf_df.rename(columns=lambda x: x[18:], inplace=True)
        # scale conversion factors to main commodity
        if main_commodity is None:
            main_commodity = output_commodities[0]
        ccf_unit_scaling_factor = ccf_df[main_commodity]
        ccf_df = ccf_df.div(ccf_unit_scaling_factor, axis='index')
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
            scale_cols = [
                col for col in ef_df.columns
                if process[1].units[col] != utils.standard_units['specific_emission']
            ]
            ef_df[scale_cols] = ef_df[scale_cols].div(ccf_unit_scaling_factor, axis='index')
            ef_df.rename(columns=lambda x: x[3:], inplace=True)

            if len(commodity_groups) > 0:
                self.fine_args['emissionFactors'] = {}
            for col in ef_df.columns:
                emi_commodity = 'emi' + col.split('_emi')[1]
                # scale unit for N2O emissions in the x2x sector by 1e-6 for smaller coefficients
                if 'emi_n2o' in emi_commodity:
                    ef_df[col] = ef_df[col] * 1e6
                # scale unit for CH4 emissions in the x2x sector by 1e-3 for smaller coefficients
                elif 'emi_ch4' in emi_commodity:
                    ef_df[col] = ef_df[col] * 1e3
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

        self.fine_args['physicalUnit'] = utils.standard_units['power']
        if (
            self.fine_args['name'].startswith('tra_')
            and any(output.startswith('exo_road') for output in process[1].outputs)
        ):
            self.fine_args['commodityConversionFactors'] = utils.calc_tra_ccf(
                self.fine_args['commodityConversionFactors'],
                self.interpolate_df(self.scalars, esM)
            )
            self.fine_args['physicalUnit'] = utils.standard_units['vehicles']
            for ip in esM.investmentPeriodNames:
                self.fine_args['opexPerOperation'][ip] = (
                        self.fine_args['opexPerOperation'][ip]
                        * self.fine_args['commodityConversionFactors'][ip][main_commodity]
                )


            self.col_set = self.col_set.difference({'market_share_range', 'mileage', 'occupancy_rate', 'tonnage'})
            if process[0].endswith('_0'):
                self.col_set = self.col_set.difference({'share_tra_charge_mode'})
        if 'kWh/100km' in process[1].units.values():
            logging.warning(f"Please check ccf unit for {self.fine_args['name']}.")

        if process[0] == 'helper_co2_delivery':
            for ip in self.fine_args['commodityConversionFactors'].keys():
                self.fine_args['commodityConversionFactors'][ip]['co2_equivalent'] = -1

        for commod in commodities_emissions:
            if commod in self.fine_args['commodityConversionFactors'][esM.investmentPeriodNames[0]].keys():
                continue
            if len(ef_columns) > 0 and commod in self.fine_args['emissionFactors'].keys():
                continue
            raise ValueError(f"Commodity {commod} not found in {self.fine_args['name']}.")

        if (abs(ccf_df[main_commodity]) != 1).any():
            logging.warning(f"Conversion factor for {main_commodity} is not 1. Please check {self.fine_args['name']}.")

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
                    raise ValueError(f"Flow share column {col} must contain 'min' or 'max'.")
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
                ccf = {process[0] + "_helper": -1, process[1].outputs[0]: 1}
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
                        physicalUnit='GW',
                        commodityConversionFactors={ip: ccf for ip in esM.investmentPeriodNames},
                        hasCapacityVariable=False,
                    )
                )
            else:
                raise ValueError(f"Process {process[0]} has no inputs and no outputs.")
        elif len(process[1].outputs) == 0:
            if len(process[1].inputs) > 1:
                raise ValueError(f"Process {process[0]} has to many inputs. Only one input per sink is allowed.")
            if 'demand_timeseries' in self.timeseries.keys():
                ts = self.timeseries['demand_timeseries'] / self.timeseries['demand_timeseries'].sum()
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
                if (self.scalars[col].isin([1e3, 1, 1e-3])).all():
                    ccf_cols.append(col)
                else:
                    raise ValueError(f"Conversion factors for Source/ Sink must be equal to 1. "
                                     f"Error in {self.fine_args['name']} ")
        if len(ccf_cols) > 1:
            raise ValueError(f"Multiple conversion factors found in {self.fine_args['name']}.")
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

        if 'helper' in process[0]:
            self.fine_args['selfDischarge'] = 0

        self.col_set = self.col_set.difference({'share_tra_charge_mode'})

        self.comps.append(
            fn.Storage(
                esM=esM,
                commodity=process[1].inputs[0],
                hasCapacityVariable=True,
                isPeriodicalStorage=True,
                **self.fine_args
            )
        )

