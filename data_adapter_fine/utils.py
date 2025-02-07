import pandas as pd
import numpy as np

helper_commodities = [
    'sec_elec',
    'exo_road_mcar_pkm',
    'co2_equivalent',
    'veh',
    "x2x_import_biogas_helper",
    "x2x_import_sng_helper",
    "x2x_import_syndiesel_helper",
    "x2x_import_syngasoline_helper",
    "x2x_x2liquid_source_biodiesel_helper",
    "x2x_x2liquid_source_bioethanol_helper",
    "x2x_import_synkerosene_helper",
    "x2x_x2liquid_source_biodiesel_helper",
    "x2x_x2liquid_source_bioethanol_helper",
    "x2x_x2liquid_source_biokerosene_helper",
]

slack_sink_opex = {
    'slack_sink_sec_naphtha_fos_orig': 100,
    'slack_sink_sec_kerosene_fos_orig': 1,
    'slack_sink_sec_refinery_gas': 100,
    'slack_sink_sec_heat_low': 1,
}

standard_units = {
    'None': None,
    'year': 'a',
    'cost': 'MEUR',
    'energy': 'GWh',
    'power': 'GW',
    'efficiency': '%',
    'transport_pass_demand': 'Gpkm',
    'vehicles': 'kvehicles',
    'emissions': 'Mt',
    'pass_transport_ccf': 'Gpkm/kvehicles',
    'energy_transport_ccf': 'GWh/kvehicles',
    'power_per_vehicle': 'GW/kvehicles',
    'milage': 'Tm/(kvehicles*a)',
    'self_discharge': '%/h',
    'cost_per_capacity': 'MEUR/GW',
    'cost_per_energy': 'MEUR/GWh',
    'cost_per_vehicle': 'MEUR/kvehicles',
    'cost_per_pkm': 'MEUR/Gpkm',
    'cost_var_per_vehicle': 'MEUR/(kvehicles*a)',
    'specific_emission': 'Mt/GWh',
    'specific_emission_co2': 'MtCO2/GWh',
    'ccf_vehicles': 'GWh/100km',
    # 'misc_ccf': 'MWh/MWh',
    'misc_ts': 'kW/kW',
    'occupancy_rate': 'persons/vehicle',
}

param_mapping = {
    'cost_fix_p': 'opexPerCapacity',
    'cost_fix_tra': 'opexPerCapacity',
    'cost_fix_w': 'opexPerCapacity',
    'cost_inv_e': 'investPerCapacity',
    'cost_inv_p': 'investPerCapacity',
    'cost_inv_tra': 'investPerCapacity',
    'cost_inv_w': 'investPerCapacity',
    'cost_var_e': 'opexPerOperation',
    'cost_var_tra': 'opexPerOperation',
    'cost_var_w': 'opexPerOperation',
    'capacity_e_max': 'capacityMax',
    'capacity_p_max': 'capacityMax',
    'capacity_tra_max': 'capacityMax',
    'capacity_p_min': 'capacityMin',
    'capacity_tra_min': 'capacityMin',
    'capacity_e_inst_0': 'capacityFix',
    'capacity_p_inst_0': 'capacityFix',
    'capacity_p_inst': 'capacityFix',
    'capacity_e_inst': 'capacityFix',
    'capacity_tra_inst_0': 'capacityFix',#TODO: check with Hedda if needed
    'lifetime': 'economicLifetime',
    'capacity_e_unit': 'capacityPerPlantUnit',
    'capacity_p_unit': 'capacityPerPlantUnit',
    'capacity_p_abs_new_max': 'commissioningMax'
}

timeseries_mapping = {
    'availability_timeseries_fixed': 'operationRateFix',
    'availability_timeseries_max': 'operationRateMax',
}

storage_param_mapping = {
    'efficiency_sto_in': 'chargeEfficiency',
    'efficiency_sto_out': 'dischargeEfficiency',
    'sto_self_discharge': 'selfDischarge',
}

storage_timeseries_mapping = {
    'sto_max_timeseries': 'stateOfChargeMax',
    'sto_min_timeseries': 'stateOfChargeMin',
}

def get_commodity_unit(commodity):
    if commodity == 'veh':
        return standard_units['vehicles']
    elif commodity.startswith('emi_'):
        return 'Mt'
    elif commodity.startswith('exo_road') and 'pkm' in commodity:
        return standard_units['transport_pass_demand']
    else:
        return standard_units['power']

def drop_unused_columns(df):
    unused_columns = [
            'method',
            'source',
            'region',
            'comment',
            'bandwidth_type',

        ]
    return df.drop(unused_columns, axis=1)

def calc_tra_ccf(ccf_dict, scalars):

    ccf_dict_return = {ip: {} for ip in ccf_dict.keys()}
    if 'occupancy_rate' in scalars.columns:
        load = scalars['occupancy_rate']
    elif 'tonnage' in scalars.columns:
        load = scalars['tonnage']
    else:
        raise ValueError('Neither occupancy_rate nor tonnage found in scalars')

    for year, ccf in ccf_dict.items():
        mileage = scalars['mileage'].loc[year]
        demand_commod = [commod for commod in ccf.keys() if commod.startswith('exo_road')][0]
        for commod, value in ccf.items():
            if isinstance(value, dict):
                ccf_dict_return[year][commod] = {}
                for subcommod, subvalue in value.items():
                    ccf_dict_return[year][commod][subcommod] = (subvalue / ccf[demand_commod]) * mileage * load.loc[year]
            else:
                ccf_dict_return[year][commod] = (value / ccf[demand_commod]) * mileage * load.loc[year]

    return ccf_dict_return

def check_coefficents(esM):
    from fine.IOManagement.dictIO import exportToDict
    from collections.abc import MutableMapping
    def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str = '.') -> MutableMapping:
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + str(k) if parent_key else k
            if isinstance(v, MutableMapping):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    esmDict, compDict = exportToDict(esM)
    flattened_dict = flatten_dict(compDict)

    # def get_coefficent(item):
    #     if isinstance(item, (int, float)):
    #         min, max = item, item
    #     elif isinstance(item, (pd.Series, pd.DataFrame)):
    #         min, max = item.max().max(), item.min().min()
    #     else:
    #         return None

    max_coefficient = 0
    min_coefficient = 1000000
    for key, value in flattened_dict.items():
        if value is None or isinstance(value, str):
            continue
        if isinstance(value, (pd.Series, pd.DataFrame)):
            if value.isnull().values.any():
                raise ValueError(f'Coefficient for {key} contains NaN')
        elif not isinstance(value, str) and np.isnan(value):
            raise ValueError(f'Coefficient for {key} is NaN')
        elif isinstance(value, (int, float)):
            if abs(value) > max_coefficient:
                max_coefficient = abs(value)
            if abs(value) < min_coefficient and value != 0:
                min_coefficient = abs(value)
        elif isinstance(value, (pd.Series, pd.DataFrame)):
            if abs(value.max().max()) > max_coefficient:
                max_coefficient = abs(value.max().max())
            if abs(value.min().min()) < min_coefficient and value.min().min() != 0:
                min_coefficient = abs(value.min().min())
        else:
            print('Unknown type: ', type(value))
            print(f'Please check the coefficient for {key}')

    for key, value in flattened_dict.items():
        if value is None or isinstance(value, str):
            continue
        elif isinstance(value, (int, float)):
            if abs(value) >= max_coefficient * 1e-2:
                print(f'Maximum coefficient: {key} = {value}')
            if abs(value) <= min_coefficient * 1e2 and value != 0:
                print(f'Minimum coefficient: {key} = {value}')
        elif isinstance(value, (pd.Series, pd.DataFrame)):
            if abs(value.max().max()) >= max_coefficient * 1e-2:
                print(f'Maximum coefficient: {key} = {value.max().max()}')
            if abs(value.min().min()) <= min_coefficient * 1e2 and value.min().min() != 0:
                print(f'Minimum coefficient: {key} = {value.min().min()}')

    print('Max coefficient: ', max_coefficient)
    print('Min coefficient: ', min_coefficient)
    print('Coefficient range: ', max_coefficient / min_coefficient)
