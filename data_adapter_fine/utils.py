helper_commodities = [
    'co2_equivalent',
    "x2x_import_biogas_helper",
    "x2x_import_sng_helper",
    "x2x_import_syndiesel_helper",
    "x2x_import_syngasoline_helper",
    "x2x_x2liquid_source_biodiesel_helper",
    "x2x_x2liquid_source_bioethanol_helper",
    # "x2x_import_synkerosene_helper",
    # "x2x_x2liquid_source_biodiesel_helper",
    # "x2x_x2liquid_source_bioethanol_helper",
    # "x2x_x2liquid_source_biokerosene_helper",
]

slack_sink_opex = {
    'slack_sink_sec_naphtha_fos_orig': 1,
    'slack_sink_sec_kerosene_fos_orig': 1,
    'slack_sink_sec_refinery_gas': 1,
    'slack_sink_sec_heat_low': 1,
}

standard_units = {
    'cost': 'BEUR',
    'energy': 'MWh',
    'power': 'MW',
    'transport_pass_demand': 'Mpkm',
    'vehicles': 'kvehicles',
    'pass_transport_ccf': 'Mpkm/kvehicles',
    'energy_transport_ccf': 'GWh/kvehicles',
    'milage': 'Mkm/kvehicles',
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
    'capacity_p_min': 'capacityMin',
    'capacity_e_inst': 'capacityFix',
    'capacity_p_inst': 'capacityFix',
    'capacity_tra_inst_0': 'capacityFix',#TODO: check with Hedda if needed
    'lifetime': 'economicLifetime',
    'capacity_e_unit': 'capacityPerPlantUnit',
    'capacity_p_unit': 'capacityPerPlantUnit',
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
    if commodity.startswith('emi_'):
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

def calc_tra_ccf(ccf_dict, scalars, scaling_factor=10e3):
    """

    :param ccf_dict:
    :param scalars:
    :param scaling_factor: represents the unit factor of demand commodity divided by the unit factor of
        the vehicles (e.g. 10e6 pkm/ 10e3 veh = 10e3 pkm/ veh)
    :return:
    """
    ccf_dict_return = dict.fromkeys(ccf_dict.keys(), {})
    if 'occupancy_rate' in scalars.columns:
        load = scalars['occupancy_rate'].mean()
    elif 'tonnage' in scalars.columns:
        load = scalars['tonnage'].mean()
    else:
        raise ValueError('Neither occupancy_rate nor tonnage found in scalars')

    for year, ccf in ccf_dict.items():
        mileage = scalars['mileage'].loc[year]
        demand_commod = [commod for commod in ccf.keys() if commod.startswith('exo_road')][0]
        for commod, value in ccf.items():
            if isinstance(value, dict):
                ccf_dict_return[year][commod] = {}
                for subcommod, subvalue in value.items():
                    ccf_dict_return[year][commod][subcommod] = (subvalue / ccf[demand_commod]) * mileage * load / scaling_factor
            else:
                ccf_dict_return[year][commod] = (value / ccf[demand_commod]) * mileage * load / scaling_factor

    return ccf_dict_return

