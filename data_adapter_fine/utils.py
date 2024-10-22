helper_commodities = [
    "x2x_import_biogas_helper",
    "x2x_import_sng_helper",
    "x2x_import_syndiesel_helper",
    "x2x_import_syngasoline_helper",
    "x2x_import_synkerosene_helper",
    "x2x_x2liquid_source_biodiesel_helper",
    "x2x_x2liquid_source_bioethanol_helper",
    "x2x_x2liquid_source_biokerosene_helper",
]

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
    'capacity_e_inst': 'capacityFix',
    'capacity_p_inst': 'capacityFix',
}

storage_param_mapping = {
    'efficiency_sto_in': 'chargeEfficiency',
    'efficiency_sto_out': 'dischargeEfficiency',
    'sto_self_discharge': 'selfDischarge',
}
def drop_unused_columns(df):
    unused_columns = [
            'method',
            'source',
            'region',
            'comment',
            'bandwidth_type',
        ]
    return df.drop(unused_columns, axis=1)
