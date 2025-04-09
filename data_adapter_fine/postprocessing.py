import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

mapping = {
    'tra_road_mcar_bev_pass_engine_flex_uni': 'BEV_flex_uni',
    'tra_road_mcar_bev_pass_engine_flex_bi': 'BEV_flex_bi',
    'tra_road_mcar_bev_pass_engine_infl_uni': 'BEV_infl',
    'tra_road_mcar_ice_pass_diesel': 'ICE_diesel',
    'tra_road_mcar_ice_pass_gasoline': 'ICE_gasoline',
    'tra_road_mcar_ice_pass_lpg': 'ICE_gas',
    'tra_road_mcar_hyb_pass_diesel': 'Hybrid_diesel',
    'tra_road_mcar_hyb_pass_gasoline': 'Hybrid_gasoline',
    'tra_road_mcar_fcev_pass_hydrogen': 'FCEV',
    'tra_road_mcar_ice_pass_methanol': 'ICE_methanol',
}

# Generate a list of unique colors
colors = plt.get_cmap('tab20').colors
# Set the custom color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

def plot_bev_operation(da, ip):
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=tuple(list(colors[0:3]) + [(0,0,0)]))
    capacities = {}
    capacities[ip] = da.esM.componentModelingDict['ConversionModel'].capacityVariablesOptimum[ip]
    capacities[ip] = capacities[ip].loc[capacities[ip].index.str.contains('engine')]
    capacities[ip].loc['BEV_inflex'] = (
            capacities[ip].loc['tra_road_mcar_bev_pass_engine_infl_uni_1']
            + capacities[ip].loc['tra_road_mcar_bev_pass_engine_infl_uni_0']
    )
    capacities[ip] = capacities[ip].T.drop(
        ['tra_road_mcar_bev_pass_engine_infl_uni_1', 'tra_road_mcar_bev_pass_engine_infl_uni_0'],
        axis=1
    )
    capacities[ip] = capacities[ip].rename(columns={
        'tra_road_mcar_bev_pass_engine_flex_uni_1': 'BEV_flex_uni',
        'tra_road_mcar_bev_pass_engine_flex_bi_1': 'BEV_flex_bi',
    })

    df = da.esM.componentModelingDict['ConversionModel'].getOptimalValues(
        'operationVariablesOptimum',
        ip=ip
    )['values']
    df = df[df.index.get_level_values(0).str.contains('wallbox')]
    df = df.reset_index(level=1, drop=True)

    if da.market_source_df.empty:
        da.calc_electricity_market_data()
    df.loc['Strommarkt'] = da.market_source_df[ip]
    df = df.T
    df['tra_road_mcar_bev_pass_wallbox_flex_bi'] = (
            df['tra_road_mcar_bev_pass_wallbox_flex_bi_g2v_1']
            - df['tra_road_mcar_bev_pass_wallbox_flex_bi_v2g_1']
    )
    df['tra_road_mcar_bev_pass_wallbox_infl'] = (
            df['tra_road_mcar_bev_pass_wallbox_infl_uni_g2v_1']
            + df['tra_road_mcar_bev_pass_wallbox_infl_uni_g2v_0']
    )
    df = df.drop([
        'tra_road_mcar_bev_pass_wallbox_flex_bi_g2v_1',
        'tra_road_mcar_bev_pass_wallbox_flex_bi_v2g_1',
        'tra_road_mcar_bev_pass_wallbox_infl_uni_g2v_1',
        'tra_road_mcar_bev_pass_wallbox_infl_uni_g2v_0'
    ], axis=1)

    df = df.rename(columns={
        'tra_road_mcar_bev_pass_wallbox_flex_uni_g2v_1':'BEV_flex_uni',
        'tra_road_mcar_bev_pass_wallbox_flex_bi': 'BEV_flex_bi',
        'tra_road_mcar_bev_pass_wallbox_infl': 'BEV_inflex'
    })
    for col in capacities[ip].columns:
        df[col] = df[col] / capacities[ip][col].values[0] * 1000

    base_date = datetime.datetime(ip, 1, 1)
    df.index = pd.to_timedelta(df.index, unit='h') + base_date

    df_plot = df.loc[datetime.datetime(ip, 5,2,0,0):datetime.datetime(ip, 5,7,0,0)]

    ax = df_plot.drop(['Strommarkt'], axis=1).plot()
    ax2 = df_plot['Strommarkt'].plot(secondary_y=True, ax=ax, ylim=(0, 0.5), label="Strommarkt (rechte Achse)")
    ax.set_ylabel('Leistung pro Fahrzeug in kW')
    ax.right_ax.set_ylabel('Strompreis EUR/kWh')
    ax2._right_label = "Strommarkt (rechte Achse)"
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    labels2 = ["Strommarkt (rechte Achse)"]
    combined_handles = handles + handles2
    combined_labels = labels + labels2

    # Move the combined legend below the plot
    ax.legend(combined_handles, combined_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.savefig('output/Wallbox_' + da.scenario + '.svg')
    plt.show()

def plot_bev_costs_revenue(da):
    # Generate a list of unique colors
    colors = plt.get_cmap('tab20').colors
    # Set the custom color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

    costs = pd.DataFrame(index=da.esM.investmentPeriodNames,
                         columns=['BEV_inflex', 'BEV_flex_uni', 'BEV_flex_bi_g2v', 'BEV_flex_bi_v2g'])
    costs_per_veh = costs.copy()
    conv_op = {}
    for ip in da.esM.investmentPeriodNames:
        capacities = {}
        capacities[ip] = da.esM.componentModelingDict['ConversionModel'].capacityVariablesOptimum[ip]
        capacities[ip] = capacities[ip].loc[capacities[ip].index.str.contains('engine')]
        capacities[ip].loc['BEV_inflex'] = (
                capacities[ip].loc['tra_road_mcar_bev_pass_engine_infl_uni_1']
                + capacities[ip].loc['tra_road_mcar_bev_pass_engine_infl_uni_0']
        )
        capacities[ip] = capacities[ip].T.drop(
            ['tra_road_mcar_bev_pass_engine_infl_uni_1', 'tra_road_mcar_bev_pass_engine_infl_uni_0'],
            axis=1
        )
        capacities[ip] = capacities[ip].rename(columns={
            'tra_road_mcar_bev_pass_engine_flex_uni_1': 'BEV_flex_uni',
            'tra_road_mcar_bev_pass_engine_flex_bi_1': 'BEV_flex_bi',
        })
        conv_op[ip] = da.esM.componentModelingDict['ConversionModel'].getOptimalValues(
            'operationVariablesOptimum',
            ip=ip
        )['values']
        inflex_op = (
                conv_op[ip].loc['tra_road_mcar_bev_pass_wallbox_infl_uni_g2v_1', 'Germany']
                + conv_op[ip].loc['tra_road_mcar_bev_pass_wallbox_infl_uni_g2v_0', 'Germany']
        )
        costs.loc[ip]['BEV_inflex'] = (inflex_op * da.market_source_df[ip]).sum()
        flex_uni_op = conv_op[ip].loc['tra_road_mcar_bev_pass_wallbox_flex_uni_g2v_1', 'Germany']
        costs.loc[ip]['BEV_flex_uni'] = (flex_uni_op * da.market_source_df[ip]).sum()
        flex_bi_g2v_op = conv_op[ip].loc['tra_road_mcar_bev_pass_wallbox_flex_bi_g2v_1', 'Germany']
        costs.loc[ip]['BEV_flex_bi_g2v'] = (flex_bi_g2v_op * da.market_source_df[ip]).sum()
        flex_bi_v2g_op = conv_op[ip].loc['tra_road_mcar_bev_pass_wallbox_flex_bi_v2g_1', 'Germany']
        costs.loc[ip]['BEV_flex_bi_v2g'] = (flex_bi_v2g_op * da.market_sink_df[ip]).sum() * -1

        for col in costs_per_veh.columns:
            if len(col.split('_')) > 3:
                col_cap = '_'.join(col.split('_')[0:3])
            else:
                col_cap = col
            if capacities[ip][col_cap].values[0] != 0:
                costs_per_veh.loc[ip, col] = costs.loc[ip, col] / capacities[ip][col_cap].values[0]
            else:
                costs_per_veh.loc[ip, col] = 0

    costs_per_veh = costs_per_veh * 1000

    # Plot the costs DataFrame as a stacked bar chart
    costs_per_veh.drop(2021).plot(kind='bar', stacked=True, figsize=(10, 8))

    # Set the title and labels
    plt.title('Electricity Cost and Revenue per Year')
    plt.xlabel('Year')
    plt.ylabel('Cost/Revenue in EUR/vehicle')

    # Place the legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.savefig('output/BEV_elec_costs_' + da.scenario + '.svg')
    # Show the plot
    plt.show()

def plot_tac(da):
    # Generate a list of unique colors
    colors = plt.get_cmap('tab20').colors
    # Set the custom color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

    npv_results = pd.DataFrame()
    for ip in da.esM.investmentPeriodNames:
        if ip == 2021:
            continue
        npv_results_ip = pd.Series()
        for mdl in da.esM.componentModelingDict.keys():
            npv_results_ip = pd.concat([
                npv_results_ip,
                da.esM.getOptimizationSummary(mdl, ip=ip).loc[:, 'TAC', '[MEUR/a]']['Germany']
            ])
        npv_results[ip] = npv_results_ip
    npv_results = npv_results.loc[~(npv_results <= 10).all(axis=1)]
    npv_results['sum'] = npv_results.sum(axis=1).astype(float)
    npv_results.nlargest(30, 'sum').drop(columns=['sum']).sort_index().T.plot(kind='bar', stacked=True, figsize=(10, 8))
    # make title for plot
    plt.title('TAC in MEUR')

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.savefig('output/costs_' + da.scenario + '.svg')
    plt.show()

def plot_capacities(da, veh_class, results):
    # Generate a list of unique colors
    colors = plt.get_cmap('tab20').colors
    # Set the custom color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

    # Create a dictionary to store the capacity data for each component
    capacity_plots = {}
    veh_list = [
        comp
        for comp in da.esM.componentNames.keys()
        if 'tra_road_' + veh_class in comp
           and not 'wallbox' in comp
           and not 'battery' in comp
    ]

    for component in veh_list:
        comp_name = component[:-2]
        capacity_dict = {
            ip: da.esM.componentModelingDict['ConversionModel'].capacityVariablesOptimum[ip].loc[component].values[0]
            for ip in da.esM.investmentPeriodNames
        }
        operation_dict = {
            ip: da.esM.getOptimizationSummary('ConversionModel', ip=ip).loc[component, 'operation'].values[0][0]
            for ip in da.esM.investmentPeriodNames
        }
        if comp_name not in capacity_plots.keys():
            capacity_plots[comp_name] = pd.DataFrame(
                0,
                index=da.esM.investmentPeriodNames,
                columns=['capacityMin', 'capacityMax', 'capacity', 'operation']
            )
        if component.endswith('1'):
            capacity_plots[comp_name]['capacityMin'] += pd.Series(da.esM.getComponent(component).capacityMin)
            capacity_plots[comp_name]['capacityMax'] += pd.Series(da.esM.getComponent(component).capacityMax)
        else:
            capacity_plots[comp_name]['capacityMin'] += pd.Series(capacity_dict)
            capacity_plots[comp_name]['capacityMax'] += pd.Series(capacity_dict)

        capacity_plots[comp_name]['capacity'] += pd.Series(capacity_dict)
        capacity_plots[comp_name]['operation'] += pd.Series(operation_dict)

    # get operation of helper_sink_exo_road_mcar_pkm for each investment period
    demand_slack_operation = results[
        (results.process.isin(['helper_sink_exo_road_' + veh_class + '_pkm', 'slack_source_' + veh_class + '_demand']))
        & (results.parameter == 'flow_volume')
        ][['year', 'process', 'value']].pivot(index='year', columns='process', values='value')

    # Create subplots with 4 rows and 3 columns
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for comp_name in capacity_plots.keys():
        if 'engine_flex' in comp_name:
            infl_name = '_'.join(comp_name.split('_')[:-2]) + '_infl_uni'
            car_class = comp_name.split('_')[2]
            factor = da.bev_constraint_data[car_class]['_'.join(comp_name.split('_')[-2:])]
            capacity_plots[comp_name]['capacityMin'] = capacity_plots[infl_name]['capacityMin'] * factor
            capacity_plots[comp_name]['capacityMax'] = capacity_plots[infl_name]['capacityMax'] * factor

    capacity_plots_selected = {
        mapping[key]: value for key, value in capacity_plots.items() if key in mapping.keys()
    }
    # Iterate over the capacity_plots dictionary and plot each one
    for ax, (comp_name, data) in zip(axes, capacity_plots_selected.items()):

        # Drop rows with NaN values
        data = data.dropna(axis=1)
        data['Kapazität'] = data['capacity']
        data[['Kapazität']].plot(ax=ax, title=comp_name, color='red', label='Kapazität', legend=False)
        ax.fill_between(
            data.index,
            data['capacityMin'],
            data['capacityMax'],
            color='lightblue',
            alpha=0.5,
            label='Capacity Range'
        )
        ax.set_ylim(0, max([df.max().max() for df in capacity_plots.values()]))
        handles, labels = ax.get_legend_handles_labels()

    # Create custom legend handles
    # capacity_patch = mpatches.Patch(color='red', label='Kapazität')
    capacity_range_patch = mpatches.Patch(color='lightblue', alpha=0.5, label='Kapazitätsbeschränkung')

    # Add a single legend below all subplots
    # handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles=[handles[0]] + [capacity_range_patch], loc='lower center', ncol=3)
    fig.supxlabel('Jahr', y=0.065)
    fig.supylabel('Anzahl Fahrzeuge in 1000')

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Show the plot
    plt.show()
    plt.savefig('output/capacity_' + da.scenario + '.svg')

def plot_pri_energy(da, veh_class):
    # Generate a list of unique colors
    colors = plt.get_cmap('tab20').colors
    # Set the custom color cycle
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

    # plot for operation of source components
    source_list = [
        comp_name
        for comp_name in da.esM.componentModelingDict['SourceSinkModel'].componentsDict.keys()
        if da.esM.getComponent(comp_name).sign == 1
        if comp_name != 'slack_source_' + veh_class + '_demand'
    ]
    source_operation = pd.DataFrame({
        comp_name: pd.Series({
            ip: da.esM.getOptimizationSummary('SourceSinkModel', ip=ip).loc[comp_name, 'operation'].values[0][0]
            for ip in da.esM.investmentPeriodNames
        })
        for comp_name in source_list
    })

    # Remove columns where the sum is 0
    source_operation = source_operation.loc[:, (source_operation.sum(axis=0) != 0)]

    # Plot the source_operation DataFrame as a stacked bar plot
    ax = source_operation.plot(kind='bar', stacked=True, figsize=(8, 6))

    # Set the title and labels
    plt.title('Primary Energy Consumption')
    plt.xlabel('Year')
    plt.ylabel('Operation')

    # Place the legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout to make room for the legend
    plt.tight_layout()

    # Show the plot
    plt.show()
    plt.savefig('output/pri_energy_import_' + da.scenario + '.svg')

def plot_operation_stacked(da, results):

    # Define the base color in RGB
    base_color = np.array([2, 61, 107]) / 255  # Normalize to [0, 1] range
    # Create different shades of the base color
    shades = np.linspace(1, 2, 3)[:, None] * base_color

    new_shades = np.array([[89, 89, 89], [128, 128, 128], [165, 165, 165], [255, 185, 109], [244, 102, 138]]) / 255
    extended_shades = np.vstack([shades, new_shades])
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=extended_shades)

    mcar_prod = results[
        (results.output_groups.apply(lambda x: 'exo_road_mcar_pkm' in x[0] if isinstance(x, list) else False))
    ]
    mcar_prod = mcar_prod[['year', 'process', 'value']].pivot(index='year', columns='process', values='value')
    mcar_prod = mcar_prod.drop(['helper_sink_exo_road_mcar_pkm'], axis=1)

    plot_df = pd.DataFrame(columns=mapping.values(), index=mcar_prod.index, data=0)
    for key in mapping.keys():
        if key + '_1' in mcar_prod.columns:
            plot_df[mapping[key]] += mcar_prod[key + '_1']
        if key + '_0' in mcar_prod.columns:
            plot_df[mapping[key]] += mcar_prod[key + '_0']
        if plot_df[mapping[key]].sum() <= 10:
            plot_df = plot_df.drop(columns=[mapping[key]])

    plot_df = plot_df[
        ['BEV_flex_uni', 'BEV_flex_bi', 'BEV_infl', 'ICE_diesel', 'ICE_gasoline', 'Hybrid_gasoline', 'ICE_gas',
         'ICE_methanol']]
    plot_df.plot(kind='bar', stacked=True, figsize=(8, 6))
    plt.ylabel('Fahrleistung in Gpkm')
    plt.xlabel('Jahr')
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.tight_layout()

    plt.savefig('output/operation_' + da.scenario + '.svg')
    plt.show()

def plot_co2(da, results):
    new_shades = np.array([[0, 0, 0], [89, 89, 89], [128, 128, 128], [165, 165, 165], [255, 185, 109], [244, 102, 138],
                           [163, 228, 169]]) / 255

    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=new_shades)

    # co2 plot
    exclusion_list = ['co2_conversion_helper', 'helper_co2_delivery', 'helper_co2_delivery',
                      'helper_co2_storage_permanent', 'co2_sink', 'co2_source', 'co2_stored_sink']

    co2_emissions = results[
        (results.output_groups.apply(lambda x: 'co2' in x[0] if isinstance(x, list) else False))
        & (results.output_groups.apply(lambda x: 'neg' not in x[0] if isinstance(x, list) else False))
        & (results.output_groups.apply(lambda x: 'emi_co2_reusable' not in x[0] if isinstance(x, list) else False))
        & (results.process.apply(lambda x: x not in exclusion_list))
        & (results.value != 0)
        ][['year', 'process', 'value']]
    co2_emissions = co2_emissions.pivot(index='year', columns='process', values='value')
    co2_emissions.index.name = None
    co2_emissions.columns.name = None

    mapping['x2x_x2liquid_oref'] = 'Raffinerie'

    co2_plot = pd.DataFrame()
    for col in co2_emissions.columns:
        if 'tra_road_mcar_hyb_pass_diesel' in col:
            continue
        if col[:-2] in mapping.keys():
            if mapping[col[:-2]] not in co2_plot.columns:
                co2_plot[mapping[col[:-2]]] = co2_emissions[col]
            else:
                co2_plot[mapping[col[:-2]]] = co2_plot[mapping[col[:-2]]].fillna(0) + co2_emissions[col].fillna(0)

    dac_op = results[
        (results.process.isin(['x2x_other_dac_ht_1', 'x2x_other_dac_lt_1']))
        & (results.output_groups.apply(lambda x: x[0] == 'emi_co2_neg_air_dacc' if isinstance(x, list) else False))
        ][['year', 'process', 'value']]
    dac_op = dac_op.pivot(index='year', columns='process', values='value')
    dac_op.index.name = None
    dac_op.columns.name = None
    co2_plot['DAC'] = dac_op.sum(axis=1) * -1

    co2_plot = co2_plot[
        ['Raffinerie', 'Hybrid_gasoline', 'ICE_diesel', 'ICE_gasoline', 'ICE_gas', 'ICE_methanol', 'DAC']]

    ax = co2_plot.plot(kind='bar', stacked=True, figsize=(8, 6))

    # Calculate the sum of each stacked bar
    bar_sums = co2_plot.sum(axis=1)

    # Add a point at the top of each bar
    scatter = ax.scatter(range(len(bar_sums)), bar_sums, color='black', marker='_', s=600)

    # Plot the emissions limit as a line
    emissions_limit = pd.Series(
        {ip: abs(da.esM.balanceLimit[ip].loc['emissions', 'Germany']) for ip in da.esM.investmentPeriodNames}
    )
    emissions_limit = emissions_limit.reindex(co2_plot.index)  # Ensure the index matches
    ax.plot(range(len(emissions_limit)), emissions_limit.values, color=(0.545, 0, 0), linestyle='--',
            label='Emissions Limit')

    # Add the scatter to the legend
    handles, labels = ax.get_legend_handles_labels()
    handles.append(scatter)
    labels.append('Jahressumme')
    handles.append(handles[0])
    labels.append(labels[0])
    handles = handles[1:]
    labels = labels[1:]

    # Update the legend
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
    # plt.title('CO2 Emissionen')
    plt.xlabel('Jahr')
    plt.ylabel('CO2 in Mt')

    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.savefig('output/co2_' + da.scenario + '.svg')
    plt.show()