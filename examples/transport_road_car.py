#%%
import os
from fine.IOManagement.standardIO import writeOptimizationOutputToExcel
from fine.IOManagement.xarrayIO import writeEnergySystemModelToNetCDF, readNetCDFtoEnergySystemModel
import utils

if os.getcwd().endswith('data_adapter_fine'):
    os.environ["COLLECTIONS_DIR"] = "./collections/"
    os.environ["STRUCTURES_DIR"] = "./structures"
else:
    os.environ["COLLECTIONS_DIR"] = "../collections/"
    os.environ["STRUCTURES_DIR"] = "../structures"

from data_adapter_fine import DataAdapter

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

da = DataAdapter(
    url="https://databus.openenergyplatform.org/johannes_beh/collections/sedos_fine",
    downloadData=False,
    structure_name="structure_fine",
    process_sheet="sys_mcar",
    helper_sheet="helper_sys",
)

# utils.check_coefficents(da.esM)

da.optimize()
results = da.export_results()

#%%
# Create a dictionary to store the capacity data for each component
capacity_plots = {}
veh_list = [
    comp
    for comp in da.esM.componentNames.keys()
    # if not comp.endswith('0')
    if 'tra_road_mcar' in comp
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


#%%
# subplot for capacity, capacityMin, capacityMax, operation
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

# Create subplots with 4 rows and 3 columns
fig, axes = plt.subplots(3, 4, figsize=(10, 8), sharex=True)

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Iterate over the capacity_plots dictionary and plot each one
for ax, (comp_name, data) in zip(axes, capacity_plots.items()):
    if 'engine_flex' in comp_name:
        continue


    # Drop rows with NaN values
    data = data.dropna(subset=['capacityMin', 'capacityMax'])
    title = '_'.join([comp_name.split('_')[3]] + comp_name.split('_')[5:])
    data[['capacity']].plot(ax=ax, title=title, color='green', label='Capacity', legend=False)
    data[['operation']].plot(ax=ax, color='red', label='Operation', legend=False)
    ax.fill_between(
        data.index,
        data['capacityMin'],
        data['capacityMax'],
        color='lightblue',
        alpha=0.5,
        label='Capacity Range'
    )
    handles, labels = ax.get_legend_handles_labels()

# Create custom legend handles
capacity_patch = mpatches.Patch(color='green', label='Capacity')
operation_patch = mpatches.Patch(color='red', label='Operation')
capacity_range_patch = mpatches.Patch(color='lightblue', alpha=0.5, label='Capacity Range')

# Add a single legend below all subplots
# handles, labels = ax.get_legend_handles_labels()
fig.legend(handles=handles[0:2] + [capacity_range_patch], loc='lower center', ncol=3)
fig.delaxes(axes[11])
fig.delaxes(axes[10])
# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0.05, 1, 1])

# Show the plot
# plt.show()
plt.savefig('capacity.png')
#%%
# stacked capacity plot
# Initialize an empty DataFrame
capacity_df = pd.DataFrame()
# Iterate over the capacity_plots dictionary
for component, data in capacity_plots.items():
    # Add a column to the DataFrame with the series stored under 'capacity'
    comp_name = '_'.join([component.split('_')[3]] + component.split('_')[5:])
    capacity_df[comp_name] = data['capacity']

veh_demand_pkm = da.processes['helper_sink_exo_road_mcar_pkm'].scalars[['year', 'demand_annual']]
veh_demand_pkm['year'] = veh_demand_pkm['year'].astype(int)
veh_demand_pkm = veh_demand_pkm.set_index('year')

mileage_pkm = da.processes['tra_road_mcar_ice_pass_diesel_1'].scalars[['year', 'mileage', 'occupancy_rate']]
mileage_pkm['year'] = mileage_pkm['year'].astype(int)
mileage_pkm = mileage_pkm.set_index('year')
mileage_pkm = mileage_pkm['mileage'] * mileage_pkm['occupancy_rate']

veh_demand = veh_demand_pkm['demand_annual'] / mileage_pkm

import matplotlib.pyplot as plt
# Create a stacked area plot
capacity_df.plot.area()
# Set the title and labels
# Add the veh_demand data as a line plot
veh_demand.plot(ax=plt.gca(), color='black', linewidth=2, label='veh_demand')
plt.title('tra_road_mcar')
plt.xlabel('Year')
plt.ylabel('No. cars in 1000')
# Place the legend outside the plot
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# Adjust layout to make room for the legend
plt.tight_layout()
# Show the plot
# plt.show()
plt.savefig('capacity_stack.png')
#%%
import fine as fn
# plot for operation of source components
source_list = [
    comp_name
    for comp_name in da.esM.componentModelingDict['SourceSinkModel'].componentsDict.keys()
    if da.esM.getComponent(comp_name).sign ==1
    if comp_name != 'slack_source_mcar_demand'
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
ax = source_operation.plot(kind='bar', stacked=True, figsize=(10, 8))

# Set the title and labels
plt.title('Primary Energy Import')
plt.xlabel('Year')
plt.ylabel('Operation')

# Place the legend outside the plot
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Adjust layout to make room for the legend
plt.tight_layout()

# Show the plot
# plt.show()
plt.savefig('pri_energy_import.png')

#%%
writeOptimizationOutputToExcel(da.esM, "output/fine_sedos_mcar", optSumOutputLevel=2, optValOutputLevel=1)
writeEnergySystemModelToNetCDF(da.esM, "output/fine_sedos_mcar.nc")
print('test')
# from data_adapter import databus

# from modelBuilder import *
# import pathlib
# import FINE as fn
#
#
# df1 = preprocessing.get_process("hack-a-thon", 'modex_tech_wind_turbine_onshore', 'hack-a-thon_links')
# df2 = preprocessing.get_process("hack-a-thon", 'modex_demand', 'hack-a-thon_links')
#
# collections_folder = pathlib.Path(__file__).parent / "collections"
# url = "https://energy.databus.dbpedia.org/felixmaur/collections/hack-a-thon/"
# # databus.download_collection(url,collections_folder)
#
# collection_dict = structure.get_energy_structure('hack-a-thon_by_file_name')
#
# esM_dict = create_energy_system_dict()
#
# esM = build_model(esM_dict)

