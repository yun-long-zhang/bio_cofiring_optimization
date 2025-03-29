from gurobipy import *
from pulp import *
import pandas as pd
import networkx as nx
import os
import time
import matplotlib.pyplot as plt  # Kept for potential visualization needs

# ---------- Data Loading ----------
DATA_DIR = "data"
INPUT_FILE = "input.xlsx"  # Original filename maintained
ROUTES_FILE = "Dji.csv"

# Load biomass collection points
collection_points = pd.read_excel(
    os.path.join(DATA_DIR, INPUT_FILE),
    engine="openpyxl",
    sheet_name="biomass"
)

# Load power plant data
power_plants = pd.read_excel(
    os.path.join(DATA_DIR, INPUT_FILE),
    engine="openpyxl",
    sheet_name="plant"
)

# Load transportation routes
transport_routes = pd.read_csv(os.path.join(DATA_DIR, ROUTES_FILE))

# ---------- Network Construction ----------
transport_network = nx.DiGraph()

# Add collection nodes
for _, row in collection_points.iterrows():
    transport_network.add_node(
        f'Collection_{row.biomass_site}',
        longitude=row.longitude,
        latitude=row.latitude,
        capacity=row.biomass_energy,
        node_type='collection'
    )

# Add power plant nodes
for _, row in power_plants.iterrows():
    transport_network.add_node(
        f'Plant_{row.Plant_ID}',
        longitude=row.Longitude,
        latitude=row.Latitude,
        capacity=row.Installed_Capacity_Xi,
        operating_hours=row.Operating_Hours,
        energy_demand=row.Energy_Requirement,
        coal_price=row.Coal_Price,
        annuity_factor=row.Annuity_Factor,
        node_type='plant'
    )

# Add transportation edges
for _, row in transport_routes.iterrows():
    transport_network.add_edge(
        f'Collection_{row.biomass_site}',
        f'Plant_{row.Plant_ID}',
        distance=row.Dji
    )

# ---------- Optimization Model Setup ----------
model = LpProblem("CBECCS_Transport_Optimization", LpMinimize)

# ---------- Decision Variables ----------
plant_nodes = [n for n, d in transport_network.nodes(data=True) if d['node_type'] == 'plant']

# Biomass blending ratio (20-100% in 20% increments)
biomass_ratio = LpVariable.dicts(
    "BlendingRatio", 
    plant_nodes, 
    lowBound=0, 
    upBound=5,
    cat="Integer"
)

# Retrofit decision binary variables
retrofit_flag = LpVariable.dicts("RetrofitDecision", plant_nodes, cat="Binary")

# Auxiliary variables for linearization
auxiliary_var = LpVariable.dicts("AuxVar", plant_nodes, lowBound=0, upBound=1)

# Transportation flow variables
shipment_flow = LpVariable.dicts("TransportFlow", transport_network.edges(), lowBound=0)

# ---------- Objective Function ----------
# Transportation cost component
transport_cost = lpSum(
    0.0017 * 1.5 * data['distance'] * shipment_flow[(src, dest)] +  # Distance cost
    2.474 * shipment_flow[(src, dest)]                              # Fixed transport cost
    for (src, dest, data) in transport_network.edges(data=True)
)

# Biomass conversion cost component
conversion_cost = lpSum(
    1.02 * (
        1153000 * 0.2 * biomass_ratio[plant] * data['capacity'] +  # Capital cost
        4628000 * auxiliary_var[plant] * data['capacity']          # Operational cost
    ) * data['annuity_factor'] / 6.8589 - 
    data['coal_price'] * data['energy_demand'] * biomass_ratio[plant]
    for plant, data in transport_network.nodes(data=True) 
    if data['node_type'] == 'plant'
)

model += transport_cost + conversion_cost

# ---------- Constraints ----------
# Plant demand fulfillment
for plant, data in transport_network.nodes(data=True):
    if data['node_type'] == 'plant':
        model += (
            lpSum(shipment_flow[(src, plant)] for src in transport_network.predecessors(plant)) 
            == data['energy_demand'] * biomass_ratio[plant] * 0.2,
            f"Demand_Fulfillment_{plant}"
        )

# Collection capacity constraints
for node, data in transport_network.nodes(data=True):
    if data['node_type'] == 'collection':
        model += (
            lpSum(shipment_flow[(node, dest)] for dest in transport_network.successors(node)) 
            <= data['capacity'],
            f"Capacity_Limit_{node}"
        )

# Emission reduction constraint
model += (
    lpSum(
        0.0901 * data['energy_demand'] * (1 - 0.2 * biomass_ratio[plant])
        for plant, data in transport_network.nodes(data=True) 
        if data['node_type'] == 'plant'
    ) <= 0.65 * 3562837317,
    "Emission_Reduction_Target"
)

# Linearization constraints using Big-M method
M = 1e12  # Large constant for big-M constraints
for plant in plant_nodes:
    plant_data = transport_network.nodes[plant]
    model += 0.2 * biomass_ratio[plant] <= 0.2 + M * retrofit_flag[plant]
    model += 0.2 * biomass_ratio[plant] >= 0.21 - M * (1 - retrofit_flag[plant])
    model += auxiliary_var[plant] >= 0.2 * biomass_ratio[plant] - M * (1 - retrofit_flag[plant])
    model += auxiliary_var[plant] <= 0.2 * biomass_ratio[plant] + M * (1 - retrofit_flag[plant])
    model += auxiliary_var[plant] <= M * retrofit_flag[plant]

# ---------- Model Solving ----------
start_time = time.perf_counter()
model.solve(GUROBI_CMD(options=[('MIPGap', '0.0003')]))
solve_duration = time.perf_counter() - start_time

print(f"Solving time: {solve_duration:.2f} seconds")
print(f"Solution status: {LpStatus[model.status]}")

# ---------- Results Processing ----------
# Transport results
transport_results = []
for (src, dest, data) in transport_network.edges(data=True):
    flow = value(shipment_flow[(src, dest)])
    if flow > 1e-6:  # Filter negligible flows
        transport_results.append({
            "Source": src,
            "Destination": dest,
            "Flow": flow,
            "Source_Lon": transport_network.nodes[src]['longitude'],
            "Source_Lat": transport_network.nodes[src]['latitude'],
            "Dest_Lon": transport_network.nodes[dest]['longitude'],
            "Dest_Lat": transport_network.nodes[dest]['latitude'],
            "TonKm": data['distance'] * flow
        })

transport_df = pd.DataFrame(transport_results).sort_values(by='Destination')
transport_df.to_excel('data/Plant_Transport_Allocation.xlsx', index=False)

# Cost analysis results
cost_analysis = []
for plant, data in transport_network.nodes(data=True):
    if data['node_type'] == 'plant':
        ratio = 0.2 * value(biomass_ratio[plant])
        cost_breakdown = {
            "Plant": plant,
            "BlendingRatio": ratio,
            "RetrofitDecision": value(retrofit_flag[plant]),
            "AuxVariable": value(auxiliary_var[plant]),
            "CapitalCost$": data['capacity'] * (1153000*0.2*value(biomass_ratio[plant]) 
                              + 4628000*value(auxiliary_var[plant])) * data['annuity_factor']/6.8589,
            "O&MCost$": 0.02 * data['capacity'] * (1153000*0.2*value(biomass_ratio[plant]) 
                       * data['annuity_factor']/6.8589,
            "CoalSavings$": data['coal_price'] * data['energy_demand'] * value(biomass_ratio[plant])
        }
        cost_analysis.append(cost_breakdown)

pd.DataFrame(cost_analysis).to_excel('data/Plant_Cost.xlsx', index=False)

print(f"Minimum total cost: {value(model.objective):,.2f}")