import matplotlib.pyplot as plt
import io
import base64
from collections import defaultdict

# CO2 savings values for different materials (in grams per 25 grams of material)
CO2_SAVINGS = {
    "Plástico": 50,
    "Metal": 225,
    "Vidrio": 7.5,
    "Papel": 25,
    "Cartón": 22.5,
    "Biodegradable": 62.5  # Assuming Biodegradable is equivalent to Organic
}

def generate_total_co2_per_material(object_records):
    material_points = defaultdict(int)
    
    # Aggregate points by material
    for record in object_records:
        material = record['label']
        material_points[material] += record['points']
    
    # Calculate total CO2 saved for each material
    material_co2 = {}
    for material, points in material_points.items():
        # Calculate total CO2 saved, assuming each record represents 25 grams
        co2_per_25g = CO2_SAVINGS.get(material, 0)
        total_co2 = points * co2_per_25g
        material_co2[material] = total_co2
    
    materials = sorted(material_co2.keys())
    co2_values = [material_co2[material] for material in materials]
    
    return materials, co2_values

def plot_co2_by_material(object_records):
    # Generate the data
    materials, co2_values = generate_total_co2_per_material(object_records)
    plt.switch_backend('Agg')  # Use non-GUI backend
    # Create the bar plot
    plt.figure(figsize=(5, 5))
    plt.bar(materials, co2_values, color=['blue', 'green', 'red', 'purple', 'orange', 'brown'])
    plt.xlabel('Residuo')
    plt.ylabel('CO₂ total ahorrado (gramos)')
    plt.title('Total CO₂ no emitido por material')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    
    return img_str

def generate_historical_data(object_records):
    month_points = defaultdict(int)
    for record in object_records:
        month = record['timestamp'][:7]  # Extract the year and month (YYYY-MM)
        month_points[month] += record['points']

    months = sorted(month_points.keys())
    points = [month_points[month] for month in months]
    
    return months, points

def create_chart(months, points):
    plt.switch_backend('Agg')  # Use non-GUI backend
    plt.figure(figsize=(5, 5))
    plt.plot(months, points, marker='o', linestyle='-', color='b')
    plt.xlabel('Mes')
    plt.ylabel('Puntos')
    plt.title('Tus puntos acumulados')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    
    return img_str