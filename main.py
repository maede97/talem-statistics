import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pyproj import Transformer
import folium
from folium.plugins import HeatMap

# Example: load your CSV
df = pd.read_csv("orders.csv")
df['Order Date'] = pd.to_datetime(df['Order Date'])

unique_orders = df.drop_duplicates(subset='Order Number').copy()

print("Total unique orders:", len(unique_orders))
print("Total gewinn:", round(unique_orders['Order Subtotal Amount'].sum(), 2))
print("Total discount:", round(unique_orders['Cart Discount Amount'].sum(), 2))
print("Total shipping:", round(unique_orders['Order Shipping Amount'].sum(), 2))
# Extract day and hour
df['Order Day'] = df['Order Date'].dt.date
df['Order Hour'] = df['Order Date'].dt.hour

# Group by day and sum total amount
df['full_amount'] = df['Quantity (- Refund)'] * df['Item Cost']
daily_amount = df.groupby('Order Day')['full_amount'].sum().reset_index()
hourly_amount = df.groupby('Order Hour')['full_amount'].sum().reset_index()
all_hours = pd.DataFrame({'Order Hour': np.arange(24)})
hourly_amount = all_hours.merge(hourly_amount, on='Order Hour', how='left').fillna(0)
first_half = hourly_amount[hourly_amount['Order Hour'] < 12].copy()
second_half = hourly_amount[hourly_amount['Order Hour'] >= 12].copy()
first_half['theta'] = 2 * np.pi * (first_half['Order Hour']) / 12
second_half['theta'] = 2 * np.pi * (second_half['Order Hour'] - 12) / 12

plt.style.use('seaborn-v0_8-white')

plt.figure(figsize=(14,6))
sns.lineplot(data=daily_amount, x='Order Day', y='full_amount')
plt.title('Total Order Amount per Day')
plt.xlabel('Date')
plt.ylabel('Order Total Amount')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('daily_total.png')




fig = plt.figure(figsize=(14,7))
# Helper function to draw a clock
def draw_clock(ax, df_half, hour_labels, title, color):
    theta = df_half['theta'].values
    r = df_half['full_amount'].values
    
    # Bars
    bars = ax.bar(theta, r, width=2*np.pi/12*0.8, bottom=0,
                  color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Clock orientation
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2)
    
    # Hour labels
    ax.set_xticks(2*np.pi*np.arange(12)/12)
    ax.set_xticklabels(hour_labels, fontsize=10)
    
    # Remove radial labels and grid
    ax.set_yticklabels([])
    ax.grid(alpha=0.2)
    
    # Title
    ax.set_title(title, y=1.1, fontsize=14, weight='bold')
    
    # Annotate values at top of bars
    for bar, val in zip(bars, r):
        angle = bar.get_x() + bar.get_width()/2
        radius = bar.get_height() + (max(r)*0.05)  # slightly outside bar
        ax.text(angle, radius, f'{val:.0f}', ha='center', va='center', fontsize=9)

# === 3. Plot first clock (0–11) ===
ax1 = plt.subplot(1, 2, 1, polar=True)
draw_clock(ax1, first_half, hour_labels=[str(h) for h in range(12)],
           title='Orders by Hour (0–11)', color='skyblue')

# === 4. Plot second clock (12–23) ===
ax2 = plt.subplot(1, 2, 2, polar=True)
draw_clock(ax2, second_half, hour_labels=[str(h) for h in range(12,24)],
           title='Orders by Hour (12–23)', color='lightcoral')

plt.tight_layout()
plt.savefig('hourly_clocks.png')



item_stats = (
    df
    .groupby('Item Name', as_index=False)['Quantity (- Refund)']
    .sum()
    .sort_values('Quantity (- Refund)', ascending=False)
)

item_stats = item_stats[item_stats['Quantity (- Refund)'] > 0]

plt.figure(figsize=(10,6))
sns.barplot(
    data=item_stats,
    y='Item Name',
    x='Quantity (- Refund)',
    color='steelblue'
)
plt.title('Total Quantity Sold by Item')
plt.xlabel('Quantity Sold')
plt.ylabel('Item Name')
plt.tight_layout()
plt.savefig('item_quantity.png')



plz_geo = pd.read_csv('res/plz_geo.csv', sep=';')
plz_geo = plz_geo[['PLZ','E','N']]  # only PLZ and coordinates

transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
plz_geo['Longitude'], plz_geo['Latitude'] = transformer.transform(plz_geo['E'].values, plz_geo['N'].values)

unique_orders['PLZ'] = unique_orders['Postcode (Billing)']
orders_geo = unique_orders.merge(plz_geo[['PLZ','Latitude','Longitude']], on='PLZ', how='left')
orders_by_plz = orders_geo.groupby(['PLZ','Latitude','Longitude'], as_index=False)['Order Number'].count()
orders_by_plz = orders_by_plz.rename(columns={'Order Number':'Order Count'})


m = folium.Map(location=[46.8, 8.3], zoom_start=8)

for _, row in orders_by_plz.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=row['Order Count']**0.5,  # scale radius
        popup=f"PLZ: {row['PLZ']}, Orders: {row['Order Count']}",
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(m)


heat_data = orders_by_plz[['Latitude','Longitude','Order Count']].values.tolist()
HeatMap(heat_data, radius=15).add_to(m)
m.save('swiss_orders_heatmap.html')