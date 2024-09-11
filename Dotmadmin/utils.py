import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering

import matplotlib.pyplot as plt
import os

def generate_bar_graph(exceeded_limit, within_limit):
    # Extract speeds for both categories
    exceeded_speeds = [record.speed for record in exceeded_limit]
    within_speeds = [record.speed for record in within_limit]

    # Create a bar graph
    plt.figure(figsize=(10, 5))
    plt.hist([exceeded_speeds, within_speeds], bins=range(0, 200, 10), label=['Exceeded Limit', 'Within Limit'], color=['red', 'green'], alpha=0.7)
    plt.xlabel('Speed')
    plt.ylabel('Number of Vehicles')
    plt.title('Vehicle Speed Distribution')
    plt.legend(loc='upper right')

    # Save the plot to a file
    chart_path = 'static/images/speed_distribution.png'
    plt.savefig(chart_path)
    plt.close()

    return chart_path

def generate_permonth_graph(labels, counts):
    # Create a bar graph for counts per month
    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts, color='blue', alpha=0.7)
    plt.xlabel('Month-Year')
    plt.ylabel('Number of Vehicles Exceeding Speed Limit')
    plt.title('Vehicles Exceeding Speed Limit Per Month')
    plt.xticks(rotation=45)

    # Save the plot to a file
    permonth_path = 'static/images/vehicles_per_month.png'
    plt.savefig(permonth_path)
    plt.close()

    return permonth_path
