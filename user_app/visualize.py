import matplotlib
import matplotlib.pyplot as plt
from django.conf import settings
import os
import io
import urllib, base64

def generate_bar_graph(exceeded_limit, within_limit):
    # Get the counts for vehicles that have crossed the speed limit and those that have not
    exceeded_limit_count = len(exceeded_limit)
    within_limit_count = len(within_limit)

    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Create the bar chart
    ax.bar(['Exceeded Speed Limit', 'Within Speed Limit'], [exceeded_limit_count, within_limit_count], color=['red', 'green'])

    # Add labels and title
    ax.set_xlabel('Speed Limit Status')
    ax.set_ylabel('Number of Vehicles')
    ax.set_title('Vehicle Speed Limit')


    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    chart_image = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    # # Save the chart to a file
    # chart_path = 'static/graph.png' # Specify the file path to save the chart image
    # plt.savefig(chart_path)  # Save the chart image to the file

    return chart_image


def generate_line_graph(speeds):
    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Create the line graph
    ax.plot(range(1, len(speeds) + 1), speeds)

    # Add labels and title
    ax.set_xlabel('Record')
    ax.set_ylabel('Speed')
    ax.set_title('Speed Record')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graph_image = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    return graph_image

def generate_permonth_graph(labels, counts):
    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Create the bar chart
    ax.bar(labels, counts, color =["purple"])

    # Add labels and title
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of voilators')
    ax.set_title('Number of Vehicles Exceeding Speed Limit per Month')

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    permonth__image = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    return permonth__image


def generate_perday_graph(labels, counts):
    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Create the bar chart
    ax.bar(labels, counts)
  

    # Add labels and title
    ax.set_xlabel('Day')
    ax.set_ylabel('Number of voilators')
    ax.set_title('Number of Vehicles Exceeding Speed Limit per Month')

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    perday_image = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    return perday_image
