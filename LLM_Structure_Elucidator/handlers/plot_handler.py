"""
Plot request handler for Socket.IO events.
"""
import base64
import pandas as pd
import numpy as np
from flask_socketio import emit
from core.socket import socketio
import traceback
import json

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print("\n[Plot Handler] ====== Client Connected to Plot Handler ======")
    print("[Plot Handler] Socket.IO connection established")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print("\n[Plot Handler] ====== Client Disconnected from Plot Handler ======")

@socketio.on('plot_request')
def handle_plot_request(request_data):
    """Handle plot generation requests."""
    print("\n[Plot Handler] ====== Starting Plot Generation ======")

    try:
        # Validate request data
        if not request_data or not isinstance(request_data, dict):
            error = "Invalid plot request data format"
            print(f"[Plot Handler] ERROR: {error}")
            emit('message', {'content': error, 'type': 'error'})
            return
        
        # Extract and validate plot type
        plot_type = request_data.get('plot_type', '').lower()
        parameters = request_data.get('parameters', {})
        print(f"[Plot Handler] Plot type: {plot_type}")
        
        if not plot_type:
            error = "No plot type specified"
            print(f"[Plot Handler] ERROR: {error}")
            emit('message', {'content': error, 'type': 'error'})
            return
            
        print("[Plot Handler] Creating plot data...")
        
        # Extract NMR data if available
        nmr_data = parameters.get('nmr_data', {})
        x_data = nmr_data.get('x')
        y_data = nmr_data.get('y')
        z_data = nmr_data.get('z') if plot_type in ['hsqc', 'cosy'] else None
        
        # Create plot data based on type
        if plot_type in ['hsqc', 'cosy']:
            print(f"[Plot Handler] Generating {plot_type.upper()} plot data")
            plot_data, layout = generate_2d_plot(plot_type, parameters, x_data, y_data, z_data)
        else:  # 1D spectra (proton or carbon)
            print(f"[Plot Handler] Generating 1D data for {plot_type}")
            plot_data, layout = generate_1d_plot(plot_type, parameters, x_data, y_data)
        
        print("[Plot Handler] Plot data created successfully")

        # Apply plot style
        style = parameters.get('style', 'default')
        apply_plot_style(plot_data, layout, style)
        
        # Create response
        response = {
            'data': [plot_data],
            'layout': layout,
            'type': plot_type,
            'config': {
                'responsive': True,
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['sendDataToCloud', 'lasso2d', 'select2d'],
                'useResizeHandler': True
            }
        }
        
        # First emit a status message
        print("[Plot Handler] Emitting status message")
        emit('message', {
            'content': f"Generating {plot_type.upper()} plot...",
            'type': 'info'
        })
        
        # Then emit the plot data
        print("[Plot Handler] Emitting plot data to client")
        emit('plot', response)
        
        # Finally emit a success message
        emit('message', {
            'content': f"Generated {plot_type.upper()} plot successfully",
            'type': 'success'
        })
        print("[Plot Handler] ====== Plot Generation Complete ======\n")
        
    except Exception as e:
        print(f"[Plot Handler] ERROR: Failed to generate plot: {str(e)}")
        print(f"[Plot Handler] Traceback: {traceback.format_exc()}")
        emit('message', {
            'content': f"Error generating plot: {str(e)}",
            'type': 'error'
        })

def generate_2d_plot(plot_type, parameters, x_data=None, y_data=None, z_data=None):
    """Generate 2D plot data (HSQC or COSY).
    
    Args:
        plot_type (str): Type of plot ('hsqc' or 'cosy')
        parameters (dict): Plot parameters
        x_data (list, optional): X-axis data points
        y_data (list, optional): Y-axis data points
        z_data (list, optional): Z-axis data points for coloring
    
    Returns:
        tuple: (plot_data, layout) for plotly
    """
    if all(data is not None for data in [x_data, y_data, z_data]):
        print("[Plot Handler] Using provided 2D NMR data")
    else:
        print("[Plot Handler] Generating default 2D data")
        # Generate default example data
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)
        
        x_data = X.flatten().tolist()
        y_data = Y.flatten().tolist()
        z_data = Z.flatten().tolist()
    
    plot_data = {
        'type': 'scatter',
        'mode': 'markers',
        'x': x_data,
        'y': y_data,
        'marker': {
            'size': 8,
            'color': z_data,
            'colorscale': 'Viridis',
            'showscale': True
        }
    }
    
    layout = {
        'title': parameters.get('title', f'{plot_type.upper()} NMR Spectrum'),
        'xaxis': {
            'title': parameters.get('x_label', 'F2 (ppm)'),
            'autorange': 'reversed'
        },
        'yaxis': {
            'title': parameters.get('y_label', 'F1 (ppm)'),
            'autorange': 'reversed'
        },
        'showlegend': False,
        'autosize': True,
        'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
    }
    
    return plot_data, layout

def generate_1d_plot(plot_type, parameters, x_data=None, y_data=None):
    """Generate 1D plot data (proton or carbon NMR).
    
    Args:
        plot_type (str): Type of plot ('proton' or 'carbon')
        parameters (dict): Plot parameters
        x_data (list, optional): X-axis data points (chemical shifts)
        y_data (list, optional): Y-axis data points (intensities)
    
    Returns:
        tuple: (plot_data, layout) for plotly
    """
    if x_data is not None and y_data is not None:
        print("[Plot Handler] Using provided 1D NMR data")
    else:
        print("[Plot Handler] Generating default 1D data")
        # Generate default example data
        x = np.linspace(0, 10, 1000)
        y = np.exp(-(x - 5)**2) + 0.5 * np.exp(-(x - 7)**2)
        
        x_data = x.tolist()
        y_data = y.tolist()
    
    plot_data = {
        'type': 'scatter',
        'mode': 'lines',
        'x': x_data,
        'y': y_data,
        'line': {
            'color': 'blue',
            'width': 1
        }
    }
    
    layout = {
        'title': parameters.get('title', f'{plot_type.upper()} NMR Spectrum'),
        'xaxis': {
            'title': parameters.get('x_label', 'Chemical Shift (ppm)'),
            'autorange': 'reversed'
        },
        'yaxis': {
            'title': parameters.get('y_label', 'Intensity'),
        },
        'showlegend': False,
        'autosize': True,
        'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
    }
    
    return plot_data, layout

def apply_plot_style(plot_data, layout, style):
    """Apply visual style to plot."""
    if style == 'publication':
        layout['font'] = {'family': 'Arial', 'size': 14}
        layout['margin'] = {'l': 60, 'r': 20, 't': 40, 'b': 60}
        if 'line' in plot_data:
            plot_data['line']['color'] = 'black'
    elif style == 'presentation':
        layout['font'] = {'family': 'Arial', 'size': 16}
        layout['margin'] = {'l': 80, 'r': 40, 't': 60, 'b': 80}
        if 'line' in plot_data:
            plot_data['line']['color'] = '#1f77b4'  # Professional blue
