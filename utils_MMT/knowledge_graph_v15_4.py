import os
from collections import defaultdict

# Third-party library imports
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import statistics
from collections import Counter

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

# IPython imports (for embedding debugger)
from IPython import embed

# Type hinting
from typing import List, Tuple, Dict, Any


def generate_substructures_with_full_connections(mol_input, nmr_shifts, radius=1):
    # Convert SMILES to mol object if input is a SMILES string
    if isinstance(mol_input, str):
        mol = Chem.MolFromSmiles(mol_input)
    else:
        mol = mol_input

    if mol is None:
        return None
    
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)
    
    substructures = []
    for _, environments in info.items():
        for atom_idx, radius in environments:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
            amap = {}
            submol = Chem.PathToSubmol(mol, env, atomMap=amap)
            if submol is not None and submol.GetNumAtoms() > 0:
                center_idx = amap[atom_idx]
                subsmiles = Chem.MolToSmiles(submol)
                atom = mol.GetAtomWithIdx(atom_idx)
                atom_symbol = atom.GetSymbol()
                c13_shift = nmr_shifts["13C"].get(atom_idx) if atom_symbol == 'C' else None
                h1_shifts = nmr_shifts["1H"].get(atom_idx, []) if atom_symbol in ['C', 'H'] else []
                hsqc_data = nmr_shifts["HSQC"].get(int(atom_idx)) if atom_symbol == 'C' else None
                cosy_data = nmr_shifts["COSY"].get(atom_idx, []) if atom_symbol in ['C', 'H'] else []
                
                submol_connection_points = []
                mol_connection_points = []
                connection_mapping = {}
                reverse_amap = {v: k for k, v in amap.items()}
                for submol_idx in range(submol.GetNumAtoms()):
                    if submol_idx in reverse_amap:
                        orig_idx = reverse_amap[submol_idx]
                        submol_atom = submol.GetAtomWithIdx(submol_idx)
                        orig_atom = mol.GetAtomWithIdx(orig_idx)
                        if submol_atom.GetDegree() < orig_atom.GetDegree():
                            submol_connection_points.append(submol_idx)
                            mol_connection_points.append(orig_idx)
                            for neighbor in orig_atom.GetNeighbors():
                                if neighbor.GetIdx() not in reverse_amap.values():
                                    connection_mapping[submol_idx] = connection_mapping.get(submol_idx, []) + [neighbor.GetIdx()]
                
                #num_open_connections = len(submol_connection_points)
                num_open_connections = sum(len(connections) for connections in connection_mapping.values())
                
                substructures.append((subsmiles, center_idx, atom_symbol, c13_shift, h1_shifts, hsqc_data, cosy_data, atom_idx, submol_connection_points, mol_connection_points, connection_mapping, num_open_connections, submol))
    
    return substructures



# Define atomic weights
ATOMIC_WEIGHTS = {
    'H': 1.008,   'He': 4.003,  'Li': 6.941,  'Be': 9.012,  'B': 10.811, 
    'C': 12.011,  'N': 14.007,  'O': 15.999,  'F': 18.998,  'Ne': 20.180,
    'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974,
    'S': 32.065,  'Cl': 35.453, 'Ar': 39.948, 'K': 39.098,  'Ca': 40.078,
    'Sc': 44.956, 'Ti': 47.867, 'V': 50.942,  'Cr': 51.996, 'Mn': 54.938,
    'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
    'Ga': 69.723, 'Ge': 72.64,  'As': 74.922, 'Se': 78.96,  'Br': 79.904,
    'Kr': 83.798, 'Rb': 85.468, 'Sr': 87.62,  'Y': 88.906,  'Zr': 91.224,
    'Nb': 92.906, 'Mo': 95.96,  'Tc': 98.0,   'Ru': 101.07, 'Rh': 102.906,
    'Pd': 106.42, 'Ag': 107.868,'Cd': 112.411,'In': 114.818,'Sn': 118.710,
    'Sb': 121.760,'Te': 127.60, 'I': 126.904, 'Xe': 131.293,'Cs': 132.905,
    'Ba': 137.327,'La': 138.905,'Ce': 140.116,'Pr': 140.908,'Nd': 144.242,
    'Pm': 145.0,  'Sm': 150.36, 'Eu': 151.964,'Gd': 157.25, 'Tb': 158.925,
    'Dy': 162.500,'Ho': 164.930,'Er': 167.259,'Tm': 168.934,'Yb': 173.054,
    'Lu': 174.967,'Hf': 178.49, 'Ta': 180.948,'W': 183.84,  'Re': 186.207,
    'Os': 190.23, 'Ir': 192.217,'Pt': 195.084,'Au': 196.967,'Hg': 200.59,
    'Tl': 204.383,'Pb': 207.2,  'Bi': 208.980,'Po': 209.0,  'At': 210.0,
    'Rn': 222.0,  'Fr': 223.0,  'Ra': 226.0,  'Ac': 227.0,  'Th': 232.038,
    'Pa': 231.036,'U': 238.029, 'Np': 237.0,  'Pu': 244.0,  'Am': 243.0
}

def calculate_fragment_weight(substructure, num_open_connections):
    atom_counts = defaultdict(int)
    for atom in substructure.GetAtoms():
        symbol = atom.GetSymbol()
        atom_counts[symbol] += 1
    
    total_weight = sum(ATOMIC_WEIGHTS.get(atom, 0) * count for atom, count in atom_counts.items())
    adjusted_weight = total_weight - (num_open_connections * ATOMIC_WEIGHTS['H'])
    
    return adjusted_weight

def get_connection_mapping_counts(connection_mapping):
    return [len(values) for values in connection_mapping.values()]

def create_knowledge_graph(example_molecules, nmr_data, radius):
    def add_node_if_not_exists(G, node_id, node_type, **attrs):
        if not G.has_node(node_id):
            G.add_node(node_id, node_type=node_type, **attrs)
        else:
            G.nodes[node_id].update(attrs)

    def calculate_mol_weight(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Descriptors.ExactMolWt(mol) if mol else None

    G = nx.Graph()
    central_fragment_node = "AllFragments"
    add_node_if_not_exists(G, central_fragment_node, "CentralFragmentNode", 
                           total_fragments=0, 
                           fragment_occurrences={}, 
                           avg_molecular_weight=0,
                           total_molecules=0)

    unique_fragments = {}
    fragment_occurrences = defaultdict(int)
    total_molecular_weight = 0
    graph_data = []
    next_fragment_id = 1

    for smiles, mol_id in example_molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Could not create molecule from SMILES: {smiles}")
            continue
        
        nmr_shifts = nmr_data[smiles]
        substructures = generate_substructures_with_full_connections(mol, nmr_shifts, radius=radius)
        
        mol_weight = calculate_mol_weight(smiles)
        total_molecular_weight += mol_weight if mol_weight is not None else 0
        
        add_node_if_not_exists(G, mol_id, "Molecule", smiles=smiles, molecular_weight=mol_weight)
        
        molecule_node = ("Molecule", {
            "id": mol_id,
            "smiles": smiles,
            "molecular_weight": mol_weight
        })
        
        fragment_nodes = []
        relationships = []
        
        # For each substructure, we extract the following information:
        # subsmiles: SMILES representation of the substructure
        # center_idx: Index of the central atom in the substructure
        # atom_symbol: Chemical symbol of the central atom (e.g., 'C', 'N')
        # c13_shift: 13C NMR chemical shift for the central atom (if it's carbon)
        # h1_shifts: List of 1H NMR chemical shifts for hydrogens connected to the central atom
        # hsqc_data: HSQC NMR data (correlations between carbons and directly bonded protons)
        # cosy_data: COSY NMR data (correlations between coupled hydrogen atoms)
        # main_mol_idx: Index of the central atom in the context of the main molecule
        # submol_connection_points: Indices of atoms in the substructure connected to the rest of the molecule
        # mol_connection_points: Indices of atoms in the main molecule connected to the substructure
        # connection_mapping: Mapping showing how the substructure connects to the rest of the molecule
        # num_open_connections: Number of connections from this substructure to the rest of the molecule
        # submol: RDKit mol object representing the substructure
        for subsmiles, center_idx, atom_symbol, c13_shift, h1_shifts, hsqc_data, cosy_data, main_mol_idx, submol_connection_points, mol_connection_points, connection_mapping, num_open_connections, submol in substructures:
            fragment_weight = calculate_fragment_weight(submol, num_open_connections)
            connection_mapping_counts = get_connection_mapping_counts(connection_mapping)

            if subsmiles not in unique_fragments:
                fragment_id = f"F{next_fragment_id}"
                next_fragment_id += 1
                unique_fragments[subsmiles] = fragment_id
                add_node_if_not_exists(G, fragment_id, "Fragment", 
                                       smiles=subsmiles, 
                                       molecular_weight=fragment_weight,
                                       occurrences=0,
                                       molecules=[],
                                       center_indices=[],
                                       atom_symbols=[],
                                       c13_shifts=[],
                                       h1_shifts_list=[],
                                       hsqc_data_list=[],
                                       cosy_data_list=[],
                                       submol_connection_points_list=[],
                                       mol_connection_points_list=[],
                                       connection_mapping_list=[],
                                       connection_mapping_counts_list=[],  # New list for counts
                                       num_open_connections_list=[],
                                       main_mol_indices=[],
                                       submols=[])
                G.add_edge(central_fragment_node, fragment_id, relationship="CONTAINS_FRAGMENT")
                G.nodes[central_fragment_node]['total_fragments'] += 1
            else:
                fragment_id = unique_fragments[subsmiles]
            
            G.nodes[fragment_id]['occurrences'] += 1
            G.nodes[fragment_id]['molecules'].append(mol_id)
            G.nodes[fragment_id]['center_indices'].append(center_idx)
            G.nodes[fragment_id]['atom_symbols'].append(atom_symbol)
            G.nodes[fragment_id]['c13_shifts'].append(c13_shift)
            G.nodes[fragment_id]['h1_shifts_list'].append(h1_shifts)
            G.nodes[fragment_id]['hsqc_data_list'].append(hsqc_data)
            G.nodes[fragment_id]['cosy_data_list'].append(cosy_data)
            G.nodes[fragment_id]['submol_connection_points_list'].append(submol_connection_points)
            G.nodes[fragment_id]['mol_connection_points_list'].append(mol_connection_points)
            G.nodes[fragment_id]['connection_mapping_counts_list'].append(connection_mapping_counts)  # Add counts
            G.nodes[fragment_id]['connection_mapping_list'].append(connection_mapping)
            G.nodes[fragment_id]['num_open_connections_list'].append(num_open_connections)
            G.nodes[fragment_id]['main_mol_indices'].append(main_mol_idx)  # Add main_mol_idx to the list
            G.nodes[fragment_id]['submols'].append(submol)
            
            fragment_occurrences[fragment_id] += 1
            
            instance_id = f"{mol_id}_{fragment_id}"
            add_node_if_not_exists(G, instance_id, "FragmentInstance", 
                                   center_index=center_idx, 
                                   center_atom_symbol=atom_symbol,
                                   main_molecule_atom_index=main_mol_idx, 
                                   c13_shift=c13_shift,
                                   h1_shifts=h1_shifts,
                                   hsqc_data=hsqc_data,
                                   cosy_data=cosy_data,
                                   submol_connection_points=submol_connection_points,
                                   mol_connection_points=mol_connection_points,
                                   connection_mapping=connection_mapping,
                                   num_open_connections=num_open_connections,
                                   molecular_weight=fragment_weight,
                                   submol=submol)

            G.add_edge(mol_id, instance_id, relationship="HAS_FRAGMENT", center_atom_index=main_mol_idx)
            G.add_edge(instance_id, fragment_id, relationship="INSTANCE_OF")

            if h1_shifts:
                h1_id = f"{instance_id}_1H"
                add_node_if_not_exists(G, h1_id, "1H_NMR", data=h1_shifts)
                G.add_edge(instance_id, h1_id, relationship="HAS_1H_NMR")

            if c13_shift is not None:
                c13_id = f"{instance_id}_13C"
                add_node_if_not_exists(G, c13_id, "13C_NMR", data=c13_shift)
                G.add_edge(instance_id, c13_id, relationship="HAS_13C_NMR")

            if hsqc_data:
                hsqc_id = f"{instance_id}_HSQC"
                add_node_if_not_exists(G, hsqc_id, "HSQC", data=hsqc_data)
                G.add_edge(instance_id, hsqc_id, relationship="HAS_HSQC")

            if cosy_data:
                cosy_id = f"{instance_id}_COSY"
                add_node_if_not_exists(G, cosy_id, "COSY", data=cosy_data)
                G.add_edge(instance_id, cosy_id, relationship="HAS_COSY")

            fragment_node = ("Fragment", {
                "id": instance_id,
                "fragment_id": fragment_id,
                "smiles": subsmiles,
                "center_index": center_idx,
                "center_atom_symbol": atom_symbol,
                "main_molecule_atom_index": main_mol_idx,
                "c13_shift": c13_shift,
                "h1_shifts": h1_shifts,
                "hsqc_data": hsqc_data,
                "cosy_data": cosy_data,
                "submol_connection_points": submol_connection_points,
                "mol_connection_points": mol_connection_points,
                "connection_mapping": connection_mapping,
                "connection_mapping_counts": connection_mapping_counts,  # Add counts
                "molecular_weight": fragment_weight,
                "num_open_connections": num_open_connections,
                'main_mol_indices': main_mol_idx,
                "submol": submol
            })
            fragment_nodes.append(fragment_node)

            relationship = ("HAS_FRAGMENT", molecule_node, fragment_node, {
                "center_atom_index": main_mol_idx,
            })
            relationships.append(relationship)
        #import IPython; IPython.embed();
        
        graph_data.append((molecule_node, fragment_nodes, relationships))

    G.nodes[central_fragment_node]['fragment_occurrences'] = dict(fragment_occurrences)
    G.nodes[central_fragment_node]['total_molecules'] = len(example_molecules)
    G.nodes[central_fragment_node]['avg_molecular_weight'] = total_molecular_weight / len(example_molecules) if example_molecules else 0

    return G, graph_data


import networkx as nx
import matplotlib.pyplot as plt

def visualize_and_print_graph_data(G, central_fragment_node):
    def plot_graph(G):
        plt.figure(figsize=(30, 30))
        pos = nx.spring_layout(G, k=0.5, iterations=50)

        node_colors = {
            'Molecule': 'lightblue',
            'Fragment': 'lightgreen',
            'FragmentInstance': 'lightpink',
            'HSQC': 'yellow',
            'COSY': 'orange',
            'CentralFragmentNode': 'red',
            '1H_NMR': 'purple',
            '13C_NMR': 'cyan'
        }

        for node_type in node_colors:
            nx.draw_networkx_nodes(G, pos, 
                                   nodelist=[node for node, data in G.nodes(data=True) if data['node_type'] == node_type],
                                   node_color=node_colors[node_type], 
                                   node_size=3000 if node_type != 'CentralFragmentNode' else 5000, 
                                   alpha=0.8)

        edge_colors = {
            'HAS_FRAGMENT': 'red',
            'INSTANCE_OF': 'blue',
            'HAS_HSQC': 'yellow',
            'HAS_COSY': 'orange',
            'CONTAINS_FRAGMENT': 'black',
            'HAS_1H_NMR': 'purple',
            'HAS_13C_NMR': 'cyan'
        }

        for relationship in edge_colors:
            nx.draw_networkx_edges(G, pos, 
                                   edgelist=[(u, v) for (u, v, d) in G.edges(data=True) if d['relationship'] == relationship],
                                   edge_color=edge_colors[relationship], 
                                   width=1.5, 
                                   alpha=0.7)

        node_labels = {}
        for node in G.nodes():
            if G.nodes[node]['node_type'] == 'FragmentInstance':
                parts = node.split('_')
                fragment_id = parts[1]
                node_labels[node] = f"{parts[0]}_{fragment_id}"
            else:
                node_labels[node] = node

        nx.draw_networkx_labels(G, pos, node_labels, font_size=20)

        plt.title("Molecular Graph with NMR Data", fontsize=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def print_graph_data(G, central_fragment_node):
        central_node = G.nodes[central_fragment_node]
        print(f"Total unique fragments: {central_node['total_fragments']}")
        print(f"Total molecules: {central_node['total_molecules']}")
        print(f"Average molecular weight: {central_node['avg_molecular_weight']:.2f}")

        print("\nFragment occurrences and weights:")
        for fragment, count in central_node['fragment_occurrences'].items():
            fragment_data = G.nodes[fragment]
            print(f"  {fragment}: count={count}")
            print(f"    Available data keys: {', '.join(fragment_data.keys())}")
            if 'molecular_weight' in fragment_data:
                if fragment_data['molecular_weight'] is not None:
                    print(f"    Weight: {fragment_data['molecular_weight']:.2f}")
                else:
                    print("    Weight: None")
            else:
                print("    Weight: Not available")
            if 'smiles' in fragment_data:
                print(f"    SMILES: {fragment_data['smiles']}")
            else:
                print("    SMILES: Not available")

        print("\nMolecule weights:")
        for mol_id in [node for node, data in G.nodes(data=True) if data['node_type'] == 'Molecule']:
            mol_data = G.nodes[mol_id]
            print(f"  {mol_id}:")
            print(f"    Available data keys: {', '.join(mol_data.keys())}")
            if 'molecular_weight' in mol_data:
                if mol_data['molecular_weight'] is not None:
                    print(f"    Weight: {mol_data['molecular_weight']:.2f}")
                else:
                    print("    Weight: None")
            else:
                print("    Weight: Not available")
            if 'smiles' in mol_data:
                print(f"    SMILES: {mol_data['smiles']}")
            else:
                print("    SMILES: Not available")

        '''print("\nChecking FragmentInstance connections:")
        for node, data in G.nodes(data=True):
            if data['node_type'] == 'FragmentInstance':
                print(f"  {node}:")
                print(f"    Available data keys: {', '.join(data.keys())}")
                fragment_connections = [n for n in G.neighbors(node) if G.nodes[n]['node_type'] == 'Fragment']
                if len(fragment_connections) == 1:
                    connected_fragment = fragment_connections[0]
                    print(f"    Connected to Fragment: {connected_fragment}")
                    expected_fragment_id = node.split('_')[1]
                    if connected_fragment != expected_fragment_id:
                        print(f"    WARNING: Connected to {connected_fragment}, but expected {expected_fragment_id}")
                else:
                    print(f"    WARNING: Connected to {len(fragment_connections)} Fragments (should be 1)")
            '''
    # Execute the functions
    plot_graph(G)
    #print_graph_data(G, central_fragment_node)


def get_all_fragments(G):
    fragments = []
    for node, data in G.nodes(data=True):
        if data['node_type'] == 'Fragment':
            fragment = {
                'id': node,
                'smiles': data.get('smiles', ''),
                'molecular_weight': data.get('molecular_weight'),
                'occurrences': data.get('occurrences', 0),
                'molecules': data.get('molecules', []),
                'center_indices': data.get('center_indices', []),
                'atom_symbols': data.get('atom_symbols', []),
                'c13_shifts': data.get('c13_shifts', []),
                'h1_shifts_list': data.get('h1_shifts_list', []),
                'hsqc_data_list': data.get('hsqc_data_list', []),
                'cosy_data_list': data.get('cosy_data_list', []),
                'submol_connection_points_list': data.get('submol_connection_points_list', []),
                'mol_connection_points_list': data.get('mol_connection_points_list', []),
                'connection_mapping_list': data.get('connection_mapping_list', []),
                'num_open_connections_list': data.get('num_open_connections_list', []),
                'submols': data.get('submols', [])
            }
            
            # Calculate and add new categories
            connection_mapping_list = fragment['connection_mapping_list']
            fragment['num_connection_points_sub_list'] = [len(mapping) for mapping in connection_mapping_list]
            fragment['num_connection_points_core_list'] = [
                len([item for sublist in mapping.values() for item in (sublist if isinstance(sublist, list) else [sublist])])
                for mapping in connection_mapping_list
            ]
            
            fragments.append(fragment)
    return fragments




def filter_fragments(G, filters):
    matching_fragments = []
    
    for node, data in G.nodes(data=True):
        if data.get('node_type') == 'Fragment':
            fragment_data = {
                'id': node,
                'smiles': data.get('smiles', ''),
                'molecular_weight': data.get('molecular_weight'),
                'occurrences': data.get('occurrences', 0),
                'submol_connection_points': data.get('submol_connection_points_list', []),
                'c13_shifts': data.get('c13_shifts', []),
                'h1_shifts_list': data.get('h1_shifts_list', []),
                'hsqc_data_list': data.get('hsqc_data_list', []),
                'num_open_connections': data.get('num_open_connections_list', []),
                'submol_connection_points_list': data.get('submol_connection_points_list', []),
                'mol_connection_points_list': data.get('mol_connection_points_list', []),
                'connection_mapping_list': data.get('connection_mapping_list', []),
                'connection_mapping_counts_list': data.get('connection_mapping_counts_list', []),  # New field
                'submols': data.get('submols', [])
            }
            matching_samples = []
            for i in range(fragment_data['occurrences']):
                connection_mapping = fragment_data['connection_mapping_list'][i]
                connection_mapping_counts = fragment_data['connection_mapping_counts_list'][i]  # New field
                sample = {
                    'molecular_weight': fragment_data['molecular_weight'],
                    'c13_shift': fragment_data['c13_shifts'][i],
                    'h1_shift': fragment_data['h1_shifts_list'][i][0][0] if fragment_data['h1_shifts_list'][i] and fragment_data['h1_shifts_list'][i][0] else None,
                    'hsqc_data': fragment_data['hsqc_data_list'][i],
                    'num_open_connections': fragment_data['num_open_connections'][i],
                    'submol_connection_points': fragment_data['submol_connection_points_list'][i],
                    'mol_connection_points': fragment_data['mol_connection_points_list'][i],
                    'connection_mapping': connection_mapping,
                    'connection_mapping_counts': connection_mapping_counts,  # New field
                    'submol': fragment_data['submols'][i],
                    'num_connection_points_sub': len(connection_mapping),
                    'num_connection_points_core': len([item for sublist in connection_mapping.values() for item in (sublist if isinstance(sublist, list) else [sublist])])
                }
                
                if all(apply_filter(sample, filter_name, filter_value) for filter_name, filter_value in filters.items()):
                    matching_samples.append(sample)
            
            if matching_samples:
                matching_fragments.append({
                    'id': node,
                    'smiles': fragment_data['smiles'],
                    'weight': fragment_data['molecular_weight'],
                    'occurrences': len(matching_samples),
                    'matching_samples': matching_samples
                })
    
    return matching_fragments

def apply_filter(sample, filter_name, filter_value):
    if filter_name == 'weight_range':
        weight = sample['molecular_weight']
        return weight is not None and filter_value[0] <= weight <= filter_value[1]
    elif filter_name == 'num_connection_points':
        return sample['num_open_connections'] in filter_value
    elif filter_name == 'num_connection_points_core':
        return sample['num_connection_points_core'] in filter_value
    elif filter_name == 'num_connection_points_sub':
        return sample['num_connection_points_sub'] in filter_value
    elif filter_name == 'c13_shift_range':
        if sample['c13_shift'] is None:
            return False
        target_ppm, delta_percent = filter_value
        delta_ppm = (delta_percent / 100) * target_ppm
        return target_ppm - delta_ppm <= sample['c13_shift'] <= target_ppm + delta_ppm
    elif filter_name == 'h1_shift_range':
        if sample['h1_shift'] is None:
            return False
        target_ppm, delta_percent = filter_value
        delta_ppm = (delta_percent / 100) * 10  # 10 ppm is the total range for 1H
        return target_ppm - delta_ppm <= sample['h1_shift'] <= target_ppm + delta_ppm
    elif filter_name == 'hsqc_shift_range':
        if not sample['hsqc_data'] or sample['hsqc_data'][0] is None or sample['hsqc_data'][1] is None:
            return False
        c13_target, h1_target, delta_percent = filter_value
        c13_shift, h1_shift = sample['hsqc_data']
        
        h1_delta_ppm = (delta_percent / 100) * 10    # 10 ppm total range for 1H
        c13_delta_ppm = (delta_percent / 100) * 200  # 200 ppm total range for 13C
        
        h1_in_range = h1_target - h1_delta_ppm <= h1_shift <= h1_target + h1_delta_ppm
        c13_in_range = c13_target - c13_delta_ppm <= c13_shift <= c13_target + c13_delta_ppm
        
        return h1_in_range and c13_in_range
    elif filter_name == 'connection_mapping_counts':
        return sample['connection_mapping_counts'] == filter_value  # Exact match
    return True  # Default case: no filtering

import networkx as nx
import pickle

def convert_dict_to_networkx(graph_dict):
    """
    Convert a dictionary containing graph data back to a NetworkX graph.
    Handles edges stored as a list of tuples.
    
    Parameters:
    -----------
    graph_dict : dict
        Dictionary with 'nodes' and 'edges' data
        
    Returns:
    --------
    networkx.Graph
        Reconstructed NetworkX graph
    """
    G = nx.Graph()
    
    # Add nodes with all their attributes
    for node, attrs in graph_dict['nodes'].items():
        G.add_node(node, **attrs)
    
    # Add edges (handling list format)
    for edge in graph_dict['edges']:
        # Edge might be stored in different formats, handle each case
        if isinstance(edge, tuple):
            if len(edge) == 2:
                node1, node2 = edge
                G.add_edge(node1, node2)
            elif len(edge) == 3:
                node1, node2, attrs = edge
                G.add_edge(node1, node2, **attrs)
        elif isinstance(edge, list):
            if len(edge) == 2:
                node1, node2 = edge
                G.add_edge(node1, node2)
            elif len(edge) == 3:
                node1, node2, attrs = edge
                G.add_edge(node1, node2, **attrs)
    
    return G