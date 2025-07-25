
from rdkit.Chem.Draw import rdMolDraw2D
import math
import cairosvg
import os
from IPython.display import SVG
import matplotlib.pyplot as plt
import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import base64
from io import BytesIO


#inspired by https://github.com/rdkit/UGM_2020/blob/master/Notebooks/Landrum_WhatsNew.ipynb
def show_mols(mols, mols_per_row = 5, size=200, min_font_size=12, legends=[], file_name=''):
    if legends and len(legends) < len(mols):
      print('legends is too short')
      return None

    mols_per_row = min(len(mols), mols_per_row)
    rows = math.ceil(len(mols)/mols_per_row)
    d2d = rdMolDraw2D.MolDraw2DSVG(mols_per_row*size,rows*size,size,size)
    d2d.drawOptions().minFontSize = min_font_size
    if legends:
       d2d.DrawMolecules(mols, legends=legends)
    else:
        d2d.DrawMolecules(mols)
    d2d.FinishDrawing()

    if file_name:
      with open('d2d.svg', 'w') as f:
        f.write(d2d.GetDrawingText())
        if 'pdf' in file_name:
           cairosvg.svg2pdf(url='d2d.svg', write_to=file_name)
        else:
           cairosvg.svg2png(url='d2d.svg', write_to=file_name)
        os.remove('d2d.svg')

    return SVG(d2d.GetDrawingText())


def plot_molecules_from_list(smiles_list, max_num, img_size=(150, 150), imgs_per_row=5):
    molecules = [Chem.MolFromSmiles(smile) for smile in smiles_list if smile != "NAN"]
    molecules = molecules[:max_num]
    rows = len(molecules) // imgs_per_row + (len(molecules) % imgs_per_row > 0)

    # Calculate figure size: width is fixed, height depends on the number of rows
    fig_width = imgs_per_row * (img_size[0] / 100)  # Convert pixels to inches
    fig_height = rows * (img_size[1] / 100)  # Convert pixels to inches
    fig, axs = plt.subplots(rows, imgs_per_row, figsize=(fig_width, fig_height))
    
    axs = axs.flatten()
    for i in range(len(molecules), len(axs)):
        axs[i].axis('off')
    for i, mol in enumerate(molecules):
        img = Draw.MolToImage(mol, size=img_size)
        axs[i].imshow(img)
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

    # Convert plot to HTML image
    tmpfile = BytesIO()
    plt.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
    plt.close(fig)
    return html

    
def plot_smiles_from_df(data_sorted,column_name):
    """ As the name suggests
    if ID column is given it will be printed with every batch of 9
    if ID is added it also shows the molecular weight to the compound in the name"""
    list_smiles = []
    sample_id_list = []
    for idx, i in enumerate(data_sorted[column_name]):
        weight = hf.calculate_weight(i)
        string = f"{column_name}_{idx}_{weight:.2f}"
        sample_id_list.append(string)
        list_smiles.append(Chem.MolFromSmiles(i))
        if len(list_smiles)==9 or idx == len(data_sorted)-1:
            pic = Draw.MolsToGridImage((list_smiles), subImgSize=(250,250), legends = sample_id_list)
            display(pic)
            list_smiles = []
            sample_id_list = []



def plot_compare_scatter_plot(df_orig, df_sim, name="Plot", transp=0.50, style=["sim","orig","both"], direction=False):
    """Plots scatter plot from two dataframes on top of each other. The direction ticker can be selected to 
    distinguish between positive and negative intensities of the spectra (then displayed in 4 colours
    TODO: the sim or orig alone plotting is not working yet
    """

    scatter_x_1 = list(np.array(df_orig['F2 (ppm)'].astype(float)))
    scatter_y_1 = list(np.array(df_orig['F1 (ppm)'].astype(float)))
    # intensity = np.array(df_orig['Intensity'].astype(float))
    if direction:
        df_orig['Phase'] = np.where(df_orig['Intensity']>=0, '+ve_orig', '-ve_orig')
    else:
        df_orig['Phase'] = np.where(df_orig['Intensity']>=0, '+ve_orig', '+ve_orig')
   

    scatter_x_2 = list(np.array(df_sim['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_sim['F1 (ppm)'].astype(float)))
    if direction:
        df_sim['Phase'] = np.where(df_sim['direction']>=0, '+ve_sim', '-ve_sim')
    else:
        df_sim['Phase'] = np.where(df_sim['direction']>=0, '+ve_sim', '+ve_sim')

     
    
    scatter_x = np.array(scatter_x_1 + scatter_x_2 )             
    scatter_y = np.array(scatter_y_1 + scatter_y_2 ) 
    df_orig = df_orig[['F2 (ppm)', 'F1 (ppm)', "Phase"]]
    df_sim = df_sim[['F2 (ppm)', 'F1 (ppm)', "Phase"]]

    df = pd.concat([df_orig, df_sim], axis=0) 
    group = df['Phase']  

    fig, ax = plt.subplots(figsize=(10,5), dpi=80)
    if (style == "orig") or (style == "both"):
        for g in np.unique(group):
            i = np.where(group == g)
            ax.scatter(scatter_x[i], scatter_y[i], label=g,  alpha=transp)

    ax.legend()
    plt.title(name)
    plt.xlim(xmin=0,xmax=11)
    plt.ylim(ymin=0,ymax=200)

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    plt.grid()
    plt.show()



    
def plot_compare_scatter_plot_without_direction(df_orig, df_sim, name="Plot", transp=0.50):
    """Plots scatter plot from two dataframes on top of each other.
    """
    fig, ax = plt.subplots(figsize=(10,5), dpi=80)
    scatter_x_1 = list(np.array(df_orig['F2 (ppm)'].astype(float)))
    scatter_y_1 = list(np.array(df_orig['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_1, scatter_y_1, label="Real",  alpha=transp, color="blue")

    scatter_x_2 = list(np.array(df_sim['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_sim['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_2, scatter_y_2, label="Simulated",  alpha=transp, color="red")

    ax.legend()
    plt.title(name)
    plt.xlim(xmin=0,xmax=11)
    plt.ylim(ymin=0,ymax=200)

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    plt.grid()
    plt.show()

    
def plot_compare_scatter_plot_without_direction_all(df_real, df_acd, df_mnova, df_dft, df_ml, labels, transp=0.50):
    """Plots scatter plot from all simulation techniques on top of each other.
    """
    fig, ax = plt.subplots(figsize=(10,5), dpi=80)
    scatter_x_1 = list(np.array(df_real['F2 (ppm)'].astype(float)))
    scatter_y_1 = list(np.array(df_real['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_1, scatter_y_1, label=labels[0],  alpha=1, color="green", s=40, marker="s")

    scatter_x_2 = list(np.array(df_acd['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_acd['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_2, scatter_y_2, label=labels[1],  alpha=transp, color="blue")
    
    scatter_x_2 = list(np.array(df_mnova['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_mnova['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_2, scatter_y_2, label=labels[2],  alpha=transp, color="red")
    
    scatter_x_2 = list(np.array(df_dft['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_dft['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_2, scatter_y_2, label=labels[3],  alpha=transp, color="orange")
    
    scatter_x_2 = list(np.array(df_ml['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_ml['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_2, scatter_y_2, label=labels[4],  alpha=transp, color="m")
    
    ax.legend()
    plt.title("All spectra together")
    plt.xlim(xmin=0,xmax=11)
    plt.ylim(ymin=0,ymax=200)

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    plt.grid()
    plt.show()

    