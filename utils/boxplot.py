import pandas as pd
import numpy as np

from scipy import stats
import plotly.express as px

from plotly.subplots import make_subplots
import plotly.graph_objects as go

ROC = 1
PR = 2

def add_p_value_annotation(fig, array_columns, subplot=None, _format=dict(interline=0.03, text_height=1.03, color='black')):
    ''' Adds notations giving the p-value between two box plot data (t-test two-sided comparison)
    
    Parameters:
    ----------
    fig: figure
        plotly boxplot figure
    array_columns: np.array
        array of which columns to compare 
        e.g.: [[0,1], [1,2]] compares column 0 with 1 and 1 with 2
    subplot: None or int
        specifies if the figures has subplots and what subplot to add the notation to
    _format: dict
        format characteristics for the lines

    Returns:
    -------
    fig: figure
        figure with the added notation
    '''
    # Specify in what y_range to plot for each pair of columns
    y_range = np.zeros([len(array_columns), 2])
    for i in range(len(array_columns)):
        y_range[i] = [1.03+i*_format['interline'], 1.04+i*_format['interline']]

    # Get values from figure
    fig_dict = fig.to_dict()

    # Get indices if working with subplots
    if subplot:
        if subplot == 1:
            subplot_str = ''
        else:
            subplot_str =str(subplot)
        indices = [] #Change the box index to the indices of the data for that subplot
        for index, data in enumerate(fig_dict['data']):
            #print(index, data['xaxis'], 'x' + subplot_str)
            if data['xaxis'] == 'x' + subplot_str:
                indices = np.append(indices, index)
        indices = [int(i) for i in indices]
        print((indices))
    else:
        subplot_str = ''

    # Print the p-values
    for index, column_pair in enumerate(array_columns):
        if subplot:
            data_pair = [indices[column_pair[0]], indices[column_pair[1]]]
        else:
            data_pair = column_pair

        # Mare sure it is selecting the data and subplot you want
        #print('0:', fig_dict['data'][data_pair[0]]['name'], fig_dict['data'][data_pair[0]]['xaxis'])
        #print('1:', fig_dict['data'][data_pair[1]]['name'], fig_dict['data'][data_pair[1]]['xaxis'])

        # Get the p-value
        pvalue = stats.ttest_ind(
            fig_dict['data'][data_pair[0]]['y'],
            fig_dict['data'][data_pair[1]]['y'],
            equal_var=False,
        )[1]
        if pvalue >= 0.05:
            symbol = 'ns'
        elif pvalue >= 0.01: 
            symbol = '*'
        elif pvalue >= 0.001:
            symbol = '**'
        else:
            symbol = '***'
        # Vertical line
        fig.add_shape(type="line",
            xref="x"+subplot_str, yref="y"+subplot_str+" domain",
            x0=column_pair[0], y0=y_range[index][0], 
            x1=column_pair[0], y1=y_range[index][1],
            line=dict(color=_format['color'], width=1.5,)
        )
        # Horizontal line
        fig.add_shape(type="line",
            xref="x"+subplot_str, yref="y"+subplot_str+" domain",
            x0=column_pair[0], y0=y_range[index][1], 
            x1=column_pair[1], y1=y_range[index][1],
            line=dict(color=_format['color'], width=1.5,)
        )
        # Vertical line
        fig.add_shape(type="line",
            xref="x"+subplot_str, yref="y"+subplot_str+" domain",
            x0=column_pair[1], y0=y_range[index][0], 
            x1=column_pair[1], y1=y_range[index][1],
            line=dict(color=_format['color'], width=1.5,)
        )
        ## add text at the correct x, y coordinates
        ## for bars, there is a direct mapping from the bar number to 0, 1, 2...
        fig.add_annotation(dict(font=dict(color=_format['color'],size=14),
            x=(column_pair[0] + column_pair[1])/2,
            y=y_range[index][1]*_format['text_height'],
            showarrow=False,
            text=symbol,
            textangle=0,
            xref="x"+subplot_str,
            yref="y"+subplot_str+" domain"
        ))
    return fig


def box_plot(df):
    
    fig = px.box(df, x = 'Task_name', y='test_auroc', color="Model")
   
    fig.update_layout(plot_bgcolor="white")
    fig.update_xaxes(linecolor='rgba(0,0,0,0.25)', gridcolor='rgba(0,0,0,0)',mirror=False)
    fig.update_yaxes(linecolor='rgba(0,0,0,0.25)', gridcolor='rgba(0,0,0,0.07)',mirror=False)
    fig.update_layout(title={'text': "<b>ROC-AUC score distribution</b>",
                             'font':{'size':40},
                             'y': 0.96,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},

                      xaxis_title={'text': "Datasets",
                             'font':{'size':30}},
                      yaxis_title={'text': "ROC-AUC",
                             'font':{'size':30}},

                      font=dict(family="Calibri, monospace",
                                size=17
                                ))

    fig = add_p_value_annotation(fig, [[0,7], [3,7], [6,7]], subplot=1)

    fig.write_image('../figures/box_plot_integration.png', width=1.5*1200, height=0.75*1200, scale=2)
    fig.show()
    


def go_box_plot(df, metric = ROC):
    dataset_list = ['BIOSNAP', 'DAVIS', 'BindingDB']
    model_list = ['LR', 'DNN', 'GNN-CPI', 'DeepDTI', 'DeepDTA', 'DeepConv-DTI', 'Moltrans', 'ours']
    clr_list = ['red', 'orange', 'green', 'indianred', 'lightseagreen', 'goldenrod', 'magenta', 'blue']

    if metric == ROC:
        # fig_title = "<b>ROC-AUC score distribution</b>"
        file_title = "boxplot_auroc.png"
        select_metric = "test_auroc"
    else:
        # fig_title = "<b>PR-AUC score distribution</b>"
        file_title = "boxplot_auprc.png"
        select_metric = "test_auprc"

    fig = make_subplots(rows=1, cols=3, subplot_titles=[c for c in dataset_list])

    groups = df.groupby(df.Task_name)
    Legand = True

    for dataset_idx, dataset in enumerate(dataset_list):
            df_modelgroup = groups.get_group(dataset)
            model_groups = df_modelgroup.groupby(df_modelgroup.Model)
            if dataset_idx != 0:
                    Legand = False
            for model_idx, model in enumerate(model_list):
                    df_data = model_groups.get_group(model)
                    fig.append_trace(go.Box(y=df_data[select_metric],
                                name=model,
                                marker_color=clr_list[model_idx],
                                showlegend = Legand
                                ),
                                row=1,
                                col=dataset_idx+1)


    

    # fig.update_layout(title={'text': fig_title,
    #                         'font':{'size':25},
    #                         'y': 0.98,
    #                         'x': 0.46,
    #                         'xanchor': 'center',
    #                         'yanchor': 'top'})

    #    fig = add_p_value_annotation(fig, [[0,7], [3,7], [6,7]], subplot=1)
    #    fig = add_p_value_annotation(fig, [[0,7], [3,7], [6,7]], subplot=2)
    #    fig = add_p_value_annotation(fig, [[0,7], [3,7], [6,7]], subplot=3)

    fig.write_image(f'../figures/{file_title}', width=1.5*1200, height=0.75*1200, scale=2)
    fig.show()


if __name__ == '__main__':
    df = pd.read_csv("../dataset/wandb_export_boxplotdata.csv")
    box_plot(df)