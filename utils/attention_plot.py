import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

def make_attention_table(att, tokens, numb, token_idx = 0, layerNumb = -1):
    token_att = att[layerNumb, token_idx, range(1, len(tokens))]
    
    token_label=[]
    token_numb=[]
    for idx, token in enumerate(tokens[1:]) :
        token_label.append(f"<b>{token}</b>")
        token_numb.append(f"{idx}")

    pair = list(zip(token_numb, token_att))

    df = pd.DataFrame(pair, columns=["Amino acid", "Attention rate"])
    df.to_csv(f"amino_acid_seq_attention_{numb}.csv", index=None)

    top3_idx = sorted(range(len(token_att)), key=lambda i: token_att[i], reverse=True)[:3]

    colors = ['cornflowerblue', ] * len(token_numb)

    for i in top3_idx:
       colors[i] = 'crimson'

    fig = go.Figure(data=[go.Bar(
        x=df["Amino acid"],
        y=df["Attention rate"],
       #  range_y=[min(token_att), max(token_att)],
        marker_color=colors  # marker color can be a single color value or an iterable
    )])

#     fig = px.histogram(df, x="Amino acid", y="Attention rate", range_y=[min(token_att), max(token_att)])

    fig.update_layout(plot_bgcolor="white")
    fig.update_xaxes(linecolor='rgba(0,0,0,0.25)', gridcolor='rgba(0,0,0,0)',mirror=False)
    fig.update_yaxes(linecolor='rgba(0,0,0,0.25)', gridcolor='rgba(0,0,0,0.07)',mirror=False)
    fig.update_layout(title={'text': "<b>Attention rate of amino acid sequence token</b>",
                             'font':{'size':40},
                             'y': 0.96,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},
                     
                      xaxis=dict(tickmode='array',
                                 tickvals=token_numb,
                                 ticktext=token_label
                                 ),

                      xaxis_title={'text': "Amino acid sequence",
                             'font':{'size':30}},
                      yaxis_title={'text': "Attention rate",
                             'font':{'size':30}},

                      font=dict(family="Calibri, monospace",
                                size=17
                                ))

    fig.write_image(f'figures/Amino_acid_seq_{numb}.png', width=1.5*1200, height=0.75*1200, scale=2)
    fig.show()


def read_attention():
    df = pd.read_csv("../amino_acid_seq_attention.csv")
        # d_flow_values = np.asarray(d_read_flow_values)

    fig = px.bar(df, x="Amino acid", y="Attention rate", range_y=[min(df["Attention rate"]), max(df["Attention rate"])])

    fig.update_layout(plot_bgcolor="white")
    fig.update_xaxes(linecolor='rgba(0,0,0,0.25)', gridcolor='rgba(0,0,0,0)',mirror=False)
    fig.update_yaxes(linecolor='rgba(0,0,0,0.25)', gridcolor='rgba(0,0,0,0.07)',mirror=False)
    fig.update_layout(title={'text': "<b>Attention rate of amino acid sequence token</b>",
                             'font':{'size':40},
                             'y': 0.96,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},

                      xaxis_title={'text': "Amino acid sequence",
                             'font':{'size':30}},
                      yaxis_title={'text': "Attention rate",
                             'font':{'size':30}},

                      font=dict(family="Calibri, monospace",
                                size=17
                                ))

    fig.write_image('figures/Amino_acid_seq.png', width=1.5*1200, height=0.75*1200, scale=2)
    fig.show()

if __name__ == '__main__':
    read_attention()