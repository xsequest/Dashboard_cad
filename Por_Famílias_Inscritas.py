import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sidrapy
from scipy import stats
from plotly.subplots import make_subplots

         
st.set_page_config(page_title="Dashboard CadÚnico", page_icon=":bar_chart:", layout="wide", initial_sidebar_state="collapsed")
st.title('Dashboard - Famílias Inscritas no CadÚnico')


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
def format_number_br(number):
    number_str = f"{number:,.0f}".replace(",", ".")
    return number_str
def format_numbers(valor):
    return '{:,.2f}'.format(valor).replace(',', 'X').replace('.', ',').replace('X', '.')

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def carrengando_informações():
    url = f'https://aplicacoes.mds.gov.br/sagi/servicos/misocial/?fl=codigo_ibge%2Canomes_s%20qtd_fam_ext_pob:cadun_qtde_fam_sit_extrema_pobreza_s%20qtd_fam_pob:cadun_qtde_fam_sit_pobreza_s%20qtd_fam_baixa_renda:cadun_qtd_familias_cadastradas_baixa_renda_i%20qtd_fam_acima_meio_sm:cadun_qtd_familias_cadastradas_rfpc_acima_meio_sm_i&fq=cadun_qtd_familias_cadastradas_baixa_renda_i%3A*&q=*%3A*&rows=600000&sort=anomes_s%20desc%2C%20codigo_ibge%20asc&wt=csv&fq=anomes_s:[201701%20TO%20202412]'

    df = pd.read_csv(url)
    df['Cod_Estado_Ibge'] = df['codigo_ibge'].astype(str).str[:2].astype(int)
    df = df.groupby(['Cod_Estado_Ibge', 'anomes_s']).sum()
    df = df.reset_index()
    df['Data'] = pd.to_datetime(df['anomes_s'].astype(str), format='%Y%m')
    df['Ano'] = df['Data'].dt.year
    df['Mês'] = df['Data'].dt.month
    df = df.drop(['anomes_s'], axis=1)
    df = df.iloc[:, [0, 6, 2, 3, 4, 5, 7, 8]]

    df_Estado = pd.read_json('https://servicodados.ibge.gov.br/api/v1/localidades/estados/')
    df_Estado = df_Estado.iloc[:, [0, 1, 2]]
    df_Estado.columns = ['Cod_Estado_Ibge', 'Sigla', 'Estado']

    df_combinado = pd.merge(
        df,
        df_Estado[['Cod_Estado_Ibge', 'Sigla', 'Estado']],
        on='Cod_Estado_Ibge',
        how='left'
    )
    df_combinado = df_combinado.iloc[:, [0, 9, 8, 1, 2, 3, 4, 5, 6, 7]]
    df_combinado.columns = ['Cod_Estado_Ibge', 'Estado', 'Sigla', 'Data', 'fam_ext_pob', 'fam_pob', 'fam_baixa_renda', 'fam_acima_meio_sm', 'Ano', 'Mês']
    df_combinado = df_combinado.sort_values(['Estado', 'Data'])
        
    return df_combinado
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def carrengando_informações_desemprego(estado=True):
    def expand_quarter(quarter):
        year = int(quarter[:4])
        q = int(quarter[4:])
        start_month = (q - 1) * 3 + 1
        return [f"{year}{m:02d}" for m in range(start_month, start_month + 3)]
   
    data = sidrapy.get_table(table_code='4099',
                            territorial_level="3" if estado else '1',
                            ibge_territorial_code="all",
                            variable='4099',
                            period='all') 

    data.columns = data.iloc[0]
    if estado:
        data = data.iloc[1:, [6, 7, 4]]
    else:
        data = data.iloc[1:, [5, 7, 4]]

    new_rows = []
    for _, row in data.iterrows():
        for month in expand_quarter(row['Trimestre (Código)']):
            if not estado:
                new_rows.append({'Unidade': 'Brasil', 'Mensal': month, 'Valor': row['Valor']})
            else:
                new_rows.append({'Unidade':row.iloc[0], 'Mensal': month, 'Valor': row['Valor']})

    new_df = pd.DataFrame(new_rows)

    new_df = new_df.sort_values('Mensal').reset_index(drop=True).sort_values(by=['Unidade', 'Mensal'])
    new_df['Data'] = pd.to_datetime(new_df['Mensal'], format='%Y%m')
    new_df['Ano'] = new_df['Data'].dt.year
    new_df['Mês'] = new_df['Data'].dt.month
    new_df['Valor'] = new_df['Valor'].astype(float)

    return new_df
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def carrengando_informações_fam_unipessoais():
    url = 'https://aplicacoes.mds.gov.br/sagi/servicos/misocial/?fl=codigo_ibge%2Canomes_s%20cadun_qtd_familias_cadastradas_i&fq=cadun_qtd_familias_cadastradas_i%3A*&q=*%3A*&rows=300000&sort=anomes_s%20desc%2C%20codigo_ibge%20asc&wt=csv&fq=anomes_s:[202301%20TO%20202412]'

    df = pd.read_csv(url)
    df['Cod_Estado_Ibge'] = df['codigo_ibge'].astype(str).str[:2].astype(int)
    df = df.groupby(['Cod_Estado_Ibge', 'anomes_s']).sum()
    df = df.reset_index()
    df['Data'] = pd.to_datetime(df['anomes_s'].astype(str), format='%Y%m')
    df['Ano'] = df['Data'].dt.year
    df['Mês'] = df['Data'].dt.month
    df = df.drop(['anomes_s', 'codigo_ibge'], axis=1)

    df_Estado = pd.read_json('https://servicodados.ibge.gov.br/api/v1/localidades/estados/')
    df_Estado = df_Estado.iloc[:, [0, 1, 2]]
    df_Estado.columns = ['Cod_Estado_Ibge', 'Sigla', 'Estado']

    df_fam = pd.merge(
        df,
        df_Estado[['Cod_Estado_Ibge', 'Sigla', 'Estado']],
        on='Cod_Estado_Ibge',
        how='left'
    )
    df_fam = df_fam.iloc[:, [0, 6, 5, 2, 1, 3, 4]]
    df_fam.columns = ['Cod_Estado_Ibge', 'Estado', 'Sigla', 'Data', 'fam_inscritas', 'Ano', 'Mês']
    df_fam = df_fam.sort_values(['Estado', 'Data'])

    # VisData -> Famílias unipessoais beneficiárias e não beneficiárias do Programa Bolsa Família inscritas no Cadastro Único
    url_base = 'https://aplicacoes.cidadania.gov.br/vis/data3/v.php?q[]=oNOclsLerpibuKep3bV%2BgW9g05Kv2rmg2a19ZW51ZmymaX6JaV2JlWCbbWCNrMmlsKyamembs61ojMfGpt2slLysiJqdtKiftJ%2BuuqqSkpyZy6mmwraIp7G1WKvtnbKtp4%2B9wGPJrZjQ7ryVm3lwoNqlwLNyk7jNps94bsPcuaehg3Ct7qZwyViQxsKfz7CWwqONpbCsmpnpm7OtZ4zHxqbdrJS8rHlkZWhgWtyorrqcoLrGU5J9pNHfspOsqpuZqpi9s6qgxsKSm2ZU2razlai7mnXfmrnBnGiSx5TWsJiYtsCpqcSGr9qnwbebjrvGU86iU8PcuvfptJ6b7FmPk4Vynap2swC0r8SOh1yspFrJq7y1qY7EwlOsrJ%2FQ3G16nbX45%2BWirm6dnMnOlM6epn3rvKZceVWj562ytamOxdWYimWoy%2BS9ma%2B7pJvirHZxiKK4z6fToZTB4G2YoWibm%2Bb8%2BrqgjsqBgS3Agn3dsqKhrp6d4vzuwKCOyoGX2V2Dz%2Bq0pp21llq7qLnBmE2dwqAt6p%2FG3G2aq7qim92awG6nnMmBZIqmodHgtKadtqmfmWHCvKCdvNSm2Z6c0KRwhbGpo67ina6ynE3L0KfLqVPB4G2anbX45%2BWirsFXk8bToMuhlNCbvaOuaGZa4qfBs56fuM%2Bnz11b0um2pKG7qKnaosB3s52S3a%2FmeA%3D%3D'
    url_data = '&ma=mes&ma=mes&dt1=2022-12-01&dt2=2024-10-01'
    url_uni = f'{url_base}{url_data}&ag=e&wt=json&tp_funcao_consulta=0&draw=2&columns[0][data]=0&columns[0][name]=codigo&columns[0][searchable]=true&columns[0][orderable]=true&columns[0][search][value]=&columns[0][search][regex]=false&columns[1][data]=1&columns[1][name]=nome&columns[1][searchable]=true&columns[1][orderable]=false&columns[1][search][value]=&columns[1][search][regex]=false&columns[2][data]=2&columns[2][name]=mes_ano_formatado&columns[2][searchable]=true&columns[2][orderable]=true&columns[2][search][value]=&columns[2][search][regex]=false&columns[3][data]=3&columns[3][name]=qtde_pbf_1_pessoa_1&columns[3][searchable]=true&columns[3][orderable]=false&columns[3][search][value]=&columns[3][search][regex]=false&columns[4][data]=4&columns[4][name]=qtde_pbf_0_pessoa_1&columns[4][searchable]=true&columns[4][orderable]=false&columns[4][search][value]=&columns[4][search][regex]=false&columns[5][data]=5&columns[5][name]=(coalesce(t.qtde_pbf_0_pessoa_1%2C0)%20%2B%20coalesce%20(t.qtde_pbf_1_pes&columns[5][searchable]=true&columns[5][orderable]=false&columns[5][search][value]=&columns[5][search][regex]=false&order[0][column]=2&order[0][dir]=asc&order[1][column]=0&order[1][dir]=asc&start=0&length=3147483647&search[value]=&search[regex]=false&export=1&export_data_comma=1&export_tipo=csv&'
    df_uni = pd.read_csv(url_uni, encoding='iso-8859-1')
    df_uni = df_uni.iloc[:, [0, 1, 2, 5]]
    df_uni['Data'] = pd.to_datetime(df_uni['Referência'] + '/01', format='%m/%Y/%d')
    df_uni['Ano'] = df_uni['Data'].dt.year
    df_uni['Mês'] = df_uni['Data'].dt.month
    df_uni = df_uni.drop(['Referência'], axis=1)
    df_uni = df_uni.iloc[:, [0, 1, 3, 2, 4, 5]]
    df_uni.columns = ['Cod_Estado_Ibge', 'Estado', 'Data', 'fam_unipessoais_inscritas', 'Ano', 'Mês']
    df_uni = df_uni.sort_values(['Estado', 'Data'])
    
    df_combinado = pd.merge(
        df_fam,
        df_uni[['Cod_Estado_Ibge', 'fam_unipessoais_inscritas', 'Data']],
        on=['Cod_Estado_Ibge', 'Data'],
        how='left'
    )
    df_combinado = df_combinado.iloc[:, [0, 1, 2, 3, 4, 7, 5, 6]]
        
    return df_combinado

@st.cache_data
def carregando_informações_media_familias_estado():
    df_populacao = sidrapy.get_table(table_code="4714", variable=93, territorial_level="3", ibge_territorial_code="all")
    df_populacao.columns = df_populacao.iloc[0]
    df_populacao = df_populacao.iloc[1:, [6, 4]]
    df_populacao.columns = ['Estado', 'População']
    df_populacao['População'] = df_populacao['População'].astype(int)

    med_fam = sidrapy.get_table(table_code="9877", territorial_level="3", ibge_territorial_code="all")
    med_fam.columns = med_fam.iloc[0]
    med_fam = med_fam.iloc[1:, [6, 4]]
    med_fam.columns = ['Estado', 'n_med_fam']
    med_fam['n_med_fam'] = med_fam['n_med_fam'].astype(float)

    df_merged = pd.merge(df_populacao, med_fam, on='Estado', how='outer')
    df_merged['Med_Fam'] = round(df_merged['População'] / df_merged['n_med_fam'], 2)
    df_merged = df_merged.sort_values(by='Estado')
    
    return df_merged

#============================================================================================================================
#     Gráficos (Funções modulares)
#============================================================================================================================
def Fig_line_chart(df, selected_faixa_, faixa_, show_media=False):
    media_estados = df_cadúnico.groupby('Data')[selected_faixa_].mean().reset_index()
    meses_anuais = df[df['Data'].dt.month == 1]['Data']
    ultimo_valor = df[selected_faixa_].iloc[-1]
    ultima_data = df['Data'].iloc[-1]

    fig_linha = px.line(df, x='Data', y=selected_faixa_, markers=False, labels={selected_faixa_: faixa_, 'Data': 'Data'})
    
    if show_media:
        fig_linha.add_scatter(
            x=media_estados['Data'],
            y=media_estados[selected_faixa_],
            mode='lines',
            name='Média Estados',
            line=dict(color='red'),
            showlegend=True
        )
    
    fig_linha.add_scatter(
        x=meses_anuais,
        y=df[df['Data'].isin(meses_anuais)][selected_faixa_],
        mode='markers',
        marker=dict(size=8, color="white", line=dict(width=2, color="green")),
        showlegend=False
    )

    for data in meses_anuais:
        valor = df[df['Data'] == data][selected_faixa_].values[0]
        fig_linha.add_annotation(
            x=data,
            y=valor,
            text=format_number_br(valor),
            showarrow=False,
            yshift=10,
            font=dict(size=12)
        )

    fig_linha.add_annotation(
        x=ultima_data,
        y=ultimo_valor,
        text=format_number_br(ultimo_valor),
        showarrow=False,
        yshift=10,
        font=dict(size=12)
    )

    fig_linha.update_traces(line_color='green', selector=dict(mode='lines', showlegend=False))
    fig_linha.update_layout(
        xaxis_title=None,
        yaxis_title='Número de famílias',
        xaxis=dict(
            tickmode='array', 
            tickvals=meses_anuais,
            tickformat='%Y-%m',  # Formato YYYYmm
            dtick='M1'  # Intervalo mensal
        ),
        yaxis=dict(gridcolor='lightgrey'),
        title={
            'text': f'Famílias inscritas no CadÚnico em {faixa_} (2017-2024)<br>{selected_estado}',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    return fig_linha
#---------------------------------------------------------------------------------------------------------------------------------
def Fig_multi_line_chart(df, selected_faixa_, faixa_, selected_estado_):
    meses_anuais = df[df['Data'].dt.month == 1]['Data']
    ultimo_valor = df[selected_faixa_].iloc[-1]
    ultima_data = df['Data'].iloc[-1]
    
    faixas = {
        'fam_ext_pob': {'color': '#a55462', 'nome': 'Extrema Pobreza'},
        'fam_pob': {'color': '#a55462', 'nome': 'Pobreza'},
        'fam_baixa_renda': {'color': '#a55462', 'nome': 'Baixa Renda'},
        'fam_acima_meio_sm': {'color': '#a55462', 'nome': 'Acima de meio SM'} #color='#a55462' Vinho 
    }
    
    fig_linha = go.Figure()
    
    for faixa, info in faixas.items():
        if faixa == selected_faixa_:
            linha_estilo = dict(color='green', dash='solid', width=2)
        else:
            linha_estilo = dict(dash='dash', width=1)
            
        fig_linha.add_trace(
            go.Scatter(
                x=df['Data'],
                y=df[faixa],
                name=info['nome'],
                line=linha_estilo,
                mode='lines'
            )
        )
        
        if faixa == selected_faixa_:
            fig_linha.add_trace(
                go.Scatter(
                    x=meses_anuais,
                    y=df[df['Data'].isin(meses_anuais)][faixa],
                    mode='markers',
                    marker=dict(size=8, color="white", line=dict(width=2, color="green")),
                    showlegend=False
                )
            )
            
            for data in meses_anuais:
                valor = df[df['Data'] == data][faixa].values[0]
                fig_linha.add_annotation(
                    x=data,
                    y=valor,
                    text=format_number_br(valor),
                    showarrow=False,
                    yshift=10,
                    font=dict(size=12)
                )
            
            fig_linha.add_annotation(
                x=ultima_data,
                y=ultimo_valor,
                text=format_number_br(ultimo_valor),
                showarrow=False,
                yshift=10,
                font=dict(size=12)
            )
    
    fig_linha.update_layout(
        xaxis_title=None,
        yaxis_title='Número de famílias inscritas',
        xaxis=dict(
            tickmode='array', 
            tickvals=meses_anuais,
            tickformat='%Y-%m',
            dtick='M1'
        ),
        yaxis=dict(gridcolor='lightgrey'),
        title={
            'text': f'Famílias inscritas no CadÚnico em {faixa_} (2017-2024)<br>{selected_estado_}',
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )    
    
    return fig_linha

#---------------------------------------------------------------------------------------------------------------------------------
def Fig_media_rank(df, selected_estado_, selected_faixa_, faixa_):
    media_estados = df.groupby('Data')[selected_faixa_].mean().reset_index()
    
    ultimos_valores = df.groupby('Sigla').agg({
        selected_faixa_: 'last',
        'Estado': 'first'
    }).reset_index()
    
    ultima_media = media_estados[selected_faixa_].iloc[-1]
    
    ultimos_valores['Diferenca'] = ((ultimos_valores[selected_faixa_] - ultima_media) / ultima_media) * 100
    
    resultados = ultimos_valores.sort_values('Diferenca', ascending=True)
    
    selected_sigla = df.loc[df['Estado'] == selected_estado_, 'Sigla'].unique()[0]
    
    posicao = resultados['Sigla'].tolist().index(selected_sigla) + 1
    
    titulo = f"Diferença percentual entre o último mês de referência e a média dos estados<br><b>{faixa_}</b>"
    
    fig = px.bar(resultados,
                 x='Sigla',
                 y='Diferenca',
                 opacity=0.7,
                 labels={'Sigla': 'Estado',
                        'Diferenca': 'Diferença da média (%)'})
    
    colors = ['green' if x == selected_sigla else '#87CEEB' for x in resultados['Sigla']]
    fig.update_traces(marker_color=colors)
    
    fig.update_layout(
        xaxis_tickangle=0,
        legend_title_text='Legenda',
        title={
            'text': titulo,
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    y_value = resultados[resultados['Sigla'] == selected_sigla]['Diferenca'].values[0]
    
    if y_value >= 0:
        ay = -40
        yanchor = 'bottom'
    else:
        ay = 40
        yanchor = 'top'
    
    fig.add_annotation(
        x=selected_sigla,
        y=y_value,
        text=f"{posicao}º",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=0,
        ay=ay,
        yanchor=yanchor,
        font=dict(size=12, color="white"),
        bgcolor="green",
        opacity=0.8,
        bordercolor="green",
        borderwidth=2,
        borderpad=4,
        align="center"
    )
    
    return fig

#---------------------------------------------------------------------------------------------------------------------------------
def Fig_candle_chart(df_, selected_faixa_, faixa_, selected_estado_):
    medias_anuais_plotly = df_.groupby('Ano')[selected_faixa_].mean()

    df_yearly = df_.groupby('Ano').agg({
        selected_faixa_: ['first', 'max', 'min', 'last']
    }).reset_index()
    df_yearly.columns = ['Ano', 'Open', 'High', 'Low', 'Close']

    df_yearly['Date'] = pd.to_datetime(df_yearly['Ano'].astype(str))

    fig_candle = go.Figure()
    fig_candle.add_trace(go.Candlestick(
        x=df_yearly['Date'],
        open=df_yearly['Open'],
        high=df_yearly['High'],
        low=df_yearly['Low'],
        close=df_yearly['Close'],
        name='Candlestick'
    ))
    fig_candle.add_trace(go.Scatter(
        x=df_yearly['Date'],
        y=medias_anuais_plotly,
        mode='lines+markers',
        name='Média Anual',
        line=dict(color='grey', width=1, dash='dash'),
        marker=dict(symbol='circle', size=3, color='white', line=dict(width=1, color='white'))
    ))
    fig_candle.update_layout(
        title={
            'text': f'Famílias inscritas no CadÚnico em <b>{faixa_}</b> por Ano<br><sup><b>{selected_estado_}</sup>',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'family': 'Calibri', 'size': 24}
        },
        yaxis_title='<b>Número de Famílias',
        xaxis_title='<b>Ano',
        xaxis_rangeslider_visible=False,
        width=1200,
        height=600,
        font=dict(family='Calibri', size=14)
    )
    fig_candle.update_xaxes(
        tickformat='%Y',
        tickmode='array',
        tickvals=df_yearly['Date']
    )
    fig_candle.update_xaxes(title_font=dict(family='Calibri', size=16))
    fig_candle.update_yaxes(title_font=dict(family='Calibri', size=16))
    return fig_candle
#------------------------------------------------------------------------------------------------------------------------------------------------------------
def Fig_trend_chart(df_, selected_faixa_, faixa_,  selected_estado_):
    df_tendencia = df_.set_index('Data')[selected_faixa_]
    df_tendencia = df_tendencia.reset_index()
    df_tendencia['tempo'] = range(len(df_tendencia))
    slope, intercept, _, _, _ = stats.linregress(df_tendencia['tempo'], df_tendencia[selected_faixa_])
    linha_tendencia = slope * df_tendencia['tempo'] + intercept

    fig_tendencia = go.Figure()
    fig_tendencia.add_trace(go.Scatter(
        x=df_tendencia['Data'],
        y=df_tendencia[selected_faixa_],
        mode='markers',
        name='Dados Originais',
        marker=dict(size=8, opacity=0.5)
    ))
    fig_tendencia.add_trace(go.Scatter(
        x=df_tendencia['Data'],
        y=linha_tendencia,
        mode='lines',
        name='Linha de Tendência',
        line=dict(color='red', width=2)
    ))
    fig_tendencia.update_layout(
        title={
            'text': f'Tendência: Número de Famílias inscritas no CadÚnico em {faixa_} - {selected_estado_}',
            'x': 0.5,
            'y': 0.9,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    fig_tendencia.update_xaxes(tickangle=0)
    return fig_tendencia
#---------------------------------------------------------------------------------------------------------------------------------------
def calcular_tendencia(grupo, tendencia=True):
    serie = grupo.reset_index(drop=True)
    serie['tempo'] = range(len(serie))
    
    slope, intercept, _, _, _ = stats.linregress(serie['tempo'], serie[selected_faixa])
    
    linha_tendencia = slope * serie['tempo'] + intercept
    
    if tendencia:
        diff_tendencia = linha_tendencia.iloc[-1] / linha_tendencia.iloc[0] - 1
    else:
        diff_tendencia = (serie[selected_faixa].iloc[-1] - linha_tendencia.iloc[-1]) / linha_tendencia.iloc[-1]
    
    return pd.Series({'Tendência': round(diff_tendencia*100, 2)})
#---------------------------------------------------------------------------------------------------------------------------------------

def Fig_trend_rank(df, selected_estado_, faixa_, metodo=True):
    resultados = df.groupby('Sigla').apply(calcular_tendencia, metodo, include_groups=False).reset_index()
    resultados = resultados.sort_values('Tendência', ascending=True)

    selected_sigla = df.loc[df['Estado'] == selected_estado_, 'Sigla'].unique()[0] 
    posicao = resultados['Sigla'].tolist().index(selected_sigla) + 1

    titulo = (f"Inclinação da linha de tendência <b>{faixa_} por estado</b><br>2017/2024" 
            if metodo else 
            f"Diferença percentual entre o último valor real e a linha de tendência <b>{faixa_} por estado</b><br>2017/2024")

    fig = px.bar(resultados, 
                x='Sigla', 
                y='Tendência',
                opacity=0.7,
                labels={'Sigla': 'Estado', 
                        'Tendência': 'Variação da tendência (%) ' if metodo else 'Diferença real vs tendência (%) '})

    colors = ['green' if x == selected_sigla else '#87CEEB' for x in resultados['Sigla']]
    fig.update_traces(marker_color=colors)

    fig.update_layout(
        xaxis_tickangle=0,
        legend_title_text='Legenda',
        title={
            'text': titulo,
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    y_value = resultados[resultados['Sigla'] == selected_sigla]['Tendência'].values[0]

    if y_value >= 0:
        ay = -40  
        yanchor = 'bottom'
    else:
        ay = 40
        yanchor = 'top'

    fig.add_annotation(
        x=selected_sigla,
        y=y_value,
        text=f"{posicao}º",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=0,
        ay=ay,
        yanchor=yanchor,
        font=dict(size=12, color="white"),
        bgcolor="green",
        opacity=0.8,
        bordercolor="green",
        borderwidth=2,
        borderpad=4,
        align="center"
    )

    return fig
#-------------------------------------------------------------------------------------------------------------------------------------
def create_line_trace(x, y, name, color, y_formatted):
    return go.Scatter(
        x=x,
        y=y,
        name=name,
        line=dict(color=color),
        mode='lines+markers',
        text=y_formatted,
        hovertemplate='%{text}<extra></extra>',
        marker=dict(
            size=6,
            color='white',
            line=dict(color=color, width=2)
        )
    )

def create_graph_layout(title, y1_title=None, y2_title=None, y1_color='#84c784', y2_color='#00766f'):
    layout = {
        'title': {
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        'showlegend': True,
        'plot_bgcolor': 'white',
        'yaxis': dict(
            showgrid=True,
            gridcolor='lightgray',
            title=dict(text=y1_title, font=dict(color=y1_color)),
            tickfont=dict(color=y1_color)
        )
    }
    
    if y2_title:
        layout['yaxis2'] = dict(
            showgrid=False,
            title=dict(text=y2_title, font=dict(color=y2_color)),
            tickfont=dict(color=y2_color)
        )
        layout['legend'] = dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    
    return layout
def plot_single_graph(df, selected_year, selected_faixa, region_name, color):
    df_selected = df.loc[df['Data'].dt.year == selected_year].copy()
    meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    meses_x = meses[:df_selected['Mês'].max()]
    
    y_formatted = [format_numbers(val) for val in df_selected[dict_faixa[selected_faixa]]]
    
    fig = go.Figure()
    fig.add_trace(create_line_trace(
        meses_x,
        df_selected[dict_faixa[selected_faixa]],
        selected_faixa,
        color,
        y_formatted
    ))
    
    fig.update_layout(create_graph_layout(
        f"Famílias em <b>{selected_faixa}</b><br>- <b>{region_name}</b> - {selected_year}",
        selected_faixa,
        y1_color=color
    ))
    
    return fig
#-----------------------------------------------------------------------------------------------------------
def plot_multi_graph(df, selected_year, selected_faixas, region_name):
    """Plota gráfico com múltiplas linhas"""
    df_selected = df.loc[df['Data'].dt.year == selected_year].copy()
    meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    meses_x = meses[:df_selected['Mês'].max()]
    
    if region_name == 'Brasil':
        colors = ['#506e9a', '#41b8d5']
    else:
        colors = ['#84c784', '#00766f']
        
    y_formatted = []
    for faixa in selected_faixas[:2]:
        y_formatted.append([format_numbers(val) for val in df_selected[dict_faixa[faixa]]])
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        create_line_trace(
            meses_x,
            df_selected[dict_faixa[selected_faixas[0]]],
            selected_faixas[0],
            colors[0],
            y_formatted[0]
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        create_line_trace(
            meses_x,
            df_selected[dict_faixa[selected_faixas[1]]],
            selected_faixas[1],
            colors[1],
            y_formatted[1]
        ),
        secondary_y=True
    )
    
    fig.update_layout(create_graph_layout(
        f"Famílias em <b>{selected_faixas[0]}</b> e <b>{selected_faixas[1]}</b><br>- <b>{region_name}</b> - {selected_year}",
        selected_faixas[0],
        selected_faixas[1],
        colors[0],
        colors[1]
    ))
    
    return fig

#==============================================================================================================================================================
#==============================================================================================================================================================
list_faixa = ["Extrema Pobreza", "Pobreza", "Baixa Renda", "Até meio salário-mínimo"]
dict_faixa = {'Extrema Pobreza': 'fam_ext_pob', 'Pobreza': 'fam_pob', 'Baixa Renda': 'fam_baixa_renda', "Até meio salário-mínimo": 'fam_acima_meio_sm'}

df_cadúnico = carrengando_informações()

with st.sidebar.expander("Configurações de seleção:", expanded=True):
    with st.container(border=True):
        faixa = st.selectbox("Faixa de renda:", (list_faixa), index=0)            
        selected_faixa = dict_faixa[faixa]
    with st.container(border=True):
        selected_estado = st.selectbox("Estado:", (df_cadúnico['Estado'].unique()), index=8)

### DATAFRAMES
#########################################################################################################################
df_cadúnico_br = df_cadúnico.groupby('Data').sum().copy()
df_cadúnico_br = df_cadúnico_br.reset_index().iloc[:, [0, 4, 5, 6, 7]]
df_cadúnico_br['Mês'] = df_cadúnico_br['Data'].dt.month
df_cadúnico_br = df_cadúnico_br.iloc[:, [0, 5, 1, 2, 3, 4]]

df_cadúnico_estado = df_cadúnico[df_cadúnico['Estado'] == selected_estado]
#####
df_desemprego = carrengando_informações_desemprego()
df_desemprego_br = carrengando_informações_desemprego(False)

df_combined_desemprego = pd.concat([df_desemprego, df_desemprego_br], ignore_index=True)
df_combined_desemprego = df_combined_desemprego.sort_values(['Unidade','Mensal']).reset_index(drop=True)
df_combined_desemprego = df_combined_desemprego.loc[df_combined_desemprego['Ano'] >= 2017]
#####
df_fam_uni = carrengando_informações_fam_unipessoais()
##########################################################################################################################


tab_panorama, tab_estado = st.tabs(["Panorama:", "Estado:"])


### PANORAMA
#=======================================================================================================================================================================
with tab_panorama:

    # Funções para criar o gráfico
    #----------------------------------------------------------------------------------------------------
    def panorama_layout():
        col_year, col_type, col_selection = st.columns(3)
        
        with col_year:
            unique_years = sorted(df_cadúnico_estado['Ano'].dropna().unique(), reverse=True)
            selected_year = st.selectbox(
                'Selecione o Ano:', 
                unique_years, 
                index=0, 
                key='panorama_year_select'
            )
                
        with col_type:
            tipo_de_gráfico = st.segmented_control(
                label='Tipo de Gráfico',
                options= ['Padrão', 'Proporção'],
                default='Padrão',
                selection_mode="single",
                key='panorama_type_radio'
            )
                
        with col_selection:
            if tipo_de_gráfico == "Padrão":
                view_options = {
                    'Média do ano': 'mean',
                    'Último mês': 'last'
                }
                selected_view = st.segmented_control(
                    label='Opções da linha:', 
                    options= list(view_options.keys()),
                    default='Média do ano',
                    key='panorama_view_select'
                )
                selected_aggregation = view_options[selected_view]
            else:
                selected_view = "Proporção"
                selected_aggregation = "Proporção"
            
        return selected_year, selected_view, selected_aggregation, tipo_de_gráfico
    #----------------------------------------------------------------------------------------------------
    def processar_dados(selected_year, selected_aggregation, selected_faixa, tipo_de_gráfico):
        df_media_fam = carregando_informações_media_familias_estado()
        
        agg_function = 'mean' if tipo_de_gráfico == 'Proporção' else selected_aggregation
        
        df_year = (df_cadúnico[df_cadúnico['Ano'] == selected_year]
                .groupby(['Sigla', 'Estado'])[selected_faixa]
                .agg(agg_function)
                .reset_index())
        
        df_combined = pd.merge(df_year, df_media_fam, on='Estado', how='outer')
        df_combined['diff'] = (df_combined[selected_faixa] / df_combined['Med_Fam'])*100
        
        if tipo_de_gráfico == 'Proporção':
            return df_combined.sort_values('diff', ascending=True)
        return df_combined.sort_values(selected_faixa, ascending=False)
    #----------------------------------------------------------------------------------------------------
    def criar_chart_title(selected_view, faixa, selected_year):
        if selected_view == 'Média do ano':
            return f"Média de famílias inscritas no CadÚnico em <b>{faixa} por estado</b><br>{selected_year}"
        return f"Famílias inscritas no CadÚnico em <b>{faixa} por estado</b><br>Último mês de referência - {selected_year}"

    def Fig_panorama_chart(df_combined, selected_estado, selected_faixa, faixa, panorama_titulo, tipo_de_gráfico):
        selected_state_sigla = df_cadúnico.loc[df_cadúnico['Estado'] == selected_estado, 'Sigla'].unique()[0]
        colors = ['green' if x == selected_state_sigla else '#87CEEB' for x in df_combined['Sigla']]

        if tipo_de_gráfico == 'Proporção':
            fig = px.bar(
                df_combined,
                x='Sigla',
                y='diff',
                opacity=0.7,
                labels={
                    'Sigla': 'Estado',
                    'diff': 'Proporção (%)'
                }
            )
            
            fig.update_traces(marker_color=colors)
            
            fig.update_layout(
                xaxis_tickangle=0,
                legend_title_text='Legenda',
                title={
                    'text': f'Diferença percentual entre o número de famílias em {faixa} e o número médio de famílias em cada estado.',
                    'x': 0.5,
                    'y': 0.95,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
            )

        else:
            fig = px.bar(
                df_combined,
                x='Sigla',
                y='Med_Fam',
                opacity=0.7,
                labels={
                    'Sigla': 'Estado',
                    'Med_Fam': 'Número médio de famílias no Estado.'
                }
            )
            
            fig.update_traces(marker_color=colors)
            
            fig.add_trace(
                go.Scatter(
                    x=df_combined['Sigla'],
                    y=df_combined[f'{selected_faixa}'],
                    mode='lines+markers',
                    name=f'{faixa}',
                    line=dict(color='#FF4444', width=1, dash='dash'),
                    marker=dict(size=4)
                )
            )
            
            fig.update_layout(
                xaxis_tickangle=0,
                legend_title_text='Legenda',
                title={
                    'text': panorama_titulo,
                    'x': 0.5,
                    'y': 0.95,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
            )
        
        return fig

    # Plotagem de Gráfico (tab Panorama - média de famílias)
    #===========================================================================================================================
    with st.container(border=True):
        with st.container(border=True):
            with st.container(border=True):
                selected_panorama_year, panorama_selection, selected_aggregation, tipo_de_gráfico = panorama_layout()
                df_combined = processar_dados(selected_panorama_year, selected_aggregation, selected_faixa, tipo_de_gráfico)
                panorama_titulo = criar_chart_title(panorama_selection, faixa, selected_panorama_year)
                
                fig = Fig_panorama_chart(
                    df_combined,
                    selected_estado,
                    selected_faixa,
                    faixa,
                    panorama_titulo,
                    tipo_de_gráfico
                )

            st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        #--------------------------------------------------------------------------------------------------------------------------------------------------------
        # Plotagem de Gráfico (tab Panorama - Sazonalidade)
        #--------------------------------------------------------------------------------------------------------------------------------------------------------
        
        with st.container(border=True):

            df_seasonal = df_cadúnico_br.groupby('Mês')[selected_faixa].mean().reset_index()
            df_seasonal['Mês'] = pd.Categorical(df_seasonal['Mês'], categories=range(1, 13), ordered=True)
            df_seasonal = df_seasonal.sort_values('Mês')

            meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
            df_seasonal['Mês_nome'] = [meses[i-1] for i in df_seasonal['Mês']]

            fig_sazonalidade = px.line(df_seasonal, x='Mês_nome', y=selected_faixa, markers=True, labels={selected_faixa: faixa, 'Mês_nome': 'Mês'})

            fig_sazonalidade.update_traces(line_color='#41b8d5', marker=dict(size=8, color="white", line=dict(width=2, color="#41b8d5")))   #'#506e9a', '#41b8d5'

            fig_sazonalidade.update_layout(
                xaxis_title='Mês',
                yaxis_title='Número médio de famílias',
                xaxis=dict(tickmode='array', tickvals=meses),
                yaxis=dict(gridcolor='lightgrey')
            )
            fig_sazonalidade.update_layout(
                title={
                    'text': f'Sazonalidade: Média mensal de famílias em {faixa} (2017-2024)<br>Brasil',
                    'x': 0.5,
                    'y': 0.95,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
            )

            for i, row in df_seasonal.iterrows():
                fig_sazonalidade.add_annotation(
                    x=row['Mês_nome'],
                    y=row[selected_faixa],
                    text=format_number_br(row[selected_faixa]),
                    showarrow=False,
                    yshift=10,
                    font=dict(size=9)
                )
            st.plotly_chart(fig_sazonalidade, theme="streamlit", use_container_width=True)
            #------------------------------------------------------------------------------------------------------------------------------------------------------------
            st.divider()
            #------------------------------------------------------------------------------------------------------------------------------------------------------------
            df_seasonal_estado = df_cadúnico_estado.groupby('Mês')[selected_faixa].mean().reset_index()
            df_seasonal_estado['Mês'] = pd.Categorical(df_seasonal_estado['Mês'], categories=range(1, 13), ordered=True)
            df_seasonal_estado = df_seasonal_estado.sort_values('Mês')

            meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
            df_seasonal_estado['Mês_nome'] = [meses[i-1] for i in df_seasonal_estado['Mês']]

            fig_sazonalidade_estado = px.line(df_seasonal_estado, x='Mês_nome', y=selected_faixa, markers=True, labels={selected_faixa: faixa, 'Mês_nome': 'Mês'})

            fig_sazonalidade_estado.update_traces(line_color='green', marker=dict(size=8, color="white", line=dict(width=2, color="green")))

            fig_sazonalidade_estado.update_layout(
                xaxis_title='Mês',
                yaxis_title='Número médio de famílias',
                xaxis=dict(tickmode='array', tickvals=meses),
                yaxis=dict(gridcolor='lightgrey')
            )
            fig_sazonalidade_estado.update_layout(
                title={
                    'text':f'{selected_estado}',
                    'x': 0.5,
                    'y': 0.95,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
            )

            for i, row in df_seasonal_estado.iterrows():
                fig_sazonalidade_estado.add_annotation(
                    x=row['Mês_nome'],
                    y=row[selected_faixa],
                    text=format_number_br(row[selected_faixa]),
                    showarrow=False,
                    yshift=10,
                    font=dict(size=9)
                )
            st.plotly_chart(fig_sazonalidade_estado, theme="streamlit", use_container_width=True)



### ESTADO
#======================================================================================================================================================================
with tab_estado:
    with st.container(border=True):
        with st.container(border=True):
            
            with st.expander("Opções de Plotagem:"):
                col_options, col_multi_plot = st.columns(2)
                with col_options:
                    selected_graph = st.pills("Selecione o tipo de gráfico para a visualização:", ['Linha','Candle', 'Tendência'], default='Linha')
                if selected_graph == 'Linha':
                    with col_multi_plot:
                        selected_multi_graph = st.radio("Opções do gráfico em linha:", ['Faixa de renda selecionada','Todas as faixas de renda'], index=0, key='multi_graph')
           
            with st.container(border=True):
                if selected_graph == 'Candle':                   
                    st.plotly_chart(Fig_candle_chart(df_cadúnico_estado, selected_faixa, faixa, selected_estado), theme="streamlit", use_container_width=True)
               
                elif selected_graph == 'Tendência':
                    st.plotly_chart(Fig_trend_chart(df_cadúnico_estado, selected_faixa, faixa, selected_estado), theme="streamlit", use_container_width=True) 

                    on_tendencia = not(st.toggle("Selecionar diferença pelo último valor."))  

                    fig_rank_tendencia = Fig_trend_rank(df_cadúnico, selected_estado, faixa, on_tendencia)                    
                    st.plotly_chart(fig_rank_tendencia, theme="streamlit", use_container_width=True)

                else: # Em linha
                    if selected_multi_graph == 'Todas as faixas de renda':
                        st.plotly_chart(Fig_multi_line_chart(df_cadúnico_estado, selected_faixa, faixa, selected_estado), theme="streamlit", use_container_width=True)
                    else:
                        on_media = st.toggle('Mostrar média dos estados.')
                        st.plotly_chart(Fig_line_chart(df_cadúnico_estado, selected_faixa, faixa, on_media), theme="streamlit", use_container_width=True)
                        
                        if on_media:
                            st.divider()
                            st.plotly_chart(Fig_media_rank(df_cadúnico, selected_estado ,selected_faixa, faixa), theme="streamlit", use_container_width=True)

                

        #------------------------------------------------------------------------------------------------------------------------------------
         
        with st.expander("Gráficos Secundários:"):
            col_candle, col_crescimento = st.columns(2) 
            with col_candle:
                with st.container(border=True):
                    with st.container(border=True):
                        #============================================================================================= 
                        col_year, col_faixa = st.columns(2)
                        with col_year:
                            anos_unicos = df_cadúnico_estado['Ano'].dropna().unique()
                            anos_unicos_ordenados = sorted(anos_unicos, reverse=True)
                            selected_year = st.selectbox('Selecione o Ano:', anos_unicos_ordenados, index=0, key='year_selector')
                        
                        with col_faixa:
                            def mostrar_opcoes(list_faixa):
                                opcoes = list_faixa.copy() 
                                opcoes.append('Todas')
                                return opcoes
                            def filtrar_opcoes(lista, excluir=None):
                                if excluir:
                                    return [item for item in lista if item != excluir]
                                return lista
                            
                            default_values = [faixa, filtrar_opcoes(list_faixa, faixa)[0]]
                            
                            selected_faixa_multi = st.multiselect(
                                "Selecione a faixa de renda:",
                                mostrar_opcoes(list_faixa),
                                default=default_values, max_selections=2
                            )
                    #=============================================================================================              
                    if 'Todas' in selected_faixa_multi:
                        st.text("Opção 'Todas' está em construção!")
                    else:
                        if len(selected_faixa_multi) == 1:
                            fig_estado = plot_single_graph(
                                df_cadúnico_estado,
                                selected_year,
                                selected_faixa_multi[0],
                                selected_estado,
                                '#84c784'
                            )
                            st.plotly_chart(fig_estado, theme="streamlit", use_container_width=True)
                            
                            st.divider()
                            
                            fig_brasil = plot_single_graph(
                                df_cadúnico_br,
                                selected_year,
                                selected_faixa_multi[0],
                                'Brasil',
                                '#506e9a'
                            )
                            st.plotly_chart(fig_brasil, theme="streamlit", use_container_width=True)
                            
                        elif len(selected_faixa_multi) > 1:
                            fig_estado_multi = plot_multi_graph(
                                df_cadúnico_estado,
                                selected_year,
                                selected_faixa_multi,
                                selected_estado
                            )
                            st.plotly_chart(fig_estado_multi, theme="streamlit", use_container_width=True)
                            
                            st.divider()
                            
                            fig_brasil_multi = plot_multi_graph(
                                df_cadúnico_br,
                                selected_year,
                                selected_faixa_multi,
                                'Brasil'
                            )
                            st.plotly_chart(fig_brasil_multi, theme="streamlit", use_container_width=True)

                #=========================================================================================================================================       
                with col_crescimento:
                    with st.container(border=True):
                        selected_variação = st.segmented_control('Variação percentual por:', ['Ano', 'Mês'], default= 'Ano', selection_mode="single", key='variação')
                        if selected_variação == 'Mês':
                            def get_pct_change(df):
                                columns_to_change = ['fam_ext_pob', 'fam_pob', 'fam_baixa_renda', 'fam_acima_meio_sm']

                                df_pct_change = round(df[columns_to_change].pct_change() *100, 2)
                                df_pct_change = df_pct_change.fillna(0)

                                new_column_names = {col: f'{col}_pct_change' for col in columns_to_change}
                                df_pct_change = df_pct_change.rename(columns=new_column_names)

                                df_result = pd.concat([df, df_pct_change], axis=1)
                                df_result = df_result.reset_index(drop=True)

                                df_result = df_result.iloc[:, [1, 2, 3, 10, 11, 12, 13]]

                                return df_result

                            def to_plot(df, selected_estado_, selected_faixa_, faixa_):
                                df_to_plot = get_pct_change(df)
                                fig_diff = px.area(
                                    df_to_plot, 
                                    x='Data', 
                                    y=f'{selected_faixa_}_pct_change'
                                )

                                fig_diff.update_layout(
                                    title={
                                        'text': f'Variação percentual mensal<br>{faixa_} - {selected_estado_}',
                                        'x': 0.5,
                                        'xanchor': 'center',
                                        'yanchor': 'top'
                                    },
                                    title_font_size=20,
                                    yaxis_title="Diferença Percentual (%)",
                                    plot_bgcolor='white',
                                    yaxis=dict(
                                        gridcolor='lightgrey',
                                        zeroline=True,
                                        zerolinecolor='lightgrey'
                                    ),
                                    xaxis=dict(
                                        showgrid=False,
                                        tickangle=0
                                    ),
                                    legend_title="Ano",
                                    legend=dict(
                                        yanchor="top",
                                        y=0.99,
                                        xanchor="left",
                                        x=0.01        
                                    )
                                )

                                fig_diff.update_traces(
                                    line_color='grey',
                                    fillcolor='#87CEEB',
                                    opacity=0.4,
                                    line=dict(width=1),
                                    selector=dict(type='scatter')
                                )
                                
                                return fig_diff

                            st.plotly_chart(to_plot(df_cadúnico_estado, selected_estado, selected_faixa, faixa), theme="streamlit", use_container_width=True)
                        else:
                            df_growth = df_cadúnico_estado.groupby('Ano')[selected_faixa].last()
                            growth_rate = round((df_growth.pct_change() * 100).dropna(), 2)
                            variacao_percentual = (df_growth.iloc[-1] / df_growth.iloc[0] - 1) * 100

                            colors = ['green' if x < 0 else '#FF4444' for x in growth_rate] #87CEEB #FF4444

                            fig_crescimento = px.bar(
                                x=growth_rate.index, 
                                y=growth_rate,
                                labels={'x': 'Ano', 'y': 'Taxa de crescimento (%)'}, 
                                text=growth_rate.map(lambda x: f'{x:.2f}%'),
                                color=growth_rate,
                                color_discrete_sequence=None, opacity=0.8
                            )

                            fig_crescimento.update_traces(marker_color=colors)

                            fig_crescimento.add_hline(
                                y=variacao_percentual, 
                                line_dash="dash", 
                                line_color="green",
                                line_width=1, 
                                annotation_text=f'Variação 2017-2024:<br>{variacao_percentual:.2f}%</b>', 
                                annotation_position="top left"
                            )

                            fig_crescimento.update_layout(
                                width=800, 
                                height=400,
                                xaxis_title=None,
                                yaxis_title='Taxa de crescimento (%)',
                                showlegend=False,
                                plot_bgcolor='white'
                            )
                            fig_crescimento.update_layout(
                                title={
                                    'text': f'Taxa de crescimento anual de famílias em <b>{faixa}</b><br>- <b>{selected_estado}</b> -',
                                    'x': 0.5,
                                    'y': 0.9,
                                    'xanchor': 'center',
                                    'yanchor': 'top'
                                }
                            )

                            fig_crescimento.add_hline(y=0, line_color='black')
                            fig_crescimento.update_traces(hovertemplate='Ano: <b>%{x}</b><br>Taxa de crescimento: <b>%{y:.2f}%</b><extra></extra>')
                            st.plotly_chart(fig_crescimento, theme="streamlit", use_container_width=True)
                        
                        #----------------------------------------------------------------------------------------------------------------------------------------
                        st.divider()
                        #----------------------------------------------------------------------------------------------------------------------------------------
                        
                        def df_pizza(_df, selected_v):
                            df_pizza = _df.loc[_df['Ano'] == 2024].iloc[-1]
                            
                            cols = ['fam_ext_pob', 'fam_pob', 'fam_baixa_renda', 'fam_acima_meio_sm']
                            data = [df_pizza[col] for col in cols]
                            labels = ['Extrema pob.', 'Pobreza', 'Baixa Renda', 'Acima meio S. Mín.']#[col.replace('_', ' ').title() for col in cols]

                            colors = ['#82c374', '#42ae4c', '#298e46', '#226f3b']
                            #colors = [('#FFE101' if col == selected_v else color) 
                            #            for col, color in zip(cols, default_colors)]
                            
                            pull_values = [0.1 if col == selected_v else 0 for col in cols]

                            fig = go.Figure(data=[go.Pie(
                                labels=labels,
                                values=data,
                                textinfo='label+percent',
                                pull=pull_values,
                                marker=dict(colors=colors),
                                showlegend=False
                            )])
                            
                            fig.update_layout(
                                title=dict(
                                    text=f'Famílias Inscritas no CadÚnico por Faixa de Renda - {selected_estado}<br>Último mês de referência.',
                                    font=dict(size=20),
                                    x=0.5,
                                    xanchor='center'
                                ),
                                font=dict(size=16),
                                margin=dict(t=50, b=50, l=100, r=100)
                            )
                            
                            return fig

                        st.plotly_chart(df_pizza(df_cadúnico_estado, selected_v=selected_faixa), theme="streamlit", use_container_width=True)
        
        #===============================================================================================================================================
        with st.expander("Famílias Unipessoais Inscritas e Taxa de desocupação:"):
            col_fam_uni, col_desemprego = st.columns(2)
            #===============================================================================================================================================
            with col_fam_uni:
                with st.container(border=True):
                    df_selected_uni = df_fam_uni.loc[df_fam_uni['Estado'] == selected_estado].copy()

                    df_selected_uni_ind = df_selected_uni.copy()
                    df_selected_uni_ind['Data'] = pd.to_datetime(df_selected_uni_ind['Data'])

                    df_selected_uni_ind = df_selected_uni_ind.sort_values('Data')

                    fig_fam_uni_ind = px.line(df_selected_uni_ind, 
                                x='Data', 
                                y='fam_unipessoais_inscritas')


                    fig_fam_uni_ind.update_layout(
                        title={
                            'text': f'Número de Famílias Unipessoais Inscritas no CadÚnico - Todas as faixas de renda<br>{selected_estado} - (2023-2024) *Janela Procad',
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                        title_font_size=20,
                        xaxis_title="Data",
                        yaxis_title="Número de Famílias Unipessoais Inscritas",
                        plot_bgcolor='white',
                        yaxis=dict(
                            gridcolor='lightgrey',
                            zeroline=True,
                            zerolinecolor='lightgrey',
                            tickformat=','
                        ),
                        xaxis=dict(
                            showgrid=False,
                            tickangle=0
                        ),
                        legend_title="Ano",
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01        
                        )
                    )
                    fig_fam_uni_ind.update_traces(
                        line_color='#2ca02c',
                        selector=dict(mode='lines')
                    )
                    fig_fam_uni_ind.update_yaxes(
                        tickformat=",.",
                        separatethousands=True
                    )
                    st.plotly_chart(fig_fam_uni_ind, theme="streamlit", use_container_width=True)  

                    #-------------------------------------------------------------------------------------------------------
                    st.divider()
                    #-------------------------------------------------------------------------------------------------------

                    df_selected_uni['fam_inscritas'] = round(df_selected_uni['fam_inscritas'] / 10**6, 3)
                    df_selected_uni['fam_unipessoais_inscritas'] = round(df_selected_uni['fam_unipessoais_inscritas'] / 10**6, 3)

                    fig_fam_uni = px.bar(df_selected_uni, 
                                x='Data', 
                                y='fam_inscritas',
                                opacity=0.5,
                                labels={'Data': 'Data', 
                                        'fam_inscritas': 'Número de famílias inscritas (milhões)'})

                    fig_fam_uni.update_traces(marker_color='#87CEEB')


                    fig_fam_uni.add_trace(
                        go.Scatter(
                            x=df_selected_uni['Data'],
                            y=df_selected_uni['fam_unipessoais_inscritas'],
                            mode='lines+markers',
                            name='Famílias Unipessoais',
                            line=dict(color='#FF4444', width=1),
                            marker=dict(size=4)
                        )
                    )

                    fig_fam_uni.update_layout(
                        xaxis_tickangle=0,
                        legend_title_text='Legenda'
                    )
                    fig_fam_uni.update_layout(
                        title={
                            'text': 'Famílias Inscritas no CadÚnico em Goiás: <b>Total e Unipessoais</b><br>Todas as faixas de renda',
                            'x': 0.5,
                            'y': 0.95,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                        xaxis_title=None,
                        yaxis_title="Número de Famílias (milhões)"
                    )

                    fig_fam_uni.update_yaxes(
                        tickformat=",.",
                        separatethousands=True
                    )
                    st.plotly_chart(fig_fam_uni, theme="streamlit", use_container_width=True)             
            
            #===============================================================================================================================================
            with col_desemprego:
                with st.container(border=True):
                    df_desemprego_estado = df_combined_desemprego.loc[df_combined_desemprego['Unidade'] == selected_estado].copy()
                    df_desemprego_br = df_combined_desemprego.loc[df_combined_desemprego['Unidade'] == 'Brasil'].copy()
                    df_plot_desemprego = pd.concat([df_desemprego_br, df_desemprego_estado])
                    
                    fig_desemprego = px.line(df_plot_desemprego, 
                                x='Data', 
                                y=df_plot_desemprego.columns.values[2],
                                color='Unidade',
                                labels={df_plot_desemprego.columns.values[2]: 'Desemprego'},
                                color_discrete_map={'Brasil': '#1f77b4', selected_estado: '#2ca02c'})
                    fig_desemprego.update_layout(
                        title={
                            'text': f'Taxa de Desemprego<br>Brasil - {selected_estado}',
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                        title_font_size=20,
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="right",
                            x=0.99
                        ),
                        plot_bgcolor='white',
                        yaxis=dict(
                            gridcolor='lightgrey',
                            showgrid=True,
                            zeroline=False
                        ),
                        xaxis=dict(
                            showgrid=False,
                            zeroline=False
                        )
                    )

                    st.plotly_chart(fig_desemprego, theme="streamlit", use_container_width=True)

                    #----------------------------------------------------------------------------------------
                    st.divider()
                    #----------------------------------------------------------------------------------------

                    datas_unicas = sorted(set(df_desemprego_estado['Data']) & set(df_desemprego_br['Data']))
                    df_diferenca = pd.DataFrame(index=datas_unicas)

                    df_diferenca['Diferenca_Percentual'] = round(
                        ((df_desemprego_estado.set_index('Data').loc[datas_unicas, 'Valor'] / 
                        df_desemprego_br.set_index('Data').loc[datas_unicas, 'Valor']) - 1) * 100, 2
                    )

                    df_diferenca['Ano'] = pd.to_datetime(df_diferenca.index).year
                    df_diferenca['Mês'] = pd.to_datetime(df_diferenca.index).month
                    df_diferenca = df_diferenca.reset_index().rename(columns={'index': 'Data'})
                    df_diferenca = df_diferenca.sort_values('Data')

                    fig_desemprego_diff = px.area(
                        df_diferenca, 
                        x='Data', 
                        y='Diferenca_Percentual',
                        title=f'Diferença percentual Taxa de Desemprego<br>{selected_estado} - Brasil (2017-2024)'
                    )

                    fig_desemprego_diff.update_layout(
                        title={
                            'text': f'Diferença percentual Taxa de Desemprego<br>{selected_estado} - Brasil (2017-2024)',
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                        title_font_size=20,
                        xaxis_title="Data",
                        yaxis_title="Diferença Percentual (%)",
                        plot_bgcolor='white',
                        yaxis=dict(
                            gridcolor='lightgrey',
                            zeroline=True,
                            zerolinecolor='lightgrey'
                        ),
                        xaxis=dict(
                            showgrid=False,
                            tickangle=0
                        ),
                        legend_title="Ano",
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01        
                        )
                    )

                    fig_desemprego_diff.update_traces(
                        line_color= '#2ca02c',
                        fillcolor='#d7f4d7', 
                        opacity= 0.5,                                               
                        selector=dict(mode='lines')
                    )

                    st.plotly_chart(fig_desemprego_diff, theme="streamlit", use_container_width=True)
