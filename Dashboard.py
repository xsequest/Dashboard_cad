import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sidrapy

# Função para formatar números no estilo brasileiro
def format_number_br(number):
    number_str = f"{number:,.0f}".replace(",", ".")
    return number_str

st.set_page_config(page_title="Dashboard CadÚnico", page_icon=":bar_chart:", layout="wide")

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

    url_uni = 'https://aplicacoes.cidadania.gov.br/vis/data3/v.php?q[]=oNOclsLerpibuKep3bV%2BgW9g05Kv2rmg2a19ZW51ZmymaX6JaV2JlWCadWCNrMmlsKyamembs61ojMfGpt2slLysiJqdtKiftJ%2BuuqqSkpyZy6mmwraIp7G1WKvtnbKtp4%2B9wGPJrZjQ7ryVm3lwoNqlwLNyk7jNps94bsPcuaehg3Ct7qZwyViQxsKfz7CWwqONpbCsmpnpm7OtZ4zHxqbdrJS8rHlkZWhgWtyorrqcoLrGU5J9pNHfspOsqpuZqpi9s6qgxsKSm2ZU2razlai7mnXfmrnBnGiSx5TWsJiYtsCpqcSGr9qnwbebjrvGU86iU8PcuvfptJ6b7FmPk4Vynap2swC0r8SOh1yspFrJq7y1qY7EwlOsrJ%2FQ3G16nbX45%2BWirm6dnMnOlM6epn3rvKZceVWj562ytamOxdWYimWoy%2BS9ma%2B7pJvirHZxiKK4z6fToZTB4G2YoWibm%2Bb8%2BrqgjsqBgS3Agn3dsqKhrp6d4vzuwKCOyoGX2V2Dz%2Bq0pp21llq7qLnBmE2dwqAt6p%2FG3G2aq7qim92awG6nnMmBZIqmodHgtKadtqmfmWHCvKCdvNSm2Z6c0KRwhbGpo67ina6ynE3L0KfLqVPB4G2anbX45%2BWirsFXk8bToMuhlNCbvaOuaGZa4qfBs56fuM%2Bnz11b0um2pKG7qKnaosB3s52S3a%2FmeA%3D%3D&ma=mes&ma=mes&dt1=2023-02-01&dt2=2024-08-01&dt1=2022-12-01&dt2=2024-08-01&ag=e&wt=json&tp_funcao_consulta=0&draw=2&columns[0][data]=0&columns[0][name]=codigo&columns[0][searchable]=true&columns[0][orderable]=true&columns[0][search][value]=&columns[0][search][regex]=false&columns[1][data]=1&columns[1][name]=nome&columns[1][searchable]=true&columns[1][orderable]=false&columns[1][search][value]=&columns[1][search][regex]=false&columns[2][data]=2&columns[2][name]=mes_ano_formatado&columns[2][searchable]=true&columns[2][orderable]=true&columns[2][search][value]=&columns[2][search][regex]=false&columns[3][data]=3&columns[3][name]=qtde_pbf_1_pessoa_1&columns[3][searchable]=true&columns[3][orderable]=false&columns[3][search][value]=&columns[3][search][regex]=false&columns[4][data]=4&columns[4][name]=qtde_pbf_0_pessoa_1&columns[4][searchable]=true&columns[4][orderable]=false&columns[4][search][value]=&columns[4][search][regex]=false&columns[5][data]=5&columns[5][name]=(coalesce(t.qtde_pbf_0_pessoa_1%2C0)%20%2B%20coalesce%20(t.qtde_pbf_1_pes&columns[5][searchable]=true&columns[5][orderable]=false&columns[5][search][value]=&columns[5][search][regex]=false&order[0][column]=2&order[0][dir]=asc&order[1][column]=0&order[1][dir]=asc&start=0&length=3147483647&search[value]=&search[regex]=false&export=1&export_data_comma=1&export_tipo=csv&'
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

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
st.title('Dashboard CadÚnico')


list_faixa = ["Extrema Pobreza", "Pobreza", "Baixa Renda", "Até meio salário-mínimo"]
dict_faixa = {'Extrema Pobreza': 'fam_ext_pob', 'Pobreza': 'fam_pob', 'Baixa Renda': 'fam_baixa_renda', "Até meio salário-mínimo": 'fam_acima_meio_sm'}

df_cadúnico = carrengando_informações()

with st.expander("Configurações de seleção:"):
    col_faixa, col_estado = st.columns(2)
    with col_faixa:
        with st.container(border=True):
            faixa = st.selectbox("Selecione a faixa:", (list_faixa), index=0)            
            selected_faixa = dict_faixa[faixa]
    with col_estado:
        with st.container(border=True):
            selected_estado = st.selectbox("Selecione o Estado:", (df_cadúnico['Estado'].unique()), index=8)

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
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
with tab_panorama:
    with st.container(border=True):
        with st.container(border=True):
            df_state = df_cadúnico[df_cadúnico['Ano'] <= 2023].groupby('Sigla')[selected_faixa].mean().reset_index()
            df_state = df_state.sort_values(selected_faixa, ascending=False)

            df_2024 = df_cadúnico[df_cadúnico['Ano'] == 2024].groupby('Sigla')[selected_faixa].mean().reset_index()

            df_combined = pd.merge(df_state, df_2024, on='Sigla', suffixes=('_geral', '_2024'))

            df_combined = df_combined.sort_values(f'{selected_faixa}_geral', ascending=False)

            fig = px.bar(df_combined, 
                        x='Sigla', 
                        y=f'{selected_faixa}_geral',
                        opacity=0.7,
                        labels={'Sigla': 'Estado', 
                                f'{selected_faixa}_geral': 'Número médio de famílias'})

            # Definir as cores das barras
            colors = ['green' if x == df_cadúnico.loc[df_cadúnico['Estado'] == selected_estado, 'Sigla'].unique()[0] else '#87CEEB' for x in df_combined['Sigla']]
            fig.update_traces(marker_color=colors)

            # Adicionar a linha para 2024
            fig.add_trace(
                go.Scatter(
                    x=df_combined['Sigla'],
                    y=df_combined[f'{selected_faixa}_2024'],
                    mode='lines+markers',
                    name='Média 2024',
                    line=dict(color='#FF4444', width=1, dash='dash'),  # Adicionando 'dash' aqui
                    marker=dict(size=4)  # Opcional: ajusta o tamanho dos marcadores
                )
            )

            # Atualizar o layout
            fig.update_layout(
                xaxis_tickangle=0,
                legend_title_text='Legenda'
            )
            fig.update_layout(
                title={
                    'text': f'Média de famílias em <b>{faixa} por estado</b><br>2017/2023',
                    'x': 0.5,
                    'y': 0.95,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
            )

            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        
        with st.container(border=True):

            df_seasonal = df_cadúnico_br.groupby('Mês')[selected_faixa].mean().reset_index()
            df_seasonal['Mês'] = pd.Categorical(df_seasonal['Mês'], categories=range(1, 13), ordered=True)
            df_seasonal = df_seasonal.sort_values('Mês')

            meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
            df_seasonal['Mês_nome'] = [meses[i-1] for i in df_seasonal['Mês']]

            fig_sazonalidade = px.line(df_seasonal, x='Mês_nome', y=selected_faixa, markers=True, labels={selected_faixa: faixa, 'Mês_nome': 'Mês'})

            fig_sazonalidade.update_traces(line_color='green', marker=dict(size=8, color="white", line=dict(width=2, color="green")))

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
            #############################################################################################################################################################
            st.divider()
            ############################################################################################################################################################
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



#-----------------------------------------------------------------------------------------------------------------------------------------------------
with tab_estado:
    with st.container(border=True):
        with st.container(border=True):
            
            with st.expander("Opções de Plotagem:"):
                selected_graph = st.radio("Selecione o tipo de gráfico para a visualização:", ['Linha','Candle', 'Tendência'], index=0)
           
            with st.container(border=True):
                ###########################################################################################################################
                if selected_graph == 'Candle':
                    medias_anuais_plotly = df_cadúnico_estado.groupby('Ano')[selected_faixa].mean()

                    df_yearly = df_cadúnico_estado.groupby('Ano').agg({
                        selected_faixa: ['first', 'max', 'min', 'last']
                    }).reset_index()

                    df_yearly.columns = ['Ano', 'Open', 'High', 'Low', 'Close']


                    # Convertendo o ano para um objeto datetime
                    df_yearly['Date'] = pd.to_datetime(df_yearly['Ano'].astype(str))

                    fig_candle = go.Figure()

                    # Adicionando o gráfico de velas
                    fig_candle.add_trace(go.Candlestick(
                        x=df_yearly['Date'],
                        open=df_yearly['Open'],
                        high=df_yearly['High'],
                        low=df_yearly['Low'],
                        close=df_yearly['Close'],
                        name='Candlestick'
                    ))

                    # Adicionando a linha de média
                    fig_candle.add_trace(go.Scatter(
                        x=df_yearly['Date'],
                        y=medias_anuais_plotly,
                        mode='lines+markers',
                        name='Média Anual',
                        line=dict(color='grey', width=1, dash='dash'),
                        marker=dict(symbol='circle', size=3, color='white', line=dict(width=1, color='white'))
                    ))


                    # Personalizando o layout
                    fig_candle.update_layout(
                        title={
                            'text': f'Famílias em <b>{faixa}</b> por Ano<br><sup><b>{selected_estado}</sup>',
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

                    st.plotly_chart(fig_candle, theme="streamlit", use_container_width=True)
                ##################################################################################################################################
                elif selected_graph == 'Tendência':
                    from scipy import stats
                    df_tendencia = df_cadúnico_estado.set_index('Data')[selected_faixa]
                    df_tendencia = df_tendencia.reset_index()
                    df_tendencia['tempo'] = range(len(df_tendencia))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(df_tendencia['tempo'], df_tendencia[selected_faixa])
                    linha_tendencia = slope * df_tendencia['tempo'] + intercept
                    fig_tendencia = go.Figure()

                    fig_tendencia.add_trace(go.Scatter(
                        x=df_tendencia['Data'],
                        y=df_tendencia[selected_faixa],
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
                            'text': f'Tendência: Número de Famílias em {faixa} - {selected_estado}',
                            'x': 0.5,
                            'y': 0.9,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        }
                    )

                    fig_tendencia.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_tendencia, theme="streamlit", use_container_width=True)

                ##############################################################################################
                else: # Em linha
                    meses_anuais = df_cadúnico_estado[df_cadúnico_estado['Data'].dt.month == 12]['Data']

                    fig_linha = px.line(df_cadúnico_estado, x='Data', y=selected_faixa, markers=False, labels={selected_faixa: faixa, 'Data': 'Data'})

                    # Adicionar apenas os marcadores para janeiro de cada ano
                    fig_linha.add_scatter(
                        x=meses_anuais,
                        y=df_cadúnico_estado[df_cadúnico_estado['Data'].isin(meses_anuais)][selected_faixa],
                        mode='markers',
                        marker=dict(size=8, color="white", line=dict(width=2, color="green")),
                        showlegend=False
                    )

                    # Adicionar anotações nos marcadores
                    for data in meses_anuais:
                        valor = df_cadúnico_estado[df_cadúnico_estado['Data'] == data][selected_faixa].values[0]
                        fig_linha.add_annotation(
                            x=data,
                            y=valor,
                            text=format_number_br(valor),
                            showarrow=False,
                            yshift=10,
                            font=dict(size=12)
                        )

                    fig_linha.update_traces(line_color='green', selector=dict(mode='lines'))

                    fig_linha.update_layout(
                        xaxis_title=None,
                        yaxis_title='Número de famílias',
                        xaxis=dict(tickmode='array', tickvals=meses_anuais),
                        yaxis=dict(gridcolor='lightgrey'),
                        title={
                            'text': f'Gráfico de famílias em {faixa} (2017-2024)',
                            'x': 0.5,
                            'y': 0.95,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        }
                    )
                    st.plotly_chart(fig_linha, theme="streamlit", use_container_width=True)
                

        #------------------------------------------------------------------------------------------------------------------------------------
         
        with st.expander("Gráficos Secundários:"):
            col_candle, col_crescimento = st.columns(2) 
            with col_candle:
                with st.container(border=True):
                    with st.container(border=True):
                        col_year, col_faixa = st.columns(2)
                        with col_year:
                            anos_unicos = df_cadúnico_estado['Ano'].dropna().unique()
                            anos_unicos_ordenados = sorted(anos_unicos, reverse=True)
                            selected_year = st.selectbox('Selecione o Ano:', anos_unicos_ordenados, index=0)
                        
                        with col_faixa:
                            def mostrar_opcoes(list_faixa):
                                opcoes = list_faixa.copy() 
                                opcoes.append('Todas')
                                return opcoes
                            
                            # Verifica se selected_faixa já está na lista antes de criar o multiselect
                            default_values = ['Pobreza']
                            if faixa in list_faixa:
                                default_values.insert(0, faixa)
                            
                            selected_faixa_multi = st.multiselect(
                                "Selecione a faixa de renda:",
                                mostrar_opcoes(list_faixa),
                                default=default_values, max_selections=2
                            )
                    #########################################################################################################################################
                    def format_numbers(valor):
                        return '{:,.2f}'.format(valor).replace(',', 'X').replace('.', ',').replace('X', '.')
                    from plotly.subplots import make_subplots
                    
                    if 'Todas' in selected_faixa_multi:
                        st.text("Opção 'Todas' está em construção!")
                    else:
                        if len(selected_faixa_multi) == 1:
                            ##### ESTADO single ###################################################################################################
                            df_selected_single = df_cadúnico_estado.loc[df_cadúnico_estado['Data'].dt.year == selected_year].copy()
                            y_single_formatted = [format_numbers(val) for val in df_selected_single[dict_faixa[selected_faixa_multi[0]]]]
                            meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
                            ultimo_mes = df_selected_single['Mês'].max()
                            meses_x = meses[:ultimo_mes]
                            
                            # Gráfico com uma linha
                            fig_single_select = go.Figure()
                            fig_single_select.add_trace(
                                go.Scatter(
                                    x=meses_x,
                                    y=df_selected_single[dict_faixa[selected_faixa_multi[0]]],
                                    name=selected_faixa_multi[0],
                                    line=dict(color='#84c784'),
                                    mode='lines+markers',
                                    text=y_single_formatted,
                                    hovertemplate='%{text}<extra></extra>',
                                    marker=dict(
                                        size=6,
                                        color='white',
                                        line=dict(color='#84c784', width=2)
                                    )
                                )
                            )
                            
                            fig_single_select.update_layout(
                                title={
                                    'text': f"Famílias em <b>{selected_faixa_multi[0]}</b><br>- <b>{selected_estado}</b> - {selected_year}",
                                    'y': 0.95,
                                    'x': 0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top',
                                    'font': dict(size=20)
                                },
                                showlegend=True,
                                plot_bgcolor='white',
                                yaxis=dict(
                                    showgrid=True,
                                    gridcolor='lightgray',
                                    title=dict(text=selected_faixa_multi[0], font=dict(color='#84c784')),
                                    tickfont=dict(color='#84c784')
                                ),
                            )
                            st.plotly_chart(fig_single_select, theme="streamlit", use_container_width=True)

                            st.divider()
                            ###### BRASIL single #########################################################################################################
                            df_selected_single_br = df_cadúnico_br.loc[df_cadúnico_br['Data'].dt.year == selected_year].copy()
                            y_single_formatted_br = [format_numbers(val) for val in df_selected_single_br[dict_faixa[selected_faixa_multi[0]]]]
                          
                            # Gráfico com uma linha
                            fig_single_select_br = go.Figure()
                            fig_single_select_br.add_trace(
                                go.Scatter(
                                    x=meses_x,
                                    y=df_selected_single_br[dict_faixa[selected_faixa_multi[0]]],
                                    name=selected_faixa_multi[0],
                                    line=dict(color='#506e9a'),
                                    mode='lines+markers',
                                    text=y_single_formatted_br,
                                    hovertemplate='%{text}<extra></extra>',
                                    marker=dict(
                                        size=6,
                                        color='white',
                                        line=dict(color='#506e9a', width=2)
                                    )
                                )
                            )
                            
                            fig_single_select_br.update_layout(
                                title={
                                    'text': f"Famílias em <b>{selected_faixa_multi[0]}</b><br>- <b>Brasil</b> - {selected_year}",
                                    'y': 0.95,
                                    'x': 0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top',
                                    'font': dict(size=20)
                                },
                                showlegend=True,
                                plot_bgcolor='white',
                                yaxis=dict(
                                    showgrid=True,
                                    gridcolor='lightgray',
                                    title=dict(text=selected_faixa_multi[0], font=dict(color='#506e9a')),
                                    tickfont=dict(color='#506e9a')
                                ),
                            )
                            st.plotly_chart(fig_single_select_br, theme="streamlit", use_container_width=True)

                            
                        ############################################################################################################
                        elif len(selected_faixa_multi) > 1:
                            ###### ESTADO Multi ###################################################################################
                            df_selected_multi = df_cadúnico_estado.loc[df_cadúnico_estado['Data'].dt.year == selected_year].copy()
                            meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
                            ultimo_mes = df_selected_multi['Mês'].max()
                            meses_x = meses[:ultimo_mes]
                            
                            # Formatando valores para exibição nas legendas
                            y1_formatted = [format_numbers(val) for val in df_selected_multi[dict_faixa[selected_faixa_multi[0]]]]
                            y2_formatted = [format_numbers(val) for val in df_selected_multi[dict_faixa[selected_faixa_multi[1]]]]

                            # Gráfico com duas linhas
                            fig_multi_select = make_subplots(specs=[[{"secondary_y": True}]])
                            
                            # Linha para a primeira faixa
                            fig_multi_select.add_trace(
                                go.Scatter(
                                    x=meses_x,
                                    y=df_selected_multi[dict_faixa[selected_faixa_multi[0]]],
                                    name=selected_faixa_multi[0],
                                    line=dict(color='#84c784'),
                                    mode='lines+markers',
                                    text=y1_formatted,
                                    hovertemplate='%{text}<extra></extra>',
                                    marker=dict(
                                        size=6,
                                        color='white',
                                        line=dict(color='#84c784', width=2)
                                    )
                                ),
                                secondary_y=False
                            )

                            # Linha para a segunda faixa
                            fig_multi_select.add_trace(
                                go.Scatter(
                                    x=meses_x,
                                    y=df_selected_multi[dict_faixa[selected_faixa_multi[1]]],
                                    name=selected_faixa_multi[1],
                                    line=dict(color='#00766f'),
                                    mode='lines+markers',
                                    text=y2_formatted,
                                    hovertemplate='%{text}<extra></extra>',
                                    marker=dict(
                                        size=6,
                                        color='white',
                                        line=dict(color='#00766f', width=2)
                                    )
                                ),
                                secondary_y=True
                            )

                            fig_multi_select.update_layout(
                                title={
                                    'text': f"Famílias em <b>{selected_faixa_multi[0]}</b> e <b>{selected_faixa_multi[1]}</b><br>- <b>{selected_estado}</b> - {selected_year}",
                                    'y': 0.95,
                                    'x': 0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top',
                                    'font': dict(size=20)
                                },
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=-0.2,
                                    xanchor="center",
                                    x=0.5
                                ),
                                plot_bgcolor='white',
                                yaxis=dict(
                                    showgrid=True,
                                    gridcolor='lightgray',
                                    title=dict(text=selected_faixa_multi[0], font=dict(color='#84c784')),
                                    tickfont=dict(color='#84c784')
                                ),
                                yaxis2=dict(
                                    showgrid=False,
                                    title=dict(text=selected_faixa_multi[1], font=dict(color='#00766f')),
                                    tickfont=dict(color='#00766f')
                                ),
                            )
                            st.plotly_chart(fig_multi_select, theme="streamlit", use_container_width=True)

                            st.divider()

                            ###### BRASIL Multi ##############################################################################################
                            df_selected_multi_br = df_cadúnico_br.loc[df_cadúnico_br['Data'].dt.year == selected_year].copy()
                           
                            # Formatando valores para exibição nas legendas
                            y1_formatted_br = [format_numbers(val) for val in df_selected_multi_br[dict_faixa[selected_faixa_multi[0]]]]
                            y2_formatted_br = [format_numbers(val) for val in df_selected_multi_br[dict_faixa[selected_faixa_multi[1]]]]

                            # Gráfico com duas linhas
                            fig_multi_select_br = make_subplots(specs=[[{"secondary_y": True}]])
                            
                            # Linha para a primeira faixa
                            fig_multi_select_br.add_trace(
                                go.Scatter(
                                    x=meses_x,
                                    y=df_selected_multi_br[dict_faixa[selected_faixa_multi[0]]],
                                    name=selected_faixa_multi[0],
                                    line=dict(color='#506e9a'),
                                    mode='lines+markers',
                                    text=y1_formatted_br,
                                    hovertemplate='%{text}<extra></extra>',
                                    marker=dict(
                                        size=6,
                                        color='white',
                                        line=dict(color='#506e9a', width=2)
                                    )
                                ),
                                secondary_y=False
                            )

                            # Linha para a segunda faixa
                            fig_multi_select_br.add_trace(
                                go.Scatter(
                                    x=meses_x,
                                    y=df_selected_multi_br[dict_faixa[selected_faixa_multi[1]]],
                                    name=selected_faixa_multi[1],
                                    line=dict(color='#41b8d5'),
                                    mode='lines+markers',
                                    text=y2_formatted_br,
                                    hovertemplate='%{text}<extra></extra>',
                                    marker=dict(
                                        size=6,
                                        color='white',
                                        line=dict(color='#41b8d5', width=2)
                                    )
                                ),
                                secondary_y=True
                            )

                            fig_multi_select_br.update_layout(
                                title={
                                    'text': f"Famílias em <b>{selected_faixa_multi[0]}</b> e <b>{selected_faixa_multi[1]}</b><br>- <b>Brasil</b> - {selected_year}",
                                    'y': 0.95,
                                    'x': 0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top',
                                    'font': dict(size=20)
                                },
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=-0.2,
                                    xanchor="center",
                                    x=0.5
                                ),
                                plot_bgcolor='white',
                                yaxis=dict(
                                    showgrid=True,
                                    gridcolor='lightgray',
                                    title=dict(text=selected_faixa_multi[0], font=dict(color='#506e9a')),
                                    tickfont=dict(color='#506e9a')
                                ),
                                yaxis2=dict(
                                    showgrid=False,
                                    title=dict(text=selected_faixa_multi[1], font=dict(color='#41b8d5')),
                                    tickfont=dict(color='#41b8d5')
                                ),
                            )
                            st.plotly_chart(fig_multi_select_br, theme="streamlit", use_container_width=True)

        ####################################################################################################################################################################################        
                with col_crescimento:
                    with st.container(border=True):
                        df_growth = df_cadúnico_estado.groupby('Ano')[selected_faixa].last()
                        growth_rate = round((df_growth.pct_change() * 100).dropna(), 2)
                        variacao_percentual = (df_growth.iloc[-1] / df_growth.iloc[0] - 1) * 100

                        # Definindo as cores com base nos valores de growth_rate
                        colors = ['#87CEEB' if x < 0 else '#FF4444' for x in growth_rate]

                        fig_crescimento = px.bar(
                            x=growth_rate.index, 
                            y=growth_rate,
                            labels={'x': 'Ano', 'y': 'Taxa de crescimento (%)'}, 
                            text=growth_rate.map(lambda x: f'{x:.2f}%'),
                            color=growth_rate,  # Usando growth_rate para definir as cores
                            color_discrete_sequence=None  # Removendo a sequência de cores discreta
                        )

                        # Atualizando as cores das barras
                        fig_crescimento.update_traces(marker_color=colors)

                        # Adicionando a linha de variação percentual
                        fig_crescimento.add_hline(
                            y=variacao_percentual, 
                            line_dash="dash", 
                            line_color="green",
                            line_width=1, 
                            annotation_text=f'Variação 2017-2024:<br>{variacao_percentual:.2f}%</b>', 
                            annotation_position="top left"
                        )

                        # Personalizando o layout
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

                        # Adicionando uma linha horizontal em y=0
                        fig_crescimento.add_hline(y=0, line_color='black')
                        fig_crescimento.update_traces(hovertemplate='Ano: <b>%{x}</b><br>Taxa de crescimento: <b>%{y:.2f}%</b><extra></extra>')
                        st.plotly_chart(fig_crescimento, theme="streamlit", use_container_width=True)
        
        
        with st.expander("Famílias Unipessoais Inscritas e Taxa de desocupação:"):
            col_fam_uni, col_desemprego = st.columns(2)
            #######################################################################################################################
            with col_fam_uni:
                with st.container(border=True):
                    df_selected_uni = df_fam_uni.loc[df_fam_uni['Estado'] == selected_estado].copy()

                    df_selected_uni_ind = df_selected_uni.copy()
                    df_selected_uni_ind['Data'] = pd.to_datetime(df_selected_uni_ind['Data'])

                    # Ordenar o dataframe por data
                    df_selected_uni_ind = df_selected_uni_ind.sort_values('Data')

                    # Criar o gráfico base com a linha
                    fig_fam_uni_ind = px.line(df_selected_uni_ind, 
                                x='Data', 
                                y='fam_unipessoais_inscritas')


                    # Personalizar o layout
                    fig_fam_uni_ind.update_layout(
                        title={
                            'text': f'Número de Famílias Unipessoais Inscritas em {selected_estado}<br>(2023-2024) *Janela Procad',
                            'x': 0.5,  # Centraliza o título
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
                            tickformat=','  # Formato com separador de milhares
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
                    # Atualizar a linha principal
                    fig_fam_uni_ind.update_traces(
                        line_color='#2ca02c',
                        selector=dict(mode='lines')
                    )
                    fig_fam_uni_ind.update_yaxes(
                        tickformat=",.",
                        separatethousands=True
                    )
                    st.plotly_chart(fig_fam_uni_ind, theme="streamlit", use_container_width=True)  

                    st.divider()

                    ###############################################################################################################
                    df_selected_uni['fam_inscritas'] = df_selected_uni['fam_inscritas'] / 1_000_000
                    df_selected_uni['fam_unipessoais_inscritas'] = df_selected_uni['fam_unipessoais_inscritas'] / 1_000_000

                    # Criando o gráfico de barras
                    fig_fam_uni = px.bar(df_selected_uni, 
                                x='Data', 
                                y='fam_inscritas',
                                opacity=0.5,
                                labels={'Data': 'Data', 
                                        'fam_inscritas': 'Número de famílias inscritas (milhões)'})

                    # Definir as cores das barras
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

                    # Atualizar o layout
                    fig_fam_uni.update_layout(
                        xaxis_tickangle=0,
                        legend_title_text='Legenda'
                    )
                    fig_fam_uni.update_layout(
                        title={
                            'text': 'Famílias Inscritas em Goiás: <b>Total e Unipessoais</b><br>Todas as faixas de renda',
                            'x': 0.5,
                            'y': 0.95,
                            'xanchor': 'center',
                            'yanchor': 'top'
                        },
                        xaxis_title=None,
                        yaxis_title="Número de Famílias (milhões)"
                    )

                    # Formatando os números no eixo y para o padrão brasileiro
                    fig_fam_uni.update_yaxes(
                        tickformat=",.",
                        separatethousands=True
                    )
                    st.plotly_chart(fig_fam_uni, theme="streamlit", use_container_width=True)             
            
            #################################################################################################################################
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
                            'text': f'Taxa de Desemprego<br>Brasil - {selected_estado}',  # Substitua pelo título desejado
                            'x': 0.5,  # Centraliza o título
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
                    ###########################################################################################
                    st.divider()
                    ###########################################################################################
                    datas_unicas = sorted(set(df_desemprego_estado['Data']) & set(df_desemprego_br['Data']))
                    df_diferenca = pd.DataFrame(index=datas_unicas)

                    # Calcula a diferença percentual para as datas em comum
                    df_diferenca['Diferenca_Percentual'] = round(
                        ((df_desemprego_estado.set_index('Data').loc[datas_unicas, 'Valor'] / 
                        df_desemprego_br.set_index('Data').loc[datas_unicas, 'Valor']) - 1) * 100, 2
                    )

                    # Extrai ano e mês a partir das datas
                    df_diferenca['Ano'] = pd.to_datetime(df_diferenca.index).year
                    df_diferenca['Mês'] = pd.to_datetime(df_diferenca.index).month
                    df_diferenca = df_diferenca.reset_index().rename(columns={'index': 'Data'})
                    df_diferenca = df_diferenca.sort_values('Data')

                    # Cria o gráfico de linha com Plotly Express
                    fig_desemprego_diff = px.line(
                        df_diferenca, 
                        x='Data', 
                        y='Diferenca_Percentual',
                        title=f'Diferença percentual Taxa de Desemprego<br>{selected_estado} - Brasil (2017-2024)'
                    )

                    fig_desemprego_diff.update_layout(
                        title={
                            'text': f'Diferença percentual Taxa de Desemprego<br>{selected_estado} - Brasil (2017-2024)',
                            'x': 0.5,  # Centraliza o título
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

                    # Atualiza a cor da linha do gráfico
                    fig_desemprego_diff.update_traces(
                        line_color='#2ca02c',
                        selector=dict(mode='lines')
                    )

                    # Exibe o gráfico no Streamlit
                    st.plotly_chart(fig_desemprego_diff, theme="streamlit", use_container_width=True)
