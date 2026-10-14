"""
NEXUS SAFE - Dashboard Web (Streamlit)
Como rodar:   pip install streamlit plotly
              streamlit run dashboard.py
Se os arquivos de resultado não existirem, o dashboard executa o protótipo para gerá-los.
"""

import json
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nexus_safe_prototype import FATORES, NIVEIS, GemeoDigital, GrafoRiscoEmergente, executar

PASTA = Path(__file__).parent
CSV, JSON = PASTA / 'nexus_safe_resultados.csv', PASTA / 'nexus_safe_resumo.json'
CORES = {'NORMAL': '#2e9e5b', 'ATENÇÃO': '#e6c229', 'ALERTA': '#f08a24', 'CRÍTICO': '#d62f2f'}

st.set_page_config(page_title='NEXUS SAFE', layout='wide')


@st.cache_data
def carregar():
    if not (CSV.exists() and JSON.exists()):
        executar(pasta=PASTA, verbose=False)

    df = pd.read_csv(CSV, parse_dates=["timestamp"])

    # Cria um índice sequencial para o dashboard
    if "seq" not in df.columns:
        df.insert(0, "seq", range(len(df)))

    df = df.set_index("seq")

    resumo = json.loads(JSON.read_text(encoding="utf-8"))

    return df, resumo


df, resumo = carregar()
av = resumo.get("avaliacao", {})
tr = resumo.get("trafego_e_latencia", {})

st.title('NEXUS SAFE - Monitoramento de risco industrial')
st.caption('Protótipo com dados sintéticos. Aquisição, borda e central são simuladas em um único computador.')

c1, c2, c3, c4 = st.columns(4)
c1.metric('Leituras analisadas', resumo['total_leituras'])
c2.metric('Alertas críticos', resumo['distribuicao_alertas'].get('CRÍTICO', 0))
c3.metric('Redução de mensagens (borda)', f"{tr.get('reducao_mensagens_pct', 'N/D')}%")
c4.metric('Latência na borda (média)', f"{tr.get('latencia_borda_media_ms', 'N/D')} ms")

aba1, aba2, aba3, aba4 = st.tabs(['Linha do tempo', 'Grafo de risco', 'Gêmeo digital (e se?)', 'Avaliação'])

with aba1:
    niveis = st.multiselect('Níveis exibidos na tabela', list(CORES), default=['CRÍTICO', 'ALERTA'])
    fig = go.Figure()
    fig.add_scatter(x=df['timestamp'], y=df['risco'], name='Risco atual', line=dict(color='#1f4e79'))
    fig.add_scatter(x=df['timestamp'], y=df['risco_previsto'], name='Risco previsto', line=dict(color='#999', dash='dot'))
    for nome, lim in NIVEIS.items():
        fig.add_hline(y=lim, line_dash='dash', line_width=1, annotation_text=nome.title())
    for nivel in ['ATENÇÃO', 'ALERTA', 'CRÍTICO']:
        s = df[df['nivel'] == nivel]
        fig.add_scatter(x=s['timestamp'], y=s['risco'], mode='markers', name=nivel, marker=dict(color=CORES[nivel], size=7))
    fig.update_layout(height=380, yaxis_title='Risco (0 a 1)', margin=dict(t=20))
    st.plotly_chart(fig, width='stretch')
    tabela = df[df['nivel'].isin(niveis)][['timestamp', 'nivel', 'risco', 'risco_individual', 'risco_previsto',
                                           'fatores', 'cenario']]
    st.dataframe(tabela.round({'risco': 2, 'risco_individual': 2, 'risco_previsto': 2}), width='stretch')

with aba2:
    seq = st.slider('Leitura (seq)', int(df.index.min()), int(df.index.max()), int(df['risco'].idxmax()))
    linha = df.loc[seq]
    grafo = GrafoRiscoEmergente()
    ativ = grafo.ativacoes({c: linha[c] for c in FATORES})
    pos = nx.spring_layout(grafo.G, seed=7)
    fig = go.Figure()
    for u, v, d in grafo.G.edges(data=True):
        cor = '#bbb' if d['tipo'] == 'individual' else '#d62f2f'
        fig.add_scatter(x=[pos[u][0], pos[v][0]], y=[pos[u][1], pos[v][1]], mode='lines',
                        line=dict(color=cor, width=1 + 4 * d['weight']), hoverinfo='skip', showlegend=False)
    nos = list(grafo.G.nodes)
    fig.add_scatter(
        x=[pos[n][0] for n in nos], y=[pos[n][1] for n in nos], mode='markers+text',
        text=[grafo.G.nodes[n]['label'] for n in nos], textposition='top center',
        marker=dict(size=[34] * len(nos), color=[ativ.get(n, linha['risco']) for n in nos],
                    colorscale='YlOrRd', cmin=0, cmax=1, showscale=True, colorbar=dict(title='Ativação')),
        showlegend=False)
    fig.update_layout(height=480, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(t=20))
    st.plotly_chart(fig, width='stretch')
    st.caption('Arestas vermelhas = correlações que amplificam o risco; cinza = contribuição individual. '
               f"Nível: {linha['nivel']} | risco {linha['risco']:.2f} (individual {linha['risco_individual']:.2f}).")

with aba3:
    st.write('Altere as variáveis para simular um cenário sem mexer na máquina real.')
    base = {'temperatura': 45.0, 'vibracao': 2.2, 'pressao': 10.5, 'fadiga_operador': 3.0, 'umidade': 50.0, 'presenca_area_risco': 0}
    faixas = {'temperatura': (20.0, 100.0), 'vibracao': (0.0, 10.0), 'pressao': (0.0, 30.0),
              'fadiga_operador': (0.0, 10.0), 'umidade': (0.0, 100.0)}
    alter = {c: st.slider(FATORES[c]['rotulo'], *faixas[c], float(base[c])) for c in faixas}
    alter['presenca_area_risco'] = int(st.checkbox('Operador em área de risco'))
    gemeo = GemeoDigital('MAQ01')
    res = gemeo.simular(alter, estado_base=base)
    a, b = st.columns(2)
    a.metric('Risco com o estado normal', f"{res['risco_antes']:.2f}")
    b.metric('Risco no cenário simulado', f"{res['risco_depois']:.2f}", f"{res['variacao']:+.2f}")
    st.write('Fatores ativos:', ', '.join(res['fatores_depois']) or 'nenhum',
             f"| ganho por correlação: {res['ganho_correlacao']:.2f}")

with aba4:
    st.subheader('Precisão, recall e F1 (rótulos reais do gerador)')
    st.dataframe(pd.DataFrame(av['geral']).T, width='stretch')
    st.subheader('Recall por tipo de cenário')
    st.dataframe(pd.DataFrame(av['recall_por_cenario']).T, width='stretch')
    st.subheader('Predição')
    st.json(av['predicao'])
    st.subheader('Segurança das mensagens')
    st.json(resumo['seguranca_mensagens'])
    st.warning('Resultados obtidos com dados sintéticos e pesos definidos pelos autores. '
               'Demonstram o funcionamento do fluxo, não o desempenho em ambiente industrial real.')