"""
NEXUS SAFE - Protótipo v2
Segurança industrial preditiva: Grafo de Risco Emergente + detecção de anomalias (Isolation Forest).

O que esta versão implementa (tudo SIMULADO em um único computador, em Python):
  - Aquisição estilo SCADA/IIoT : "tags" publicadas por um simulador (SimuladorSCADA)
  - Barramento de mensagens     : fila em memória que imita um broker MQTT, com assinatura HMAC
  - Edge Computing              : NoBorda faz a inferência local e só transmite eventos relevantes
  - Nuvem/central               : NoCentral treina o modelo, valida mensagens e consolida eventos
  - Grafo de Risco Emergente    : o risco é CALCULADO a partir dos nós e arestas do grafo
  - Predição                    : tendência (regressão linear em janela) projetada para N minutos
  - Gêmeo Digital simplificado  : estado virtual da máquina + simulação "e se?" (what-if)
  - Avaliação                   : precisão/recall/F1 contra rótulos reais do gerador e baselines
"""

import hashlib
import hmac
import json
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support

# ==============================================================================
# 0. CONFIGURAÇÃO (fonte única de verdade para limiares, pesos e correlações)
# ==============================================================================

# [EXPLICAÇÃO]: Aqui ficam as regras básicas dos seus sensores[cite: 1]. 
# inicio = valor normal; limiar = valor perigoso; peso = gravidade[cite: 1].
# [O QUE ALTERAR]: Para adicionar um sensor novo (ex: 'fluxo_agua'), basta criar uma nova linha aqui com seus limites[cite: 1].
FATORES = {
    'temperatura':         dict(no='temp',     rotulo='Temperatura elevada',       peso=0.30, inicio=55,  limiar=70),
    'vibracao':            dict(no='vibra',    rotulo='Vibração alta',             peso=0.35, inicio=4.5, limiar=7),
    'pressao':             dict(no='pressao',  rotulo='Pressão elevada',           peso=0.25, inicio=14,  limiar=20),
    'fadiga_operador':     dict(no='fadiga',   rotulo='Fadiga do operador',        peso=0.28, inicio=5,   limiar=7),
    'umidade':             dict(no='umidade',  rotulo='Umidade anômala',           peso=0.15, inicio=20,  limiar=30, centro=50),
    'presenca_area_risco': dict(no='presenca', rotulo='Operador em área de risco', peso=0.20, inicio=0,   limiar=1),
}

# [EXPLICAÇÃO]: Quando dois problemas acontecem ao mesmo tempo, o perigo aumenta (se multiplicam)[cite: 1].
# [O QUE ALTERAR]: Se você adicionou um sensor acima, crie as combinações perigosas dele aqui[cite: 1].
CORRELACOES = [
    ('temp', 'vibra', 0.60),
    ('vibra', 'fadiga', 0.50),
    ('pressao', 'temp', 0.45),
    ('umidade', 'temp', 0.40),
    ('presenca', 'temp', 0.50),
    ('presenca', 'vibra', 0.50),
]

# [EXPLICAÇÃO]: O limite de pontuação de risco para disparar os alarmes[cite: 1].
NIVEIS = {'CRITICO': 0.70, 'ALERTA': 0.40, 'ATENCAO': 0.25}

# [EXPLICAÇÃO]: Traduz o nome técnico do sensor na fábrica (ex: MAQ01.TEMP) para o nome no código (temperatura)[cite: 1].
TAG_PARA_COLUNA = {
    'MAQ01.TEMP': 'temperatura',
    'MAQ01.VIB': 'vibracao',
    'MAQ01.PRESS': 'pressao',
    'OPER01.FADIGA': 'fadiga_operador',
    'AMB01.UMID': 'umidade',
    'AREA01.PRESENCA': 'presenca_area_risco',
}

CHAVE_SECRETA = b'chave-de-demonstracao'   # Senha para garantir segurança das mensagens
HEARTBEAT = 60                              # Envia um "Estou vivo" a cada 60 leituras
HORIZONTE_MIN = 8                           # Quantos minutos no futuro o sistema vai tentar prever
JANELA_TENDENCIA = 8                        # Quantas leituras passadas usar para adivinhar o futuro


def classificar_nivel(risco, anomalia_ia):
    """
    [EXPLICAÇÃO]: Transforma um número de risco (ex: 0.80) em um alerta legível ('CRÍTICO')[cite: 1].
    [O QUE ALTERAR]: Se quiser criar um nível "PERIGO EXTREMO", adicione a regra (o IF) aqui e o valor na variável NIVEIS[cite: 1].
    Corrige o ramo inalcançável da v1: a IA sozinha só eleva a ATENÇÃO.
    """
    if risco >= NIVEIS['CRITICO']:
        return 'CRÍTICO'
    if risco >= NIVEIS['ALERTA']:
        return 'ALERTA'
    if risco >= NIVEIS['ATENCAO'] or anomalia_ia:
        return 'ATENÇÃO'
    return 'NORMAL'


# ==============================================================================
# 1. GRAFO DE RISCO EMERGENTE
# ==============================================================================

class GrafoRiscoEmergente:
    """
    [EXPLICAÇÃO]: Constrói a rede lógica. Ele soma os riscos de cada sensor.
    Nós = fatores de risco; arestas ponderadas = contribuição individual e correlações.
    O risco final é calculado A PARTIR do grafo.
    """

    def __init__(self):
        # [O QUE FAZ]: Inicializa os sensores e as pontes entre eles utilizando as configurações[cite: 1].
        self.G = nx.DiGraph()
        for col, f in FATORES.items():
            self.G.add_node(f['no'], coluna=col, label=f['rotulo'], ativacao=0.0)
        self.G.add_node('risco_critico', label='RISCO CRÍTICO')
        for f in FATORES.values():
            self.G.add_edge(f['no'], 'risco_critico', weight=f['peso'], tipo='individual')
        for u, v, w in CORRELACOES:
            self.G.add_edge(u, v, weight=w, tipo='correlacao')

    @staticmethod
    def _ativacao(f, valor):
        # [O QUE FAZ]: Converte o valor medido em uma nota de 0 (seguro) a 1 (perigo total) baseando-se no 'limiar'[cite: 1].
        x = abs(valor - f['centro']) if 'centro' in f else valor
        return float(np.clip((x - f['inicio']) / (f['limiar'] - f['inicio']), 0.0, 1.0))

    def ativacoes(self, leitura):
        return {f['no']: self._ativacao(f, leitura[col]) for col, f in FATORES.items()}

    def calcular(self, leitura, persistir=False):
        """
        [EXPLICAÇÃO]: Esta função realiza a matemática do risco e aplica as regras de correlação para amplificar ameaças simultâneas[cite: 1].
        Retorna o risco (0-1). Com persistir=True o estado dos nós é atualizado (grafo dinâmico).
        """
        ativ = self.ativacoes(leitura)
        if persistir:
            nx.set_node_attributes(self.G, ativ, 'ativacao')

        individual = sum(self.G[n]['risco_critico']['weight'] * a for n, a in ativ.items())
        risco = individual
        for u, v, d in self.G.edges(data=True):
            if d['tipo'] == 'correlacao':
                risco *= 1 + d['weight'] * min(ativ[u], ativ[v])

        risco, individual = min(risco, 1.0), min(individual, 1.0)
        return {
            'risco': risco,
            'risco_individual': individual,
            'ganho_correlacao': risco - individual,   # parcela "emergente"
            'fatores': [self.G.nodes[n]['label'] for n, a in ativ.items() if a >= 0.5],
        }


# ==============================================================================
# 2. SIMULADOR DE AQUISIÇÃO (SCADA/IIoT) E GERADOR DE CENÁRIOS ROTULADOS
# ==============================================================================

class SimuladorSCADA:
    # [EXPLICAÇÃO]: Forja dados como se viessem de sensores verdadeiros para permitir os testes sem a máquina de verdade[cite: 1].
    TIPOS_PONTUAIS = ['temp_alta', 'vibra_alta', 'pressao_temp', 'combinado_acima', 'combinado_sublimiar']

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def _normal(self, n):
        # [O QUE ALTERAR]: Se mudar algum sensor lá no topo (FATORES), tem que criar a média falsa de funcionamento normal dele aqui embaixo[cite: 1].
        r = self.rng
        return pd.DataFrame({
            'temperatura': r.normal(45, 3, n),
            'vibracao': r.normal(2.2, 0.4, n),
            'pressao': r.normal(10.5, 0.8, n),
            'fadiga_operador': r.normal(3, 1, n),
            'umidade': r.normal(50, 8, n),
            'presenca_area_risco': (r.random(n) < 0.20).astype(int),
        })

    def gerar_baseline(self, n=1000):
        """
        [EXPLICAÇÃO]: Histórico de operação normal, usado só para TREINAR o modelo de IA (separado do teste)[cite: 1].
        """
        return self._normal(n)

    def gerar_teste(self, n=600, n_por_tipo=10, n_derivas=4, comp_deriva=15):
        # [EXPLICAÇÃO]: Cria dados com defeitos artificiais injetados para testar se as contas estão apitando os alarmes certos[cite: 1].
        r = self.rng
        df = self._normal(n)
        df['timestamp'] = pd.date_range('2026-10-01 08:00', periods=n, freq='min')
        df['cenario'] = 'normal'
        df['episodio'] = -1
        livres = set(range(n))

        # Derivas: temperatura e vibração subindo ao longo de ~15 min (dá estrutura temporal à predição)
        inicios = np.linspace(40, n - 40, n_derivas).astype(int)
        t = np.linspace(0, 1, comp_deriva)
        for e, ini in enumerate(inicios):
            idx = np.arange(ini, ini + comp_deriva)
            livres -= set(idx.tolist())
            df.loc[idx, 'temperatura'] = 48 + 30 * t + r.normal(0, 1, comp_deriva)
            df.loc[idx, 'vibracao'] = 2.5 + 5 * t + r.normal(0, 0.2, comp_deriva)
            df.loc[idx, 'fadiga_operador'] = r.normal(6, 0.5, comp_deriva)
            df.loc[idx, 'presenca_area_risco'] = 1
            df.loc[idx, 'cenario'] = 'deriva'
            df.loc[idx, 'episodio'] = e

        # Situações pontuais rotuladas
        posicoes = r.choice(sorted(livres), size=n_por_tipo * len(self.TIPOS_PONTUAIS), replace=False)
        for i, p in enumerate(posicoes):
            tipo = self.TIPOS_PONTUAIS[i % len(self.TIPOS_PONTUAIS)]
            df.loc[p, 'cenario'] = tipo
            if tipo == 'temp_alta':
                df.loc[p, 'temperatura'] = r.uniform(72, 85)
            elif tipo == 'vibra_alta':
                df.loc[p, 'vibracao'] = r.uniform(7.5, 9)
            elif tipo == 'pressao_temp':
                df.loc[p, 'pressao'] = r.uniform(20, 24)
                df.loc[p, 'temperatura'] = r.uniform(70, 76)
            elif tipo == 'combinado_acima':
                df.loc[p, ['temperatura', 'vibracao', 'fadiga_operador']] = [r.uniform(70, 80), r.uniform(7, 9), r.uniform(7, 10)]
                df.loc[p, 'presenca_area_risco'] = 1
            else:  # combinado_sublimiar: cada fator ainda "normal" sozinho, perigoso em conjunto
                df.loc[p, ['temperatura', 'vibracao', 'fadiga_operador']] = [r.uniform(60, 66), r.uniform(4.8, 6.0), r.uniform(5.2, 6.3)]
                df.loc[p, 'presenca_area_risco'] = 1

        cols = [c for c in FATORES if c != 'presenca_area_risco']
        df[cols] = df[cols].clip(lower=0)
        # rótulo real: 1 = situação de risco injetada. Derivas são avaliadas à parte (antecedência).
        df['rotulo_real'] = (~df['cenario'].isin(['normal', 'deriva'])).astype(int)
        return df

    @staticmethod
    def para_tags(linha, seq):
        # [O QUE FAZ]: Formata as informações para parecerem dados enviados por equipamentos reais nas fábricas[cite: 1].
        tags = {tag: float(linha[col]) for tag, col in TAG_PARA_COLUNA.items()}
        return {'seq': seq, 'ts': str(linha['timestamp']), 'qualidade': 'GOOD', 'tags': tags}


# ==============================================================================
# 3. DETECÇÃO DE ANOMALIAS (IA)
# ==============================================================================

class DetectorAnomaliasIA:
    """
    [EXPLICAÇÃO]: Usa IA (Machine Learning) para encontrar anomalias. 
    Isolation Forest treinado em histórico NORMAL (aprendizado não supervisionado).
    Não usa StandardScaler: árvores de isolamento não dependem de escala.
    """

    def __init__(self, taxa_falso_alarme=0.01):
        self.modelo = IsolationForest(n_estimators=100, contamination=taxa_falso_alarme, random_state=42)
        self.colunas = list(FATORES)

    def treinar(self, baseline):
        # [O QUE FAZ]: Ensina ao modelo o que é considerado "saudável"[cite: 1].
        self.modelo.fit(baseline[self.colunas].values)
        return self

    def avaliar(self, leitura):
        # [O QUE FAZ]: Pega uma leitura nova e diz se é estranha ou não em relação ao treino[cite: 1].
        X = np.array([[leitura[c] for c in self.colunas]])
        anomalia = bool(self.modelo.predict(X)[0] == -1)
        score = float(-self.modelo.score_samples(X)[0])   # ~0.5 = fronteira; maior = mais anômalo
        return anomalia, score


# ==============================================================================
# 4. BARRAMENTO DE MENSAGENS (imita um broker MQTT) COM INTEGRIDADE HMAC
# ==============================================================================

class Barramento:
    # [EXPLICAÇÃO]: Imita o transporte de rede. Ele passa as mensagens dos sensores da máquina para o servidor central[cite: 1].
    def __init__(self, chave=CHAVE_SECRETA):
        self._q = queue.Queue()
        self._chave = chave
        self.mensagens = 0
        self.bytes = 0

    def _assinar(self, corpo):
        # [O QUE FAZ]: Cria uma assinatura digital blindando os dados contra invasores e modificações[cite: 1].
        return hmac.new(self._chave, corpo.encode(), hashlib.sha256).hexdigest()

    def publicar(self, topico, payload, assinatura=None):
        corpo = json.dumps(payload, sort_keys=True, default=str)
        self._q.put({'topico': topico, 'corpo': corpo, 'assinatura': assinatura or self._assinar(corpo)})
        self.mensagens += 1
        self.bytes += len(corpo.encode())

    def encerrar(self):
        self._q.put(None)

    def consumir(self):
        return self._q.get()

    def validar(self, msg):
        """
        [EXPLICAÇÃO]: Confere se o dado que chegou é autêntico e não foi alterado na rede antes de o sistema abrir[cite: 1].
        Retorna o payload se a assinatura for válida; caso contrário, None.
        """
        if not hmac.compare_digest(msg['assinatura'], self._assinar(msg['corpo'])):
            return None
        return json.loads(msg['corpo'])


# ==============================================================================
# 5. NÓ DE BORDA (EDGE) - inferência local, transmissão seletiva
# ==============================================================================

class NoBorda:
    # [EXPLICAÇÃO]: Representa um equipamento grudado na máquina rodando a IA de forma ultra-rápida[cite: 1].
    def __init__(self, id_no, barramento):
        self.id = id_no
        self.bus = barramento
        self.grafo = GrafoRiscoEmergente()
        self.detector = None
        self.janela = deque(maxlen=JANELA_TENDENCIA)
        self.log = []          # log local (usado na avaliação do experimento)
        self.enviados = []

    def implantar_modelo(self, detector):
        """
        [O QUE FAZ]: Recebe a IA treinada do servidor e instala na máquina local.
        Modelo treinado na central e implantado na borda (treino na nuvem, inferência na borda).
        """
        self.detector = detector

    def _projetar(self):
        # [EXPLICAÇÃO]: Tenta adivinhar (predição matemática) o que vai acontecer no futuro avaliando os dados dos últimos minutos[cite: 1].
        if len(self.janela) < self.janela.maxlen:
            return None
        cols = list(FATORES)
        Y = np.array([[l[c] for c in cols] for l in self.janela])
        t = np.arange(len(Y))
        # Inclinação robusta (estilo Theil-Sen): mediana das inclinações entre todos os pares de pontos.
        # Um pico isolado quase não altera a mediana, ao contrário de uma regressão por mínimos quadrados.
        i, j = np.triu_indices(len(Y), k=1)
        a = np.median((Y[j] - Y[i]) / (j - i)[:, None], axis=0)
        b = np.median(Y - a * t[:, None], axis=0)
        proj = dict(zip(cols, a * (len(Y) - 1 + HORIZONTE_MIN) + b))
        proj['presenca_area_risco'] = float(self.janela[-1]['presenca_area_risco'])
        return {c: max(v, 0.0) for c, v in proj.items()}

    def processar(self, msg_tags):
        # [EXPLICAÇÃO]: Essa é a função principal que trabalha em tempo real. Pega as leituras e analisa TUDO[cite: 1].
        # Ela é a responsável por não enviar lixo pra nuvem: só envia se de fato tiver um risco/evento (transmitindo menos arquivos)[cite: 1].
        t0 = time.perf_counter()
        leitura = {col: msg_tags['tags'][tag] for tag, col in TAG_PARA_COLUNA.items()}
        self.janela.append(leitura)

        r = self.grafo.calcular(leitura, persistir=True)
        anomalia, score = self.detector.avaliar(leitura)
        proj = self._projetar()
        risco_prev = self.grafo.calcular(proj)['risco'] if proj else 0.0
        preditivo = proj is not None and risco_prev >= NIVEIS['CRITICO'] and r['risco'] < NIVEIS['CRITICO']
        nivel = classificar_nivel(r['risco'], anomalia)
        proc_ms = (time.perf_counter() - t0) * 1000

        registro = {
            'seq': msg_tags['seq'], 'ts': msg_tags['ts'], 'risco': r['risco'],
            'risco_individual': r['risco_individual'], 'ganho_correlacao': r['ganho_correlacao'],
            'risco_previsto': risco_prev, 'alerta_preditivo': bool(preditivo),
            'anomalia_ia': anomalia, 'score_ia': score, 'nivel': nivel,
            'fatores': ', '.join(r['fatores']) or 'Nenhum', 'proc_ms': proc_ms,
        }
        self.log.append(registro)

        evento = nivel != 'NORMAL' or preditivo
        if evento or msg_tags['seq'] % HEARTBEAT == 0:
            payload = {**registro, 'id_no': self.id, 'leitura': leitura,
                       'tipo': 'evento' if evento else 'heartbeat', 't_envio': time.perf_counter()}
            self.enviados.append(payload)
            self.bus.publicar(f'nexus/{self.id}/{payload["tipo"]}', payload)


# ==============================================================================
# 6. GÊMEO DIGITAL SIMPLIFICADO
# ==============================================================================

class GemeoDigital:
    """
    [EXPLICAÇÃO]: Permite você "brincar" com variáveis fictícias para avaliar como os riscos seriam na vida real ("what if" / "E Se?")[cite: 1].
    Representação virtual do estado da máquina/operador, com simulação 'e se?'.
    Não é um modelo físico: reutiliza o Grafo de Risco para avaliar alterações hipotéticas.
    """

    def __init__(self, id_maquina):
        self.id = id_maquina
        self.grafo = GrafoRiscoEmergente()
        self.estado = {}
        self.historico = deque(maxlen=500)

    def sincronizar(self, leitura):
        self.estado = dict(leitura)
        self.historico.append(dict(leitura))

    def simular(self, alteracoes, estado_base=None):
        # [O QUE FAZ]: Analisa o nível de risco entre as informações falsas que você inseriu em "alterações" contra o estado atual da máquina[cite: 1].
        base = dict(estado_base or self.estado)
        antes = self.grafo.calcular(base)
        depois = self.grafo.calcular({**base, **alteracoes})
        return {
            'alteracoes': alteracoes,
            'risco_antes': antes['risco'],
            'risco_depois': depois['risco'],
            'variacao': depois['risco'] - antes['risco'],
            'ganho_correlacao': depois['ganho_correlacao'],
            'fatores_depois': depois['fatores'],
        }


# ==============================================================================
# 7. NÓ CENTRAL (nuvem) - treino, validação de mensagens, consolidação
# ==============================================================================

class NoCentral(threading.Thread):
    # [EXPLICAÇÃO]: Simula o computador chefe (nuvem/servidor) onde chegam apenas as mensagens mais importantes e filtradas[cite: 1].
    def __init__(self, barramento):
        super().__init__(daemon=True)
        self.bus = barramento
        self.gemeo = GemeoDigital('MAQ01')
        self.eventos = []
        self.latencias_ms = []
        self.rejeitadas = {'assinatura_invalida': 0, 'replay': 0}
        self._ultima_seq = {}

    @staticmethod
    def treinar_modelo(baseline):
        return DetectorAnomaliasIA().treinar(baseline)

    def run(self):
        # [O QUE FAZ]: Fica lendo a fila de forma automática recusando dados se a assinatura ou ordem não for segura[cite: 1].
        while True:
            msg = self.bus.consumir()
            if msg is None:
                break
            payload = self.bus.validar(msg)
            if payload is None:
                self.rejeitadas['assinatura_invalida'] += 1
                continue
            if payload['seq'] <= self._ultima_seq.get(payload['id_no'], -1):
                self.rejeitadas['replay'] += 1
                continue
            self._ultima_seq[payload['id_no']] = payload['seq']
            self.latencias_ms.append((time.perf_counter() - payload['t_envio']) * 1000)
            self.gemeo.sincronizar(payload['leitura'])
            self.eventos.append(payload)


# ==============================================================================
# 8. AVALIAÇÃO EXPERIMENTAL
# ==============================================================================

# [EXPLICAÇÃO]: Funções matemáticas que vão validar e dar notas ao sistema no final simulando se o programa de fato foi útil e se não falhou[cite: 1].
def regra_isolada(df):
    """
    [EXPLICAÇÃO]: Método antigo usado por fábricas (se o sensor X ficar vermelho, avisa) para usar de comparação contra a inteligência nova desse projeto[cite: 1].
    Baseline sem correlação: qualquer variável física acima do seu limiar (presença não conta).
    """
    flag = pd.Series(False, index=df.index)
    for col, f in FATORES.items():
        if col == 'presenca_area_risco':
            continue
        x = (df[col] - f['centro']).abs() if 'centro' in f else df[col]
        flag |= x >= f['limiar']
    return flag


def metricas(y, pred):
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    return {'precisao': round(float(p), 3), 'recall': round(float(r), 3), 'f1': round(float(f1), 3)}


def avaliar(df):
    # [O QUE FAZ]: Cruza os métodos e as precisões comparando de maneira técnica e retorna um relatório de eficiência[cite: 1].
    d = df[df['cenario'] != 'deriva'].copy()
    metodos = {
        'A) Regra isolada (limiares)': d['regra_isolada'],
        'B) Isolation Forest': d['anomalia_ia'],
        'C) Grafo de risco': d['risco'] >= NIVEIS['ATENCAO'],
        'D) Grafo + IA (nível != NORMAL)': d['nivel'] != 'NORMAL',
    }
    geral = {nome: metricas(d['rotulo_real'], pred.astype(int)) for nome, pred in metodos.items()}
    positivos = d[d['rotulo_real'] == 1]
    recall_cenario = {
        nome: {c: round(float(pred[positivos.index][positivos['cenario'] == c].mean()), 3)
               for c in sorted(positivos['cenario'].unique())}
        for nome, pred in metodos.items()
    }

    # Predição: antecedência = minutos entre o alerta preditivo e o risco CRÍTICO (por deriva)
    antecedencias = []
    for e in sorted(df.loc[df['episodio'] >= 0, 'episodio'].unique()):
        ep = df[df['episodio'] == e].reset_index(drop=True)
        crit = ep.index[ep['risco'] >= NIVEIS['CRITICO']]
        prev = ep.index[ep['alerta_preditivo']]
        if len(crit):
            antecedencias.append(int(crit[0] - prev[0]) if len(prev) else None)
    validas = [a for a in antecedencias if a is not None and a > 0]
    normais = df[df['cenario'] == 'normal']
    predicao = {
        'antecedencia_por_deriva_min': antecedencias,
        'antecedencia_media_min': round(float(np.mean(validas)), 2) if validas else 0.0,
        'derivas_com_antecedencia': f'{len(validas)}/{len(antecedencias)}',
        'falso_alarme_preditivo_em_normais_pct': round(float(normais['alerta_preditivo'].mean() * 100), 2),
    }

    # Risco emergente: casos que só atingem CRÍTICO por causa das correlações (soma individual < limite)
    emergente = d[(d['risco_individual'] < NIVEIS['CRITICO']) & (d['risco'] >= NIVEIS['CRITICO'])]
    return {'geral': geral, 'recall_por_cenario': recall_cenario, 'predicao': predicao,
            'casos_risco_emergente': int(len(emergente)),
            'casos_emergentes_que_regra_isolada_perdeu': int((~emergente['regra_isolada']).sum())}


# ==============================================================================
# 9. PIPELINE COMPLETO
# ==============================================================================

def executar(n_leituras=600, seed=42, ritmo_s=0.0, pasta=Path('.'), verbose=True):
    # [EXPLICAÇÃO]: Essa é a função principal que "liga" todos os botões que detalhamos acima[cite: 1].
    pasta = Path(pasta)
    
    # 1. Cria a simulação falsa[cite: 1]
    sim = SimuladorSCADA(seed=seed)
    baseline = sim.gerar_baseline(1000)
    teste = sim.gerar_teste(n_leituras)

    # 2. Cria o servidor, o barramento e as conexões e Inicia a IA com os dados básicos de treino[cite: 1]
    bus = Barramento()
    central = NoCentral(bus)
    borda = NoBorda('borda-MAQ01', bus)
    borda.implantar_modelo(central.treinar_modelo(baseline))   # treino na central -> inferência na borda
    central.start()

    # 3. Começa a testar e repassar os dados lidos 1 por 1[cite: 1]
    for seq, (_, linha) in enumerate(teste.iterrows(), start=1):   # "tempo real": leitura a leitura
        borda.processar(sim.para_tags(linha, seq))
        if ritmo_s:
            time.sleep(ritmo_s)

    trafego = {'mensagens_legitimas': bus.mensagens, 'bytes': bus.bytes}
    # Testes de segurança: mensagem forjada (assinatura inválida) e repetição de mensagem antiga (replay)[cite: 1]
    bus.publicar('nexus/borda-MAQ01/evento', {**borda.enviados[-1], 'seq': 10**6}, assinatura='0' * 64)
    bus.publicar('nexus/borda-MAQ01/evento', borda.enviados[0])
    bus.encerrar()
    central.join()

    # Junta o log da borda com os rótulos do gerador
    log = pd.DataFrame(borda.log).set_index('seq')
    df = teste.copy()
    df.index = df.index + 1
    df = df.join(log.drop(columns=['ts']))
    df['regra_isolada'] = regra_isolada(df)
    df.index.name = 'seq'

    n = len(df)
    trafego.update({
        'leituras_brutas': n,
        'reducao_mensagens_pct': round((1 - trafego['mensagens_legitimas'] / n) * 100, 1),
        'latencia_borda_media_ms': round(float(df['proc_ms'].mean()), 2),
        'latencia_borda_p95_ms': round(float(df['proc_ms'].quantile(0.95)), 2),
        'latencia_fim_a_fim_media_ms': round(float(np.mean(central.latencias_ms)), 2),
    })

    gemeo_demo = central.gemeo.simular(
        {'vibracao': 5.5, 'temperatura': 64, 'fadiga_operador': 6, 'presenca_area_risco': 1},
        estado_base={'temperatura': 45, 'vibracao': 2.2, 'pressao': 10.5, 'fadiga_operador': 3,
                     'umidade': 50, 'presenca_area_risco': 1})

    resumo = {
        'total_leituras': n,
        'distribuicao_alertas': df['nivel'].value_counts().to_dict(),
        'risco_medio_pct': round(float(df['risco'].mean() * 100), 2),
        'risco_maximo_pct': round(float(df['risco'].max() * 100), 2),
        'avaliacao': avaliar(df),
        'trafego_e_latencia': trafego,
        'seguranca_mensagens': {**central.rejeitadas, 'eventos_aceitos': len(central.eventos)},
        'gemeo_digital_what_if': gemeo_demo,
        'grafo': {'nos': GrafoRiscoEmergente().G.number_of_nodes(),
                  'arestas': GrafoRiscoEmergente().G.number_of_edges()},
    }

    # 4. Salva todo o histórico da simulação na sua pasta[cite: 1]
    df.to_csv(pasta / 'nexus_safe_resultados.csv')
    (pasta / 'nexus_safe_resumo.json').write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding='utf-8')
    if verbose:
        imprimir_resumo(resumo)
    return df, resumo


def imprimir_resumo(r):
    # [O QUE FAZ]: "Cuspi" (exibe) o relatório e todos os resumos diretamente na sua tela[cite: 1].
    a = r['avaliacao']
    t = r['trafego_e_latencia']
    print('=' * 72)
    print('NEXUS SAFE v2 - RESULTADOS')
    print('=' * 72)
    print(f"Leituras: {r['total_leituras']} | Alertas: {r['distribuicao_alertas']}")
    print('\nDesempenho (rótulos reais do gerador, derivas excluídas):')
    for nome, m in a['geral'].items():
        print(f"  {nome:34} P={m['precisao']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    print('\nRecall por cenário:')
    for nome, d in a['recall_por_cenario'].items():
        print(f"  {nome:34} " + '  '.join(f'{c}={v:.2f}' for c, v in d.items()))
    p = a['predicao']
    print(f"\nPredição: antecedência média {p['antecedencia_media_min']} min "
          f"({p['derivas_com_antecedencia']} derivas) | falso alarme preditivo em normais: "
          f"{p['falso_alarme_preditivo_em_normais_pct']}%")
    print(f"Risco emergente: {a['casos_risco_emergente']} casos "
          f"({a['casos_emergentes_que_regra_isolada_perdeu']} perdidos pela regra isolada)")
    print(f"\nBorda: {t['mensagens_legitimas']} mensagens para {t['leituras_brutas']} leituras "
          f"(redução {t['reducao_mensagens_pct']}%) | latência média {t['latencia_borda_media_ms']} ms "
          f"(p95 {t['latencia_borda_p95_ms']} ms)")
    print(f"Segurança: {r['seguranca_mensagens']}")
    g = r['gemeo_digital_what_if']
    print(f"Gêmeo digital (what-if): risco {g['risco_antes']:.2f} -> {g['risco_depois']:.2f} "
          f"(ganho por correlação {g['ganho_correlacao']:.2f})")
    print('=' * 72)


if __name__ == '__main__':
    ritmo = 0.05 if '--tempo-real' in sys.argv else 0.0
    executar(ritmo_s=ritmo)